# Ported from StarVLA @ 985c223 (NVIDIA GR00T code, Apache-2.0):
#   starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py:189-408
#   (DiTConfig fallback, LayerwiseFlowmatchingActionHead without predict_action_realtime)
# Verbatim except the changes listed in oftvla/heads/flow_matching/__init__.py (C1-C5).
# Mode used by oftvla (decision 11.1, the 985c223 legacy switch): diffusion_model_cfg has
# interleave_self_attention=False, so every DiT block cross-attends to its own VLM layer, and the
# padding mask is passed through (QwenPI.py:228-243 @ 985c223 passes encoder_attention_mask).

import torch
from torch import nn
from torch.distributions import Beta

from oftvla.heads.flow_matching.dit import DiT
from oftvla.heads.flow_matching.encoders import MLP, ActionEncoder

# fork: LayerwiseFM_ActionHeader.py:189-194 (fallback only; the framework populates the real shape)
DiTConfig = {
    "num_layers": 36,
    "input_embedding_dim": 2048,
    "attention_head_dim": 64,
    "num_attention_heads": 32,
}  # default for qwen2.5-vl


# fork: LayerwiseFM_ActionHeader.py:197-408
class LayerwiseFlowmatchingActionHead(nn.Module):
    def __init__(self, action_config, **kwargs):
        """action_config: the fork's ``framework.action_model`` node as an object with the same field names,
        with diffusion_model_cfg already populated (see config.populate_layerwise_dit_cfg). Change C1: the
        fork receives the full config (``global_config.framework.action_model``, :223)."""
        super().__init__()
        diffusion_model_cfg = action_config.diffusion_model_cfg

        # fork :231-243: trust diffusion_model_cfg; DiTConfig only as fallback for missing keys.
        for k, v in DiTConfig.items():
            if diffusion_model_cfg.get(k, None) is None:
                diffusion_model_cfg[k] = v

        _DIT_NON_KWARGS = {"action_dit_hidden_dim"}
        diffusion_model_cfg_kwargs = {k: v for k, v in diffusion_model_cfg.items() if k not in _DIT_NON_KWARGS}

        self.input_embedding_dim = diffusion_model_cfg_kwargs["input_embedding_dim"]
        self.model = DiT(**diffusion_model_cfg_kwargs)  # TODO: ideally copy LLM init from VLM
        self.dit_out_hidden_size = self.input_embedding_dim
        self.action_dim = action_config.action_dim
        self.action_horizon = int(action_config.action_horizon)
        self.num_inference_timesteps = action_config.num_inference_timesteps

        # fork :252-259. Change C3: the fork's PI-local MLP has defaults hidden_dim=1024 (used here);
        # passed explicitly because the shared MLP has no defaults.
        self.state_encoder = (
            MLP(
                input_dim=action_config.state_dim,
                hidden_dim=1024,
                output_dim=self.input_embedding_dim,
            )
            if action_config.state_dim
            else None
        )

        self.action_encoder = ActionEncoder(
            action_dim=action_config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.input_embedding_dim,
            hidden_dim=1024,
            output_dim=self.action_dim,
        )
        # fork :270-279
        self.future_tokens = nn.Embedding(action_config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(action_config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config

    # fork :281-283 (PI does NOT clamp: t can be slightly negative; kept, decision 11.7)
    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    # fork :288-347. Change C2: optional ``noise`` / ``t`` instead of sampling (None = fork sampling).
    def forward(
        self,
        vl_embs_list: list,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        noise: torch.Tensor = None,
        t: torch.Tensor = None,
    ):
        """vl_embs_list: list of (B, seq_length, feature_dim); actions: (B, action_horizon, D_action);
        encoder_attention_mask: optional (B, seq_length) bool/int"""
        device = actions.device
        num_layers = len(vl_embs_list)  # noqa: F841 (fork)
        B, L, D = vl_embs_list[0].shape
        # Embed noised action trajectory.
        if noise is None:
            noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        if t is None:
            t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # Embed state
        state_features = self.state_encoder(state) if state is not None else None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        # Layer-wise DiT forward. DiT handles cross/self-attention interleaving.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs_list,
            timestep=t_discretized,
            encoder_attention_mask=encoder_attention_mask,
            return_pre_output=True,
        )

        # Decode only the action-token positions.
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    # fork :349-408 (Euler sampler). Change C2: optional ``init_noise``.
    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        init_noise: torch.Tensor = None,
    ) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        if init_noise is None:
            actions = torch.randn(
                size=(batch_size, self.action_horizon, self.action_dim),
                dtype=vl_embs_list[0].dtype,
                device=device,
            )
        else:
            actions = init_noise

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            # Embed current action trajectory with timestep
            action_features = self.action_encoder(actions, timesteps_tensor)

            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            # Layer-wise DiT forward. DiT handles cross/self-attention interleaving.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs_list,
                timestep=timesteps_tensor,
                encoder_attention_mask=encoder_attention_mask,
                return_pre_output=True,
            )
            # Decode only the action-token positions.
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]

            # Euler integration
            actions = actions + dt * pred_velocity
        return actions
