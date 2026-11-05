# Ported from StarVLA @ 985c223 (NVIDIA GR00T code, Apache-2.0):
#   starVLA/model/modules/action_model/GR00T_ActionHeader.py:190-429 (DiTConfig, FlowmatchingActionHead)
# Verbatim except the changes listed in oftvla/heads/flow_matching/__init__.py (C1, C2, C4, C5).

import torch
from torch import nn
from torch.distributions import Beta

from oftvla.heads.flow_matching.dit import DiT
from oftvla.heads.flow_matching.encoders import MLP, ActionEncoder

# fork: GR00T_ActionHeader.py:190-193
DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


# fork: GR00T_ActionHeader.py:196-429
class FlowmatchingActionHead(nn.Module):
    def __init__(self, action_config):
        """action_config: the fork's ``framework.action_model`` node as an object with the same field names
        (see oftvla.heads.flow_matching.config.gr00t_action_config). Change C1: the fork receives the full
        config and reads ``full_config.framework.action_model`` (GR00T_ActionHeader.py:202)."""
        super().__init__()
        config = action_config

        # DiT architecture selection (fork :217-223). diffusion_model_cfg.cross_attention_dim must already
        # be the VLM hidden size (the fork's QwenGR00T sets it before building the head, QwenGR00T.py:155-157).
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]

        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg)

        # fork :233-275
        self.action_horizon = int(config.action_horizon)
        self.action_dim = config.action_dim
        self.num_inference_timesteps = config.num_inference_timesteps
        self.hidden_size = config.hidden_size

        self.state_encoder = (
            MLP(
                input_dim=config.state_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            )
            if config.state_dim
            else None
        )

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # fork :282-292
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # fork :301-303
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

    # fork :305-307 (GR00T clamps the Beta sample at noise_s)
    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.config.noise_s)
        return (self.config.noise_s - sample) / self.config.noise_s

    # fork :312-363. Change C2: optional ``noise`` / ``t`` (shape (B,)) are used instead of sampling; when
    # None, the fork's sampling runs unchanged and in the same order (noise first, then t).
    def forward(
        self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None, encoder_attention_mask=None,
        noise: torch.Tensor = None, t: torch.Tensor = None,
    ):
        """vl_embs: (B, seq_length, feature_dim); actions: (B, action_horizon, action_dim)"""
        device = vl_embs.device

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

        # embed state
        state_features = self.state_encoder(state) if state is not None else None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        # Join VLM features with state and action embedding along sequence dimension.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    # fork :365-421 (Euler sampler). Change C2: optional ``init_noise`` replaces the initial randn.
    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        init_noise: torch.Tensor = None,
    ) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        if init_noise is None:
            actions = torch.randn(
                size=(batch_size, self.action_horizon, self.action_dim),
                dtype=vl_embs.dtype,
                device=device,
            )
        else:
            actions = init_noise

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return actions

    # fork :423-429
    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
