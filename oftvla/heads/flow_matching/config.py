"""Action-model configs and framework glue for the flow-matching heads (change C5: plain functions instead
of the fork's framework classes / OmegaConf merge).

Values are the fork's EFFECTIVE values for LIBERO at 985c223: framework defaults (QwenGR00T.py:78-120,
QwenPI.py:69-104) merged with starvla_cotrain_libero.yaml:15-18 (action_dim 7, state_dim 7, action_horizon 8)
and the runtime overrides each framework applies in __init__ (docs/briefs/flow_heads_spec.md §2).
"""

import copy
from types import SimpleNamespace
from typing import Optional

import torch

# QwenGR00T.py:97 default, read at QwenGR00T.py:199-203 (the LIBERO yaml does not override it)
GR00T_REPEATED_DIFFUSION_STEPS = 8
# QwenPI.py:224: hard-coded, whatever the config says
PI_REPEATED_DIFFUSION_STEPS = 2

# QwenGR00T.py:78-120 (action_model defaults) + LIBERO yaml
_GR00T_ACTION_MODEL = {
    "action_model_type": "DiT-B",
    "action_hidden_dim": 1024,
    "hidden_size": 1024,
    "add_pos_embed": True,
    "max_seq_len": 1024,
    "action_dim": 7,
    "state_dim": 7,
    "action_horizon": 8,
    "repeated_diffusion_steps": GR00T_REPEATED_DIFFUSION_STEPS,
    "noise_beta_alpha": 1.5,
    "noise_beta_beta": 1.0,
    "noise_s": 0.999,
    "num_timestep_buckets": 1000,
    "num_inference_timesteps": 4,
    "num_target_vision_tokens": 32,
    "diffusion_model_cfg": {
        "cross_attention_dim": 2048,  # replaced by the VLM hidden size (QwenGR00T.py:155-157)
        "dropout": 0.2,
        "final_dropout": True,
        "interleave_self_attention": True,
        "norm_type": "ada_norm",
        "num_layers": 16,
        "output_dim": 1024,
        "positional_embeddings": None,
    },
}

# QwenPI.py:69-104 (action_model defaults) + LIBERO yaml (action_horizon 8 overrides the default 16)
_PI_ACTION_MODEL = {
    "action_model_type": "LayerwiseFM",
    "action_dim": 7,
    "state_dim": 7,
    "action_horizon": 8,
    "repeated_diffusion_steps": 2,
    "num_inference_timesteps": 4,
    "add_pos_embed": True,
    "max_seq_len": 1024,
    "num_target_vision_tokens": 32,
    "noise_beta_alpha": 1.5,
    "noise_beta_beta": 1.0,
    "noise_s": 0.999,
    "num_timestep_buckets": 1000,
    "diffusion_model_cfg": {
        "dropout": 0.2,
        "final_dropout": True,
        "interleave_self_attention": True,
        "norm_type": "ada_norm",
        "positional_embeddings": None,
        "attention_head_dim": 64,
    },
}


def _ns(d: dict, **overrides) -> SimpleNamespace:
    d = copy.deepcopy(d)
    dm = overrides.pop("diffusion_model_cfg", None)
    d.update(overrides)
    if dm:
        d["diffusion_model_cfg"].update(dm)
    return SimpleNamespace(**d)


def gr00t_action_config(vlm_hidden_size: int, **overrides) -> SimpleNamespace:
    """QwenGR00T's action_model config with cross_attention_dim = VLM hidden size (QwenGR00T.py:155-157)."""
    cfg = _ns(_GR00T_ACTION_MODEL, **overrides)
    cfg.diffusion_model_cfg["cross_attention_dim"] = int(vlm_hidden_size)
    return cfg


def populate_layerwise_dit_cfg(dit_cfg: dict, *, dit_hidden_dim: int, num_dit_layers: int) -> dict:
    """fork: share_tools.py:261-295 (same fields, same order)."""
    head_dim = dit_cfg.get("attention_head_dim", None) or 64
    dit_cfg["attention_head_dim"] = head_dim
    dit_cfg["num_layers"] = int(num_dit_layers)
    dit_cfg["input_embedding_dim"] = int(dit_hidden_dim)
    dit_cfg["cross_attention_dim"] = int(dit_hidden_dim)
    dit_cfg["num_attention_heads"] = int(dit_hidden_dim) // int(head_dim)
    return dit_cfg


def pi_action_config(vlm_hidden_size: int, vlm_num_layers: int, use_canonical_forward: Optional[bool] = False,
                     **overrides) -> SimpleNamespace:
    """QwenPI's action_model config: DiT width = VLM hidden size, depth = number of VLM layers (QwenPI.py:144-157).

    use_canonical_forward=False is the fork's official legacy switch (run_policy_server.sh:29 sets
    framework.action_model.diffusion_model_cfg.use_canonical_forward; DiT maps it onto
    interleave_self_attention, cross_attention_dit.py:223-225): every block cross-attends (decision 11.1).
    None leaves the key out (canonical interleaving from the defaults).
    """
    cfg = _ns(_PI_ACTION_MODEL, **overrides)
    populate_layerwise_dit_cfg(cfg.diffusion_model_cfg, dit_hidden_dim=vlm_hidden_size, num_dit_layers=vlm_num_layers)
    if use_canonical_forward is not None:
        cfg.diffusion_model_cfg["use_canonical_forward"] = bool(use_canonical_forward)
    return cfg


def repeat_for_diffusion(vl_embs, actions, attention_mask=None, state=None, repeats: int = 1):
    """Framework step before the head: QwenGR00T.py:204-214 / QwenPI.py:225-236.

    vl_embs: one tensor (GR00T) or a list of per-layer tensors (PI). Copy r of sample b becomes row r*B + b.
    The mask is repeated and cast to bool exactly as in the fork.
    """
    actions_rep = actions.repeat(repeats, 1, 1)
    if isinstance(vl_embs, (list, tuple)):
        vl_rep = [h.repeat(repeats, 1, 1) for h in vl_embs]
    else:
        vl_rep = vl_embs.repeat(repeats, 1, 1)
    mask_rep = attention_mask.repeat(repeats, 1).to(dtype=torch.bool) if attention_mask is not None else None
    state_rep = state.repeat(repeats, 1, 1) if state is not None else None
    return vl_rep, actions_rep, mask_rep, state_rep
