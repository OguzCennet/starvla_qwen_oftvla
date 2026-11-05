"""Flow-matching action-head components ported from StarVLA @ 985c223 (GR00T and PI heads).

Files and their fork sources:
  dit.py        <- flow_matching_head/cross_attention_dit.py:1-333 (TimestepEncoder, AdaLayerNorm,
                   BasicTransformerBlock, DiT), byte-identical code body
  encoders.py   <- flow_matching_head/action_encoder.py:20-54; GR00T_ActionHeader.py:52-101
  gr00t_head.py <- GR00T_ActionHeader.py:190-429 (FlowmatchingActionHead)
  pi_head.py    <- LayerwiseFM_ActionHeader.py:189-408 (LayerwiseFlowmatchingActionHead)
  config.py     <- effective action_model configs; share_tools.py:261-295; the repeat step of
                   QwenGR00T.py:204-214 / QwenPI.py:225-236
Library calls are the fork's (diffusers Attention / FeedForward / Timesteps / TimestepEmbedding,
ModelMixin / ConfigMixin, torch.distributions.Beta).

Changes vs the fork code (everything else is verbatim):
  C1  Heads take the ``action_model`` config node (an object with the fork's field names) instead of the
      full OmegaConf config (fork: ``full_config.framework.action_model``); the GR00T head no longer
      stores the unused ``self.full_config``. Reason: no OmegaConf / framework machinery in oftvla.
  C2  ``forward`` takes optional ``noise`` and ``t``; ``predict_action`` takes optional ``init_noise``.
      When None, the fork's sampling code runs unchanged and in the same order (randn, then Beta).
      Reason: bitwise equivalence tests need injected randomness.
  C3  One ``MLP`` / ``ActionEncoder`` (GR00T copy, no default arguments) for both heads. The PI copies
      have identical bodies and only defaults (MLP hidden_dim=1024, output_dim=2048; ActionEncoder
      hidden_size=1024); the PI head passes the value it relied on (hidden_dim=1024) explicitly.
  C4  Not ported (unused by QwenGR00T / QwenPI on LIBERO): ``SelfAttentionTransformer``,
      ``prepare_input`` (BatchFeature wrapper), PI ``predict_action_realtime`` (RTC),
      ``FlowmatchingActionHeadConfig`` dataclasses, ``CategorySpecific*`` / ``MultiEmbodimentActionEncoder``.
  C5  Framework glue (config population, the repeat for repeated diffusion steps) as plain functions in
      config.py instead of inside the framework classes. Same operations, same order.
"""

from oftvla.heads.flow_matching.config import (
    GR00T_REPEATED_DIFFUSION_STEPS,
    PI_REPEATED_DIFFUSION_STEPS,
    gr00t_action_config,
    pi_action_config,
    populate_layerwise_dit_cfg,
    repeat_for_diffusion,
)
from oftvla.heads.flow_matching.dit import DiT
from oftvla.heads.flow_matching.gr00t_head import FlowmatchingActionHead
from oftvla.heads.flow_matching.pi_head import LayerwiseFlowmatchingActionHead
