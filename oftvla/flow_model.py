"""QwenGR00T and QwenPI: VLM adapter + flow-matching action head.

Same forward / predict_action structure as StarVLA @ 985c223:
  QwenGR00T  <- starVLA/model/framework/VLM4A/QwenGR00T.py:150-278
  QwenPI     <- starVLA/model/framework/VLM4A/QwenPI.py:121-295 (predict_action_realtime not ported)
with the decisions of docs/briefs/flow_heads_spec.md §11:
  * CoT_prompt ON: the whole per-frame text (text_A0 / text_AS / text_MAS) replaces {instruction}
    (QWen3.py:126-128; the wrapper is starvla_cotrain_libero.yaml:40);
  * robot state OFF: the LIBERO loader packs no state (datasets.py:1420), so the heads get state=None;
    the (unused) state encoders exist, as in the fork, so state_dicts match;
  * VLM attention: "sdpa" for both heads. The fork's Qwen3-VL wrapper forces it for every framework
    (QWen3.py:52-53 @ 985c223, overriding the yaml's flash_attention_2);
  * PI: the 985c223 legacy switch (diffusion_model_cfg.use_canonical_forward=False): every DiT block
    cross-attends to its own VLM layer, padding mask applied;
  * repeats: GR00T from config (8, QwenGR00T.py:97/199-203); PI hard-coded 2 (QwenPI.py:224);
  * action horizon 8 (starvla_cotrain_libero.yaml:18; PI's code default 16 is overridden), = our chunk of 8.
Numerics: the VLM runs under bf16 autocast; the head under ``torch.autocast("cuda", dtype=torch.float32)``
with the action tensor created in the hidden-state dtype, exactly as the fork does (spec §4). With bf16
weights (DeepSpeed bf16, decision 11.5) noise, t and the timestep bucketing are therefore bf16.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from oftvla.heads.flow_matching import (
    PI_REPEATED_DIFFUSION_STEPS,
    FlowmatchingActionHead,
    LayerwiseFlowmatchingActionHead,
    gr00t_action_config,
    pi_action_config,
    repeat_for_diffusion,
)
from oftvla.model import resize_images, to_pil
from oftvla.vlm import get_vlm_adapter

# starvla_cotrain_libero.yaml:40 @ 985c223 (also in both released Qwen3-VL LIBERO checkpoint configs)
COT_PROMPT = (
    "Your task is {instruction}. To identify the key objects for your task. "
    "Locate their bounding boxes in [x1,y1,x2,y2] format."
)


@dataclass
class FlowConfig:
    head: str  # "gr00t" | "pi"
    base_vlm: str = "Qwen/Qwen3-VL-4B-Instruct"
    attn_implementation: str = "sdpa"  # QWen3.py:52-53 forces sdpa for all frameworks
    vlm_dtype: str = "bfloat16"  # QWen3.py:63
    cot_prompt: Optional[str] = COT_PROMPT  # None = wrapper off (negative control / OFT-style prompt)
    # The LIBERO yaml sets no obs_image_size -> predict_action does not resize (QwenGR00T.py:245-247);
    # our loader already delivers 224x224 (datasets.py:1392-1404 equivalent).
    obs_image_size: Optional[Tuple[int, int]] = None
    pi_use_canonical_forward: Optional[bool] = False  # decision 11.1: legacy switch


def remap_starvla_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Fork key prefixes -> ours (same mapping as QwenOFT.load_starvla_state_dict)."""
    remap = {"qwen_vl_interface.model.": "vlm.", "action_model.": "action_head."}
    out = {}
    for k, v in sd.items():
        for old, new in remap.items():
            if k.startswith(old):
                out[new + k[len(old):]] = v
                break
        else:
            out[k] = v
    return out


class _QwenFlowBase(nn.Module):
    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.cfg = cfg
        self.adapter = get_vlm_adapter(cfg.base_vlm, attn_implementation=cfg.attn_implementation, dtype=cfg.vlm_dtype)
        self.vlm = self.adapter.model  # registered here -> state_dict keys vlm.*, action_head.*
        self.processor = self.adapter.processor

    def _vlm_forward(self, images, texts):
        """QWen3.py:114-171 + QwenGR00T.py:180-190 / QwenPI.py:171-183: inputs, mask, all hidden states."""
        inputs = self.adapter.build_inputs(images, texts, cot_prompt=self.cfg.cot_prompt)
        backbone_attention_mask = inputs.get("attention_mask", None)
        hidden_states, _ = self.adapter.forward(inputs)  # bf16 autocast inside
        return hidden_states, backbone_attention_mask

    def _prepare_images(self, batch):
        images = [to_pil(s["image"]) for s in batch]  # = to_pil_preserve for PIL / uint8 HWC
        if self.cfg.obs_image_size:
            images = resize_images(images, tuple(self.cfg.obs_image_size))
        return images

    def load_starvla_state_dict(self, sd: Dict[str, torch.Tensor], strict: bool = True):
        return self.load_state_dict(remap_starvla_state_dict(sd), strict=strict)


class QwenGR00T(_QwenFlowBase):
    def __init__(self, cfg: FlowConfig):
        super().__init__(cfg)
        # QwenGR00T.py:155-159: cross_attention_dim = VLM hidden size
        self.action_config = gr00t_action_config(self.adapter.hidden_size)
        self.action_head = FlowmatchingActionHead(self.action_config)
        self.action_horizon = int(self.action_config.action_horizon)  # QwenGR00T.py:165
        self.repeated_diffusion_steps = self.action_config.repeated_diffusion_steps  # QwenGR00T.py:199-203

    def forward(self, batch: List[dict], noise: torch.Tensor = None, t: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """QwenGR00T.py:167-221. Each sample: {image: [PIL, PIL], lang: str, action: [8, 7] normalised}."""
        images = [s["image"] for s in batch]
        texts = [s["lang"] for s in batch]
        actions = [s["action"] for s in batch]
        hidden_states, backbone_attention_mask = self._vlm_forward(images, texts)
        last_hidden = hidden_states[-1]  # [B, L, H]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]
            vl, act, mask, _ = repeat_for_diffusion(
                last_hidden, actions_target, backbone_attention_mask, None, self.repeated_diffusion_steps
            )
            action_loss = self.action_head(vl, act, None, encoder_attention_mask=mask, noise=noise, t=t)
        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, batch: List[dict], init_noise: torch.Tensor = None) -> np.ndarray:
        """QwenGR00T.py:223-278. Returns NORMALISED actions [B, 8, 7]."""
        images = self._prepare_images(batch)
        hidden_states, backbone_attention_mask = self._vlm_forward(images, [s["lang"] for s in batch])
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
        last_hidden = hidden_states[-1]
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_head.predict_action(
                last_hidden, None, encoder_attention_mask=backbone_attention_mask, init_noise=init_noise
            )
        return pred_actions.detach().cpu().numpy()


class QwenPI(_QwenFlowBase):
    def __init__(self, cfg: FlowConfig):
        super().__init__(cfg)
        # QwenPI.py:144-159: DiT width = VLM hidden size, depth = number of VLM layers (from the config)
        self.action_config = pi_action_config(
            self.adapter.hidden_size, self.adapter.num_layers, use_canonical_forward=cfg.pi_use_canonical_forward
        )
        self.action_head = LayerwiseFlowmatchingActionHead(self.action_config)
        self.action_horizon = int(self.action_config.action_horizon)  # QwenPI.py:165
        self.repeated_diffusion_steps = PI_REPEATED_DIFFUSION_STEPS  # QwenPI.py:224 (hard-coded 2)

    def _encode_vl_hidden_states(self, images, texts):
        """QwenPI.py:167-184: the last len(DiT blocks) hidden states, one per DiT block."""
        hidden_states, attention_mask = self._vlm_forward(images, texts)
        expected_layers = len(self.action_head.model.transformer_blocks)
        return list(hidden_states[-expected_layers:]), attention_mask

    def forward(self, batch: List[dict], noise: torch.Tensor = None, t: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """QwenPI.py:186-245."""
        images = [s["image"] for s in batch]
        texts = [s["lang"] for s in batch]
        actions = [s["action"] for s in batch]
        vl_embs_list, backbone_attention_mask = self._encode_vl_hidden_states(images, texts)
        base_hidden = vl_embs_list[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]
            vl, act, mask, _ = repeat_for_diffusion(
                vl_embs_list, actions_target, backbone_attention_mask, None, self.repeated_diffusion_steps
            )
            action_loss = self.action_head(vl, act, None, encoder_attention_mask=mask, noise=noise, t=t)
        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, batch: List[dict], init_noise: torch.Tensor = None) -> np.ndarray:
        """QwenPI.py:247-295. Returns NORMALISED actions [B, 8, 7]."""
        images = self._prepare_images(batch)
        vl_embs_list, backbone_attention_mask = self._encode_vl_hidden_states(images, [s["lang"] for s in batch])
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_head.predict_action(
                vl_embs_list, None, encoder_attention_mask=backbone_attention_mask, init_noise=init_noise
            )
        return pred_actions.detach().cpu().numpy()


def build_flow_model(cfg: FlowConfig) -> nn.Module:
    return {"gr00t": QwenGR00T, "pi": QwenPI}[cfg.head](cfg)
