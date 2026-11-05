"""QwenOFT: Qwen3-VL backbone + parallel action queries + MLP L1 regression head.

Faithful re-implementation of StarVLA's ``QwenOFT`` framework
(starVLA/model/framework/VLM4A/QwenOFT.py @ 985c223, MIT licence), stripped
of the registry/config machinery so it can be read and modified in one file.

Data flow (identical for training and inference):
    images + instruction + " ... <action>🔍×N<action>."
      -> Qwen3-VL (one forward pass, causal attention)
      -> last-layer hidden states at the N 🔍 positions      [B, N, H]
      -> shared MLPResNet applied to each position           [B, N, action_dim]
      -> L1 loss vs. normalised action chunk (training) / returned (inference)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from oftvla.vlm import get_vlm_adapter


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
@dataclass
class OFTConfig:
    # Backbone. StarVLA's default is the "-Action" checkpoint; plain Instruct also loads.
    base_vlm: str = "Qwen/Qwen3-VL-4B-Instruct"
    # StarVLA's QWen3.py hard-codes "sdpa" regardless of the yaml value; we keep that default.
    attn_implementation: str = "sdpa"
    action_dim: int = 7
    chunk_len: int = 8
    query_token: str = "🔍"
    # Images are resized to this (W, H) at inference; StarVLA reads datasets.vla_data.obs_image_size.
    obs_image_size: Optional[Tuple[int, int]] = (224, 224)
    num_resnet_blocks: int = 2
    # Storage dtype of the VLM weights. bf16 for inference / equivalence check;
    # float32 for training (fp32 master weights, bf16 compute via autocast).
    vlm_dtype: str = "bfloat16"


# ──────────────────────────────────────────────────────────────────────
# Action head (adapted from openvla-oft prismatic/models/action_heads.py)
# Parameter names match StarVLA so its state_dict loads 1:1.
# ──────────────────────────────────────────────────────────────────────
class MLPResNetBlock(nn.Module):
    """Pre-LN residual block: x + ReLU(Linear(LN(x)))."""

    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(x)


class MLPResNet(nn.Module):
    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([MLPResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for block in self.mlp_resnet_blocks:
            x = block(x)
        return self.fc2(self.layer_norm2(x))


class L1RegressionActionHead(nn.Module):
    """Maps each query hidden state independently to one action (weights shared over time)."""

    def __init__(self, input_dim: int, action_dim: int, num_blocks: int = 2):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(num_blocks, input_dim, hidden_dim=2 * input_dim, output_dim=action_dim)

    def forward(self, query_states: torch.Tensor) -> torch.Tensor:
        b, n, h = query_states.shape
        return self.model(query_states.reshape(b * n, h)).view(b, n, self.action_dim)


# ──────────────────────────────────────────────────────────────────────
# Pure functions (unit-testable without the VLM)
# ──────────────────────────────────────────────────────────────────────
def gather_query_states(hidden: torch.Tensor, input_ids: torch.Tensor, token_id: int, n: int) -> torch.Tensor:
    """Return hidden states at the LAST ``n`` occurrences of ``token_id`` per sample, in sequence order.

    hidden: [B, L, H], input_ids: [B, L]  ->  [B, n, H]
    Works with left padding because positions are found per row.
    """
    b, l, h = hidden.shape
    mask = input_ids == token_id
    counts = mask.sum(dim=1)
    if (counts < n).any():
        raise RuntimeError(f"Samples with < {n} query tokens: counts={counts.tolist()}")
    pos = torch.arange(l, device=input_ids.device).expand(b, l)
    pos = torch.where(mask, pos, torch.full_like(pos, -1))
    selected = pos.topk(k=n, dim=-1).values.sort(dim=-1).values  # [B, n]
    return hidden.gather(1, selected.unsqueeze(-1).expand(-1, -1, h))


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean |pred - target| over valid cells (all cells if mask is None)."""
    err = (pred - target).abs()
    if mask is None:
        return err.mean()
    if mask.shape != err.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} != action shape {tuple(err.shape)}")
    if not mask.any():
        raise ValueError("mask contains no valid targets")
    valid = mask.to(err.dtype)
    return (err * valid).sum() / valid.sum()


def build_prompt(instruction: str, chunk_len: int, query_token: str) -> str:
    # Query tokens must be contiguous: a space between them changes the tokenisation.
    queries = query_token * chunk_len
    return f"{instruction} Please predict the next {chunk_len} robot actions: <action>{queries}<action>."


def resize_images(images, size: Tuple[int, int]):
    """Recursively resize PIL images in a nested list (same as StarVLA's helper)."""
    if isinstance(images, Image.Image):
        return images.resize(size)
    if isinstance(images, (list, tuple)):
        return [resize_images(im, size) for im in images]
    raise TypeError(f"Unsupported image type: {type(images)}")


def to_pil(images):
    """Recursively convert HWC numpy arrays (uint8, or float in [0,1]) to PIL; PIL passes through."""
    if isinstance(images, Image.Image):
        return images
    if isinstance(images, (list, tuple)):
        return [to_pil(im) for im in images]
    if isinstance(images, np.ndarray):
        arr = images
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))
    raise TypeError(f"Unsupported image type: {type(images)}")


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────
class QwenOFT(nn.Module):
    def __init__(self, cfg: OFTConfig):
        super().__init__()
        self.cfg = cfg
        # The adapter is a plain object; its HF model is registered here as ``self.vlm`` so the
        # state_dict keys (vlm.*, action_head.*) and all saved checkpoints stay unchanged.
        self.adapter = get_vlm_adapter(cfg.base_vlm, attn_implementation=cfg.attn_implementation, dtype=cfg.vlm_dtype)
        self.vlm = self.adapter.model
        self.processor = self.adapter.processor

        self.action_head = L1RegressionActionHead(self.adapter.hidden_size, cfg.action_dim, cfg.num_resnet_blocks)
        # Must be a single token (StarVLA silently takes ids[0]; the adapter refuses).
        self.query_token_id = self.adapter.token_id(cfg.query_token)

    # --- inputs ---------------------------------------------------------
    def build_inputs(self, images: Sequence[List[Image.Image]], instructions: Sequence[str]) -> Dict[str, torch.Tensor]:
        """One user turn per sample: [img_0][img_1]...[text]; generation prompt appended.

        OFT keeps CoT_prompt OFF (cot_prompt=None): the finished OFT runs were trained without it
        (known deviation D3, docs/briefs/flow_heads_spec.md), and this keeps them reproducible.
        """
        texts = [build_prompt(instr, self.cfg.chunk_len, self.cfg.query_token) for instr in instructions]
        return self.adapter.build_inputs(images, texts, cot_prompt=None)

    # --- core -----------------------------------------------------------
    def _predict_normalized(self, images, instructions) -> torch.Tensor:
        inputs = self.build_inputs(images, instructions)
        hidden_states, _ = self.adapter.forward(inputs)  # bf16 autocast inside, as before
        last_hidden = hidden_states[-1]  # [B, L, H]
        queries = gather_query_states(last_hidden, inputs["input_ids"], self.query_token_id, self.cfg.chunk_len)
        # Head runs in fp32 (StarVLA: autocast float32 around the head).
        with torch.autocast("cuda", enabled=False):
            return self.action_head(queries.float())  # [B, N, action_dim]

    def forward(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        """Training. Each sample: {image: [PIL,...], lang: str, action: [T, D] normalised, action_mask?: [T, D]}."""
        pred = self._predict_normalized([s["image"] for s in batch], [s["lang"] for s in batch])
        n = self.cfg.chunk_len
        target = torch.as_tensor(np.stack([s["action"] for s in batch]), device=pred.device, dtype=pred.dtype)[:, -n:]
        has_mask = [s.get("action_mask") is not None for s in batch]
        if any(has_mask) and not all(has_mask):
            raise ValueError("action_mask must be present for all samples in a batch or none")
        mask = None
        if all(has_mask):
            mask = torch.as_tensor(np.stack([s["action_mask"] for s in batch]), device=pred.device, dtype=torch.bool)[:, -n:]
        return {"action_loss": masked_l1_loss(pred, target, mask)}

    @torch.inference_mode()
    def predict_action(self, batch: List[dict]) -> np.ndarray:
        """Inference. Returns NORMALISED actions [B, N, action_dim]; unnormalisation is the policy's job."""
        images = [to_pil(s["image"]) for s in batch]
        if self.cfg.obs_image_size:
            images = resize_images(images, tuple(self.cfg.obs_image_size))
        pred = self._predict_normalized(images, [s["lang"] for s in batch])
        return pred.cpu().numpy()

    # --- interop --------------------------------------------------------
    def load_starvla_state_dict(self, sd: Dict[str, torch.Tensor], strict: bool = True):
        """Load weights from a StarVLA QwenOFT checkpoint (prefix mapping only)."""
        remap = {"qwen_vl_interface.model.": "vlm.", "action_model.": "action_head."}
        new_sd = {}
        for k, v in sd.items():
            for old, new in remap.items():
                if k.startswith(old):
                    new_sd[new + k[len(old):]] = v
                    break
            else:
                new_sd[k] = v
        return self.load_state_dict(new_sd, strict=strict)
