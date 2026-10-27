"""Equivalence check: oftvla.QwenOFT vs. StarVLA's Qwenvl_OFT (same weights, same inputs).

Passes if (1) tokenised inputs are identical, (2) predicted normalised actions match,
(3) training losses match. Run on a GPU node inside starvla.sif, e.g.:

  apptainer exec --nv --env PYTHONPATH=$HOME/VLA_FT/oftvla $SIF \
      python scripts/check_equivalence.py --starvla_root $HOME/VLA_FT/starVLA \
      --base_vlm /path/to/Qwen3-VL-4B-Instruct
"""

import argparse
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from oftvla.model import OFTConfig, QwenOFT, build_prompt


def make_batch(seed: int, chunk_len: int, action_dim: int):
    """Two samples with different instruction lengths so left padding is exercised."""
    rng = np.random.default_rng(seed)
    batch = []
    for lang in ["pick up the black bowl", "open the top drawer of the cabinet and put the bowl inside"]:
        imgs = [Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(2)]
        act = rng.uniform(-1, 1, (chunk_len, action_dim)).astype(np.float32)
        batch.append({"image": imgs, "lang": lang, "action": act})
    return batch


def build_reference(starvla_root: str, base_vlm: str, chunk_len: int, action_dim: int):
    sys.path.insert(0, starvla_root)
    from starVLA.model.framework.share_tools import apply_config_compat
    from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT

    cfg = OmegaConf.create({
        "framework": {
            "name": "QwenOFT",
            "qwenvl": {"base_vlm": base_vlm, "attn_implementation": "sdpa"},
            "action_model": {"action_model_type": "MLP", "action_dim": action_dim, "action_horizon": chunk_len},
        },
        "datasets": {"vla_data": {"obs_image_size": [224, 224]}},  # no CoT_prompt -> raw instruction
    })
    cfg = apply_config_compat(cfg)
    return Qwenvl_OFT(cfg)


def report(name: str, a: torch.Tensor, b: torch.Tensor, atol: float) -> bool:
    diff = (a.float() - b.float()).abs().max().item()
    ok = diff <= atol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max|diff| = {diff:.3e} (atol {atol})")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--starvla_root", required=True)
    p.add_argument("--base_vlm", required=True)
    p.add_argument("--chunk_len", type=int, default=8)
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--atol", type=float, default=1e-4)
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")

    ref = build_reference(args.starvla_root, args.base_vlm, args.chunk_len, args.action_dim).to(device).eval()
    ours = QwenOFT(OFTConfig(base_vlm=args.base_vlm, chunk_len=args.chunk_len, action_dim=args.action_dim))
    print(ours.load_starvla_state_dict(ref.state_dict(), strict=True))  # identical weights incl. random head
    ours = ours.to(device).eval()

    batch = make_batch(seed=0, chunk_len=args.chunk_len, action_dim=args.action_dim)
    images = [s["image"] for s in batch]
    langs = [s["lang"] for s in batch]
    results = []

    # 1) Tokenisation / pixel inputs
    ref_in = ref.qwen_vl_interface.build_qwenvl_inputs(
        images=images, instructions=[build_prompt(l, args.chunk_len, ref.action_token) for l in langs]
    )
    our_in = ours.build_inputs(images, langs)
    same_ids = torch.equal(ref_in["input_ids"], our_in["input_ids"])
    print(f"[{'PASS' if same_ids else 'FAIL'}] input_ids identical, shape {tuple(our_in['input_ids'].shape)}")
    results.append(same_ids)
    results.append(report("pixel_values", ref_in["pixel_values"], our_in["pixel_values"], 0.0))
    n_q = (our_in["input_ids"] == ours.query_token_id).sum(1).tolist()
    print(f"       query tokens per sample: {n_q}, query id ours={ours.query_token_id} ref={ref.action_token_id}")
    results.append(ours.query_token_id == ref.action_token_id)

    # 2) Inference
    ref_pred = torch.as_tensor(ref.predict_action(examples=batch)["normalized_actions"])
    our_pred = torch.as_tensor(ours.predict_action(batch))
    results.append(report("predict_action", ref_pred, our_pred, args.atol))

    # 3) Training loss (eval mode -> no dropout; gradients not needed for the comparison)
    with torch.no_grad():
        ref_loss = ref(batch)["action_loss"]
        our_loss = ours(batch)["action_loss"]
    results.append(report("action_loss", ref_loss, our_loss, args.atol))

    print("\nALL PASS" if all(results) else "\nSOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
