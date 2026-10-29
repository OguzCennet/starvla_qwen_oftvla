"""Offline action error of saved checkpoints on held-out demos vs. training demos.

Same metric as training (mean L1 on normalised 8x7 chunks), computed at fixed weights on
  * val:   demos of e3_val_ids.csv (training tasks, demos never trained on)
  * train: an equally sized, fixed random subset of e3_train_ids.csv demos
A growing val - train gap across checkpoints indicates memorisation of the training demos.
Also reported: L1 of the arm dims, gripper accuracy (threshold 0.5), and a trivial baseline
that always predicts the training mean action (shows what "no learning" scores).

Usage: python -m oftvla.offline_eval --run_dir RUN --ckpts 10000 20000 30000 ...
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from oftvla.data import LiberoLangDataset, identity_collate, load_stats, normalize_actions
from oftvla.model import OFTConfig, QwenOFT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ckpts", type=int, nargs="+", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--train_ids", required=True)
    p.add_argument("--val_ids", required=True)
    p.add_argument("--forbidden_ids", nargs="*", default=[])
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--base_vlm", required=True)
    p.add_argument("--stride", type=int, default=4, help="use every k-th sample start (consecutive starts overlap)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def demo_subset(ds: LiberoLangDataset, n_demos: int, seed: int) -> None:
    """Keep only n_demos randomly chosen demos (in place), so train and val have the same size."""
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(len(ds.demos), size=min(n_demos, len(ds.demos)), replace=False).tolist())
    ds.starts = [(di, t) for di, t in ds.starts if di in keep]


@torch.inference_mode()
def score(model: QwenOFT, ds, stride: int, batch_size: int, num_workers: int, mean_action: np.ndarray) -> dict:
    idx = list(range(0, len(ds), stride))
    loader = DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=identity_collate)
    abs_err, base_err, grip_ok = [], [], []
    for batch in loader:
        pred = model.predict_action(batch)                                  # [B, 8, 7] normalised
        gt = np.stack([s["action"] for s in batch]).astype(np.float32)      # [B, 8, 7] normalised
        abs_err.append(np.abs(pred - gt))
        base_err.append(np.abs(mean_action[None, None, :] - gt))
        grip_ok.append((pred[..., 6] > 0.5) == (gt[..., 6] > 0.5))
    e, b, g = np.concatenate(abs_err), np.concatenate(base_err), np.concatenate(grip_ok)
    return {
        "n_samples": int(len(e)),
        "l1": float(e.mean()),
        "l1_arm": float(e[..., :6].mean()),
        "l1_per_dim": e.mean(axis=(0, 1)).round(4).tolist(),
        "l1_first_step": float(e[:, 0].mean()),       # the action executed first
        "l1_last_step": float(e[:, -1].mean()),       # furthest into the future
        "gripper_acc": float(g.mean()),
        "baseline_mean_action_l1": float(b.mean()),
    }


def main():
    args = parse_args()
    stats = load_stats(os.path.join(args.run_dir, "dataset_statistics.json"))  # training statistics
    common = dict(labels_csv=args.labels_csv, stats=stats, forbidden_id_files=args.forbidden_ids)
    val = LiberoLangDataset(args.data_root, [args.val_ids], **common)
    train = LiberoLangDataset(args.data_root, [args.train_ids], **common)
    demo_subset(train, len(val.demos), args.seed)
    print(f"[data] val demos={len(val.demos)} starts={len(val)} | train subset starts={len(train)}", flush=True)

    # Training-mean action in normalised units (arm dims via min_max, gripper raw)
    mean_action = normalize_actions(np.asarray(stats["mean"], np.float32)[None], stats)[0]

    model = QwenOFT(OFTConfig(base_vlm=args.base_vlm)).to("cuda").eval()  # bf16 weights, as at inference
    out_path = os.path.join(args.run_dir, "offline_eval.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for step in args.ckpts:
        sd = torch.load(os.path.join(args.run_dir, f"steps_{step}_model.pt"), map_location="cpu")
        print(model.load_state_dict(sd), flush=True)
        del sd
        res = {name: score(model, ds, args.stride, args.batch_size, args.num_workers, mean_action)
               for name, ds in (("train", train), ("val", val))}
        res["gap_val_minus_train"] = res["val"]["l1"] - res["train"]["l1"]
        results[str(step)] = res
        print(json.dumps({"step": step, **res}), flush=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print("\nstep    train_l1  val_l1   gap      val_grip_acc  baseline", flush=True)
    for step in args.ckpts:
        r = results[str(step)]
        print(f"{step:<7} {r['train']['l1']:.4f}    {r['val']['l1']:.4f}   {r['gap_val_minus_train']:+.4f}  "
              f"{r['val']['gripper_acc']:.3f}         {r['val']['baseline_mean_action_l1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
