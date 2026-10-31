"""Offline action error on the held-out E3 test demos, with per-frame text (A0 / AS / MAS).

For one run and its saved checkpoints, every frame of every test demo (A0ASMAS test.csv) is scored:
the model gets the demo's two images and the SAME text column it was trained with, predicts the
8 x 7 chunk, and the prediction is compared with the demo's actions (normalised with the run's
training statistics; same metric as training: mean L1 over the chunk).

Outputs (in RUN/heldout_eval/):
  step{N}_frames.csv  one row per evaluated frame: suite, task, demo, t, l1, l1_arm, l1_first, grip_agree
  step{N}_pred.npz    predicted chunks (float16, [frames, 8, 7]) in the same row order, for later analyses
  summary.json        means per checkpoint + a trivial baseline (always the training-mean action)
Distances to step changes are NOT computed here; the analysis joins gold_test.csv by (suite, task, demo, t).

Usage: python -m oftvla.heldout_eval --run_dir RUN --ckpts 30000 20000 10000 --data_root ... \
           --test_ids e3_testdemo_ids.csv --text_csv A0ASMAS/test.csv --base_vlm ...
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from oftvla.data import LiberoLangDataset, identity_collate, load_stats, normalize_actions
from oftvla.model import OFTConfig, QwenOFT

GRIPPER_THRESHOLD = 0.5  # 1 open / 0 close, as in the LIBERO policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ckpts", type=int, nargs="+", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--test_ids", required=True, help="split file of the held-out test demos")
    p.add_argument("--text_csv", required=True, help="A0ASMAS test.csv")
    p.add_argument("--setup", default=None, help="text column; default = the run's training setup (args.json)")
    p.add_argument("--base_vlm", required=True)
    p.add_argument("--stride", type=int, default=1, help="evaluate every k-th frame (1 = all)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=16)
    return p.parse_args()


def frame_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Per-sample metrics for normalised chunks pred, gt of shape [B, 8, 7] (gripper = last dim)."""
    err = np.abs(pred.astype(np.float32) - gt.astype(np.float32))
    grip = (pred[..., 6] > GRIPPER_THRESHOLD) == (gt[..., 6] > GRIPPER_THRESHOLD)
    return {
        "l1": err.mean(axis=(1, 2)),             # the training loss, per sample
        "l1_arm": err[..., :6].mean(axis=(1, 2)),
        "l1_first": err[:, 0].mean(axis=1),      # the action executed first
        "grip_agree": grip.mean(axis=1),         # fraction of the 8 steps with the same open/close
    }


def resolve_setup(run_dir: str, setup) -> str:
    """The text column must be the one the run was trained with, unless explicitly overridden."""
    with open(os.path.join(run_dir, "args.json")) as f:
        trained = json.load(f).get("setup", "A0")
    if setup is None:
        return trained
    if setup != trained:
        print(f"[warn] evaluating with setup {setup!r}, but the run was trained with {trained!r}", flush=True)
    return setup


@torch.inference_mode()
def predict(model: QwenOFT, ds, idx, batch_size: int, num_workers: int):
    loader = DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=identity_collate)
    preds, gts = [], []
    for batch in loader:
        preds.append(model.predict_action(batch).astype(np.float16))           # [B, 8, 7] normalised
        gts.append(np.stack([s["action"] for s in batch]).astype(np.float16))  # [B, 8, 7] normalised
    return np.concatenate(preds), np.concatenate(gts)


def main():
    args = parse_args()
    setup = resolve_setup(args.run_dir, args.setup)
    stats = load_stats(os.path.join(args.run_dir, "dataset_statistics.json"))  # training statistics
    ds = LiberoLangDataset(args.data_root, [args.test_ids], None, setup=setup, stats=stats, text_csv=args.text_csv)
    idx = list(range(0, len(ds), args.stride))
    keys = [(*ds.demos[di]["key"], t) for di, t in (ds.starts[i] for i in idx)]
    print(f"[data] setup={setup} test demos={len(ds.demos)} frames={len(ds)} evaluated={len(idx)} "
          f"texts={len(ds.texts)} skipped_not_in_text_csv={ds.n_skipped_not_in_text}", flush=True)
    out_dir = os.path.join(args.run_dir, "heldout_eval")
    os.makedirs(out_dir, exist_ok=True)

    mean_action = normalize_actions(np.asarray(stats["mean"], np.float32)[None], stats)[0]
    model = QwenOFT(OFTConfig(base_vlm=args.base_vlm)).to("cuda").eval()  # bf16 weights, as at inference
    summary_path = os.path.join(out_dir, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    for step in args.ckpts:
        t0 = time.time()
        sd = torch.load(os.path.join(args.run_dir, f"steps_{step}_model.pt"), map_location="cpu")
        print(model.load_state_dict(sd), flush=True)
        del sd
        pred, gt = predict(model, ds, idx, args.batch_size, args.num_workers)
        m = frame_metrics(pred, gt)
        base = frame_metrics(np.broadcast_to(mean_action, gt.shape), gt)

        with open(os.path.join(out_dir, f"step{step}_frames.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["suite", "task", "demo", "t", "l1", "l1_arm", "l1_first", "grip_agree"])
            for k, l1, arm, first, g in zip(keys, m["l1"], m["l1_arm"], m["l1_first"], m["grip_agree"]):
                w.writerow([*k, f"{l1:.6f}", f"{arm:.6f}", f"{first:.6f}", f"{g:.4f}"])
        np.savez_compressed(os.path.join(out_dir, f"step{step}_pred.npz"), pred=pred)

        summary[str(step)] = {"setup": setup, "frames": len(idx), "demos": len(ds.demos),
                              **{k: float(v.mean()) for k, v in m.items()},
                              "baseline_mean_action_l1": float(base["l1"].mean()),
                              "minutes": round((time.time() - t0) / 60, 1)}
        print(json.dumps({"step": step, **summary[str(step)]}), flush=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
