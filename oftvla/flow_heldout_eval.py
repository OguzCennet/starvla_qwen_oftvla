"""Offline action error of a flow-head run (GR00T / PI) on the held-out E3 test demos, K samples per frame.

Same frames, texts and action space as oftvla/heldout_eval.py (the OFT eval), so the numbers are on the same scale:
every frame of the test demos (test_ids), test.csv next to the run's training text_csv, the run's text column
(args.json), actions normalised with the run's dataset_statistics.json, L1 = mean over the 8 x 7 chunk,
gripper open/close at 0.5.

Inference mode as the fork's policy server: the whole model in bf16 (run_policy_server.sh:9/20-21 USE_BF16=1 ->
policy_wrapper.py:68-69 framework.to(torch.bfloat16)), .eval() (policy_wrapper.py:70: dropout off),
torch.inference_mode (QwenGR00T.py:223 / QwenPI.py:247), head under fp32 autocast as in predict_action.
The CoT wrapper comes from args.json ("cot_prompt"; runs that do not record it were trained with the
FlowConfig default COT_PROMPT, train_flow.py builds FlowConfig(head, base_vlm) only).

Per frame: one VLM forward, then K head samplings (Euler, 4 steps). Sample k starts from noise drawn by a CPU
generator seeded with sha256("suite/task/demo/t/k") -> the same initial noise for a frame in every run and both
heads (float32 normal, cast to bf16 = the dtype the fork's randn uses with bf16 weights).

Outputs (RUN/heldout_eval/, three files per checkpoint):
  step{N}_frames.csv    one row per frame: suite, task, demo, t, l1_s0, l1_arm_s0, l1_first_s0, l1_meanK, l1_bestK,
                        spread (mean over 8 x 7 of the std over K), grip_s0, grip_meanK (gripper agreement)
  step{N}_samples.npz   pred [frames, K, 8, 7] and gt [frames, 8, 7], float16, rows as in the CSV
  step{N}_summary.json  means, frame count, K, text_csv md5, noise md5, runtime
--check also runs the model's own predict_action(batch, init_noise = noise of sample 0) and records max|diff|.
Sharding: --shard i --num_shards N writes step{N}_s{i}of{N}.npz (+ _summary.json); batches of the unsharded order
are dealt out whole (batch b -> shard b mod N), so a frame is computed in the same batch as unsharded; --merge
(same args, no GPU) joins them into the three files above (bitwise identical to an unsharded run) and removes them.

Usage: python -m oftvla.flow_heldout_eval --run_dir RUN --ckpt 30000 --data_root ... --test_ids e3_testdemo_ids.csv \
           --base_vlm ... [--K 8] [--shard i --num_shards N | --merge --num_shards N] [--max_frames 200 --out_dir D --check]
"""

import argparse
import csv
import hashlib
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from oftvla.data import LiberoLangDataset, identity_collate, load_stats
from oftvla.flow_model import COT_PROMPT, FlowConfig, build_flow_model
from oftvla.heldout_eval import GRIPPER_THRESHOLD, frame_metrics, md5_of, resolve_setup, resolve_text_csv

HORIZON, ACTION_DIM = 8, 7


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ckpt", type=int, required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--test_ids", required=True, help="split file of the held-out test demos")
    p.add_argument("--text_csv", default=None, help="default: test.csv next to the run's training text_csv")
    p.add_argument("--setup", default=None, help="text column; default = the run's training setup (args.json)")
    p.add_argument("--base_vlm", required=True)
    p.add_argument("--K", type=int, default=8, help="head samplings per frame")
    p.add_argument("--max_frames", type=int, default=None,
                   help="validation: this many frames spread evenly over the test set (default: every frame)")
    p.add_argument("--first_frames", type=int, default=None, help="timing: only the first N frames (in order)")
    p.add_argument("--out_dir", default=None, help="default RUN/heldout_eval")
    p.add_argument("--check", action="store_true", help="compare sample 0 with the model's predict_action")
    p.add_argument("--shard", type=int, default=0, help="this job's shard (0 .. num_shards-1)")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--merge", action="store_true", help="join the shard files (CPU, no model) and remove them")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=14)
    return p.parse_args()


def noise_seed(key, k: int) -> int:
    """Seed of sample k of frame key = (suite, task, demo, t); sha256, not hash() (salted per process)."""
    s = "/".join(str(x) for x in (*key, k))
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little") & ((1 << 63) - 1)


def frame_noise(keys, K: int) -> torch.Tensor:
    """Initial noise [B, K, 8, 7] float32 (CPU), a function of the frame keys only."""
    out = torch.empty((len(keys), K, HORIZON, ACTION_DIM))
    for i, key in enumerate(keys):
        for k in range(K):
            g = torch.Generator().manual_seed(noise_seed(key, k))
            out[i, k] = torch.randn((HORIZON, ACTION_DIM), generator=g)
    return out


def resolve_cot(run_dir: str):
    with open(os.path.join(run_dir, "args.json")) as f:
        a = json.load(f)
    if "cot_prompt" in a:
        return a["cot_prompt"], "args.json"
    return COT_PROMPT, "FlowConfig default (args.json has no cot_prompt; train_flow uses the default)"


def encode(model, batch):
    """The VLM part of predict_action (flow_model.py QwenGR00T / QwenPI.predict_action), once per frame."""
    images = model._prepare_images(batch)
    texts = [s["lang"] for s in batch]
    if model.cfg.head == "pi":
        embs, mask = model._encode_vl_hidden_states(images, texts)  # one hidden state per DiT block
    else:
        hidden_states, mask = model._vlm_forward(images, texts)
        embs = hidden_states[-1]
    if mask is not None:
        mask = mask.to(dtype=torch.bool)
    return embs, mask


def sample(model, embs, mask, init_noise: torch.Tensor) -> torch.Tensor:
    """The head part of predict_action with the given initial noise."""
    with torch.autocast("cuda", dtype=torch.float32):
        return model.action_head.predict_action(embs, None, encoder_attention_mask=mask, init_noise=init_noise)


def k_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """pred [B, K, 8, 7], gt [B, 8, 7] (float16, normalised) -> per-frame metrics."""
    p = pred.astype(np.float32)
    s0 = frame_metrics(pred[:, 0], gt)
    mean_k = p.mean(axis=1)
    mk = frame_metrics(mean_k, gt)
    l1_k = np.abs(p - gt.astype(np.float32)[:, None]).mean(axis=(2, 3))  # [B, K]
    return {"l1_s0": s0["l1"], "l1_arm_s0": s0["l1_arm"], "l1_first_s0": s0["l1_first"],
            "l1_meanK": mk["l1"], "l1_bestK": l1_k.min(axis=1), "spread": p.std(axis=1).mean(axis=(1, 2)),
            "grip_s0": s0["grip_agree"], "grip_meanK": mk["grip_agree"]}


def main():
    args = parse_args()
    setup = resolve_setup(args.run_dir, args.setup)
    text_csv = resolve_text_csv(args.run_dir, args.text_csv)
    with open(os.path.join(args.run_dir, "args.json")) as f:
        head = json.load(f)["head"]
    cot, cot_src = resolve_cot(args.run_dir)
    text_md5 = md5_of(text_csv)
    print(f"[run] {args.run_dir} head={head} ckpt={args.ckpt} setup={setup} K={args.K}", flush=True)
    print(f"[text_csv] {text_csv} md5={text_md5}", flush=True)
    print(f"[cot] {cot_src}: {cot!r}", flush=True)
    stats = load_stats(os.path.join(args.run_dir, "dataset_statistics.json"))
    ds = LiberoLangDataset(args.data_root, [args.test_ids], None, setup=setup, stats=stats, text_csv=text_csv)
    if args.max_frames:
        idx = np.linspace(0, len(ds) - 1, args.max_frames).round().astype(int).tolist()
    else:
        idx = list(range(len(ds) if args.first_frames is None else min(args.first_frames, len(ds))))
    keys = [(*ds.demos[di]["key"], t) for di, t in (ds.starts[i] for i in idx)]
    print(f"[data] test demos={len(ds.demos)} frames={len(ds)} evaluated={len(idx)} texts={len(ds.texts)} "
          f"skipped_not_in_text_csv={ds.n_skipped_not_in_text}", flush=True)
    real_dir = os.path.join(args.run_dir, "heldout_eval")
    out_dir = args.out_dir or real_dir
    if (args.max_frames or args.first_frames) and os.path.realpath(out_dir) == os.path.realpath(real_dir):
        raise SystemExit(f"[eval] refusing to write a partial frame set (--max_frames / --first_frames) into "
                         f"{real_dir}; give --out_dir outside it (e.g. runs/oftvla/eval_tests/<run>_step<STEP>/)")
    os.makedirs(out_dir, exist_ok=True)
    base = {"run_dir": args.run_dir, "head": head, "step": args.ckpt, "setup": setup, "K": args.K,
            "frames": len(idx), "demos": len(ds.demos), "text_csv": text_csv, "text_csv_md5": text_md5,
            "cot_prompt": cot, "gripper_threshold": GRIPPER_THRESHOLD, "batch_size": args.batch_size}
    if args.merge:
        return merge(args, out_dir, keys, base)

    # Batches are formed over ALL evaluated frames in order (as an unsharded run) and dealt out whole:
    # shard i takes batches i, i + N, i + 2N, ... -> every frame is computed in exactly the same batch
    # (same padding, same kernel shapes) as unsharded, so the merged output is bitwise identical.
    batches = [list(range(j, min(j + args.batch_size, len(idx)))) for j in range(0, len(idx), args.batch_size)]
    mine = batches[args.shard::args.num_shards]
    rows = [r for b in mine for r in b]
    print(f"[shard] {args.shard}/{args.num_shards}: {len(mine)} of {len(batches)} batches, {len(rows)} frames",
          flush=True)

    t_load = time.time()
    model = build_flow_model(FlowConfig(head=head, base_vlm=args.base_vlm, cot_prompt=cot))
    assert (model.action_horizon, model.action_head.action_dim) == (HORIZON, ACTION_DIM)
    sd = torch.load(os.path.join(args.run_dir, f"steps_{args.ckpt}_model.pt"), map_location="cpu")
    print(model.load_state_dict(sd, strict=True), flush=True)
    del sd
    model = model.to(torch.bfloat16).to("cuda").eval()  # policy_wrapper.py:68-70
    print(f"[model] loaded in {time.time() - t_load:.0f} s; training={model.training}", flush=True)

    loader = DataLoader(Subset(ds, idx), batch_sampler=mine, num_workers=args.num_workers,
                        collate_fn=identity_collate)
    preds, gts, noise_md5, check_diff = [], [], hashlib.md5(), 0.0
    t0, t_vlm, t_head = time.time(), 0.0, 0.0
    with torch.inference_mode():
        for b, batch in zip(mine, loader):
            noise = frame_noise([keys[r] for r in b], args.K)
            noise_md5.update(noise.numpy().tobytes())
            noise = noise.to("cuda").to(torch.bfloat16)
            torch.cuda.synchronize(); ta = time.time()
            embs, mask = encode(model, batch)
            torch.cuda.synchronize(); tb = time.time()
            pk = [sample(model, embs, mask, noise[:, k]).float() for k in range(args.K)]
            torch.cuda.synchronize(); t_vlm += tb - ta; t_head += time.time() - tb
            if args.check:
                ref = torch.from_numpy(model.predict_action(batch, init_noise=noise[:, 0])).float().to(pk[0].device)
                check_diff = max(check_diff, (ref - pk[0]).abs().max().item())
            preds.append(torch.stack(pk, 1).cpu().numpy().astype(np.float16))  # [B, K, 8, 7]
            gts.append(np.stack([s["action"] for s in batch]).astype(np.float16))
    pred, gt = np.concatenate(preds), np.concatenate(gts)
    seconds = time.time() - t0
    timing = {"seconds": round(seconds, 1), "seconds_per_1000_frames": round(1000 * seconds / len(rows), 1),
              "seconds_vlm": round(t_vlm, 1), "seconds_head_K": round(t_head, 1)}
    if args.check:
        timing["check_predict_action_max_abs_diff"] = check_diff
    if args.num_shards == 1:
        write_outputs(out_dir, args.ckpt, keys, pred, gt, {**base, "noise_md5": noise_md5.hexdigest(), **timing})
        return
    tag = f"step{args.ckpt}_s{args.shard}of{args.num_shards}"
    np.savez(os.path.join(out_dir, f"{tag}.npz"), pred=pred, gt=gt, rows=np.asarray(rows, np.int64))
    with open(os.path.join(out_dir, f"{tag}_summary.json"), "w") as f:
        json.dump({**base, "shard": args.shard, "num_shards": args.num_shards, "shard_frames": len(rows),
                   **timing}, f, indent=2)
    print(json.dumps({"shard": args.shard, "shard_frames": len(rows), **timing}), flush=True)


def write_outputs(out_dir, ckpt, keys, pred, gt, summary):
    m = k_metrics(pred, gt)
    with open(os.path.join(out_dir, f"step{ckpt}_frames.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "task", "demo", "t", *m])
        for i, key in enumerate(keys):
            w.writerow([*key, *(f"{m[c][i]:.6f}" for c in m)])
    np.savez_compressed(os.path.join(out_dir, f"step{ckpt}_samples.npz"), pred=pred, gt=gt)
    summary = {**summary, **{c: float(v.mean()) for c, v in m.items()}}
    with open(os.path.join(out_dir, f"step{ckpt}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary), flush=True)


def merge(args, out_dir, keys, base):
    """Join the shard files of one run x checkpoint into the unsharded outputs, then delete the shard files."""
    tags = [os.path.join(out_dir, f"step{args.ckpt}_s{i}of{args.num_shards}") for i in range(args.num_shards)]
    missing = [t for t in tags if not os.path.exists(t + ".npz") or not os.path.exists(t + "_summary.json")]
    if missing:
        raise SystemExit(f"[merge] missing shard files: {missing}")
    n = len(keys)
    pred = np.zeros((n, args.K, HORIZON, ACTION_DIM), np.float16)
    gt = np.zeros((n, HORIZON, ACTION_DIM), np.float16)
    seen = np.zeros(n, np.int64)
    sums = []
    for t in tags:
        z = np.load(t + ".npz")
        pred[z["rows"]], gt[z["rows"]] = z["pred"], z["gt"]
        np.add.at(seen, z["rows"], 1)
        sums.append(json.load(open(t + "_summary.json")))
    if not (seen == 1).all():
        raise SystemExit(f"[merge] rows not covered exactly once: {int((seen == 0).sum())} missing, "
                         f"{int((seen > 1).sum())} duplicated")
    for s in sums:  # every shard evaluated the same thing, with the same batching
        for k, v in {**base, "batch_size": sums[0]["batch_size"]}.items():
            if s[k] != v:
                raise SystemExit(f"[merge] shard {s['shard']}: {k} = {s[k]!r}, expected {v!r}")
    noise_md5 = hashlib.md5()
    for j in range(0, n, 1024):  # same bytes, same order as an unsharded run
        noise_md5.update(frame_noise(keys[j: j + 1024], args.K).numpy().tobytes())
    gpu_s = sum(s["seconds"] for s in sums)
    timing = {"num_shards": args.num_shards, "batch_size": sums[0]["batch_size"],
              "seconds": round(gpu_s, 1), "seconds_per_1000_frames": round(1000 * gpu_s / n, 1),
              "seconds_vlm": round(sum(s["seconds_vlm"] for s in sums), 1),
              "seconds_head_K": round(sum(s["seconds_head_K"] for s in sums), 1),
              "shard_seconds_max": max(s["seconds"] for s in sums)}
    if "check_predict_action_max_abs_diff" in sums[0]:
        timing["check_predict_action_max_abs_diff"] = max(s["check_predict_action_max_abs_diff"] for s in sums)
    write_outputs(out_dir, args.ckpt, keys, pred, gt, {**base, "noise_md5": noise_md5.hexdigest(), **timing})
    for t in tags:
        os.remove(t + ".npz")
        os.remove(t + "_summary.json")
    print(f"[merge] {args.num_shards} shards merged, shard files removed", flush=True)


if __name__ == "__main__":
    main()
