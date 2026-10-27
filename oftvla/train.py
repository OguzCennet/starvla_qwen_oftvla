"""Single-GPU training of QwenOFT, following StarVLA's LIBERO recipe (train_starvla.py @ 985c223).

Recipe (StarVLA)                               | Here
-----------------------------------------------|---------------------------------------------
8 GPUs x 16 per device = 128 per step          | 1 GPU x 16 x 8 accumulation = 128 per step
30K steps, 5K warmup, cosine to min_lr 1e-6    | same (min_lr / base_lr 2.5e-5 = 0.04 of peak)
AdamW betas (0.9, 0.95), eps 1e-8, wd 1e-8     | same
lr: VLM 1e-5, action head 1e-4, full FT        | same
grad clip 1.0, bf16 (DeepSpeed ZeRO-2)         | grad clip 1.0, fp32 weights + bf16 autocast
eval every 100 steps: mse_score on next batch  | same (the batch is consumed, as in StarVLA)
save weights every 10K steps                   | same + resume state (needed for 24 h job limit)

Usage: python -m oftvla.train --run_dir DIR --data_root ... --train_ids ... [see --help]
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from oftvla.data import LiberoLangDataset, identity_collate, load_stats, save_stats
from oftvla.model import OFTConfig, QwenOFT


def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data_root", required=True)
    p.add_argument("--train_ids", nargs="+", required=True)
    p.add_argument("--forbidden_ids", nargs="*", default=[], help="split files whose TASKS must not be trained on")
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--setup", default="A0")
    p.add_argument("--num_workers", type=int, default=8)
    # model
    p.add_argument("--base_vlm", required=True)
    # optimisation (StarVLA LIBERO defaults)
    p.add_argument("--micro_batch", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--warmup_steps", type=int, default=5000)
    p.add_argument("--lr_vlm", type=float, default=1e-5)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--min_lr_ratio", type=float, default=1e-6 / 2.5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    # bookkeeping
    p.add_argument("--run_dir", required=True)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--resume_every", type=int, default=2000)
    p.add_argument("--max_hours", type=float, default=1e9, help="save resume state and exit after this wall time")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_factor(step: int, warmup: int, total: int, min_ratio: float) -> float:
    """Linear warmup, then cosine from 1 to min_ratio (transformers 'cosine_with_min_lr')."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def infinite(loader):
    """Endless stream of batches; each pass over the data is a new shuffle drawn from the loader's generator."""
    while True:
        yield from loader


def save_weights(model, path):
    """bf16 state dict for evaluation (torch.save: Qwen ties lm_head to the embedding)."""
    sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.state_dict().items()}
    torch.save(sd, path)


def main():
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda")
    log_path = os.path.join(args.run_dir, "log.jsonl")
    resume_path = os.path.join(args.run_dir, "resume.pt")
    stats_path = os.path.join(args.run_dir, "dataset_statistics.json")
    done_path = os.path.join(args.run_dir, "DONE")
    if os.path.exists(done_path):
        print("[done] run already finished (DONE marker exists), nothing to do", flush=True)
        return

    # ---- data (stats fixed at the first start, reused on resume) ----
    stats = load_stats(stats_path) if os.path.exists(stats_path) else None
    ds = LiberoLangDataset(args.data_root, [os.path.expanduser(x) for x in args.train_ids],
                           os.path.expanduser(args.labels_csv), args.setup, stats=stats,
                           forbidden_id_files=[os.path.expanduser(x) for x in args.forbidden_ids])
    if stats is None:
        save_stats(ds.stats, stats_path)
    print(f"[data] demos={len(ds.demos)} samples={len(ds)} skipped_not_ok={ds.n_skipped_not_ok}", flush=True)
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.micro_batch, shuffle=True, drop_last=True, generator=gen,
                        num_workers=args.num_workers, collate_fn=identity_collate,
                        persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers > 0 else None)
    batches = infinite(loader)

    # ---- model + optimiser ----
    cfg = OFTConfig(base_vlm=args.base_vlm, vlm_dtype="float32")
    model = QwenOFT(cfg).to(device).train()
    groups = [
        {"name": "vlm", "params": [p for p in model.vlm.parameters() if p.requires_grad], "lr": args.lr_vlm},
        {"name": "head", "params": list(model.action_head.parameters()), "lr": args.lr_head},
    ]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, args.warmup_steps, args.max_steps, args.min_lr_ratio))

    step = 0
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        step = ckpt["step"]
        torch.set_rng_state(ckpt["torch_rng"])
        gen.manual_seed(args.seed + step)  # new shuffle order after resume
        print(f"[resume] from step {step}", flush=True)
        del ckpt
    with open(os.path.join(args.run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    def save_resume():
        tmp = resume_path + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "step": step, "torch_rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, resume_path)  # atomic: a crash during saving never corrupts the last good state
        print(f"[resume] saved at step {step}", flush=True)

    # ---- loop: one iteration = one optimizer step = grad_accum micro-batches ----
    t_start = time.time()
    t_log, data_time = time.time(), 0.0
    while step < args.max_steps:
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.grad_accum):
            t0 = time.time()
            batch = next(batches)
            data_time += time.time() - t0
            loss = model(batch)["action_loss"]
            (loss / args.grad_accum).backward()
            loss_sum += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        opt.step()
        sched.step()
        step += 1

        rec = None
        if step % args.log_every == 0:
            dt = time.time() - t_log
            rec = {"step": step, "loss": loss_sum / args.grad_accum, "grad_norm": grad_norm,
                   "lr_vlm": opt.param_groups[0]["lr"], "lr_head": opt.param_groups[1]["lr"],
                   "sec_per_step": dt / args.log_every, "data_frac": data_time / dt,
                   "gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
            t_log, data_time = time.time(), 0.0
        if step % args.eval_every == 0:
            ev = next(batches)
            model.eval()
            pred = model.predict_action(ev)
            model.train()
            gt = np.stack([s["action"] for s in ev]).astype(np.float32)
            rec = rec or {"step": step}
            rec["mse_score"] = float(np.linalg.norm(pred - gt) / gt.size)
        if rec:
            print(json.dumps(rec), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if step % args.save_every == 0 or step == args.max_steps:
            save_weights(model, os.path.join(args.run_dir, f"steps_{step}_model.pt"))
            print(f"[ckpt] weights saved at step {step}", flush=True)
        if step % args.resume_every == 0 and step < args.max_steps:
            save_resume()
        if (time.time() - t_start) / 3600 > args.max_hours and step < args.max_steps:
            save_resume()
            print("[time] max_hours reached, exiting for resubmission", flush=True)
            return
    open(done_path, "w").write(f"finished at step {step}\n")
    if os.path.exists(resume_path):
        os.remove(resume_path)  # final weights are saved; the large resume state is no longer needed
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
