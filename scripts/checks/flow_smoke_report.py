"""Report for a flow-head smoke run (train_flow.py with --log_every 1 --log_indices).

Prints loss / grad-norm curves (window means), step time, peak memory, the projected wall time to --target
steps, and the resume check: the logged rank-0 dataset indices of every step must equal
GlobalStream(num_samples, seed).draw(draws - 1, world, per_device, 0), the draw counter must advance by one
per step plus one per eval (every eval_every steps), and both must continue across the resume boundary.
Optionally writes a PNG with the curves (--png).

Usage: python scripts/checks/flow_smoke_report.py --run_dir R --num_samples N [--resume_step 150] [--png F]
"""

import argparse
import json
import os
import statistics as st

from oftvla.flow_data import GlobalStream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--num_samples", type=int, required=True, help="len(dataset), printed as [data] samples=")
    ap.add_argument("--resume_step", type=int, default=150)
    ap.add_argument("--target", type=int, default=30000)
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    args = json.load(open(os.path.join(a.run_dir, "args.json")))
    recs = [json.loads(l) for l in open(os.path.join(a.run_dir, "log.jsonl")) if l.strip()]
    recs = [r for r in recs if "loss" in r]
    world, per, ev = args["world"], args["per_device_batch"], args["eval_every"]
    print(f"### {os.path.basename(a.run_dir)}: head {args['head']} world {world} x per_device {per} "
          f"(global {world * per}), {len(recs)} logged steps")

    # ---- data order / resume ----
    stream = GlobalStream(a.num_samples, args["seed"])
    steps = [r["step"] for r in recs]
    bad_idx = [r["step"] for r in recs if r["idx_rank0"] != stream.draw(r["draws"] - 1, world, per, 0)]
    exp_draws = [s + (s - 1) // ev for s in steps]  # eval at step k*ev consumes one draw after that step's batch
    bad_draws = [s for s, r, e in zip(steps, recs, exp_draws) if r["draws"] != e]
    print(f"  steps contiguous 1..{steps[-1]}: {steps == list(range(1, steps[-1] + 1))}")
    print(f"  draw counter as expected (step + evals): {not bad_draws} {bad_draws[:5]}")
    print(f"  rank-0 indices == GlobalStream draw, all steps: {not bad_idx} {bad_idx[:5]}")
    rs = a.resume_step
    if rs and rs < steps[-1]:
        b, c = recs[rs - 1], recs[rs]
        print(f"  resume boundary: step {b['step']} draws {b['draws']} idx {b['idx_rank0'][:4]}... | "
              f"step {c['step']} draws {c['draws']} idx {c['idx_rank0'][:4]}... "
              f"(expected draw {c['draws'] - 1} = {stream.draw(c['draws'] - 1, world, per, 0)[:4]}...)")
        lr_keys = [k for k in b if k.startswith("lr_")]
        print("  LR across boundary: " + "; ".join(f"{k} {b[k]:.3e} -> {c[k]:.3e}" for k in lr_keys))
        pre = [r["loss"] for r in recs[rs - 10:rs]]
        post = [r["loss"] for r in recs[rs:rs + 10]]
        print(f"  loss mean 10 steps before resume {st.mean(pre):.4f}, 10 after {st.mean(post):.4f}")

    # ---- curves ----
    w = a.window
    print(f"  {'steps':>9} {'loss':>11} {'loss min/max':>21} {'grad_norm':>11} {'gn max':>10} {'s/step':>7}")
    for i in range(0, len(recs), w):
        ch = recs[i:i + w]
        ls, gn = [r["loss"] for r in ch], [r["grad_norm"] for r in ch]
        print(f"  {ch[0]['step']:>4}-{ch[-1]['step']:<4} {st.mean(ls):>11.4g} {min(ls):>10.4g}/{max(ls):<10.4g} "
              f"{st.mean(gn):>11.4g} {max(gn):>10.4g} {st.median(r['sec_per_step'] for r in ch):>7.3f}")

    # ---- step time / memory / projection ----
    skip = {1, 2, rs + 1, rs + 2} | {s for s in steps if s % ev == 0 or s % ev == 1 or s == rs}
    times = [r["sec_per_step"] for r in recs if r["step"] not in skip]
    med, mean = st.median(times), st.mean(times)
    all_mean = st.mean(r["sec_per_step"] for r in recs)
    print(f"  step time: median {med:.3f} s, mean {mean:.3f} s (excl. first steps, eval/resume steps); "
          f"mean incl. everything {all_mean:.3f} s; data wait median "
          f"{st.median(r['data_frac'] for r in recs[2:]) * 100:.1f}%")
    print(f"  peak memory per GPU: max {max(r['peak_mem_gb_max'] for r in recs):.1f} GB, "
          f"min {min(r['peak_mem_gb_min'] for r in recs):.1f} GB (of 140 GB H200)")
    h = a.target * all_mean / 3600
    print(f"  projected training time to {a.target}: {h:.1f} h at {all_mean:.3f} s/step (incl. eval), "
          f"{a.target * med / 3600:.1f} h at the median")

    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(steps, [r["loss"] for r in recs]); ax[0].set_title("loss (mean over ranks)")
        ax[1].plot(steps, [r["grad_norm"] for r in recs]); ax[1].set_title("grad norm (pre-clip)"); ax[1].set_yscale("log")
        ax[2].plot(steps, [r["sec_per_step"] for r in recs]); ax[2].set_title("s / step"); ax[2].set_yscale("log")
        for x in ax:
            x.axvline(rs, color="gray", ls="--", lw=0.8)
            x.set_xlabel("step")
        fig.suptitle(os.path.basename(a.run_dir))
        fig.tight_layout()
        fig.savefig(a.png, dpi=100)
        print(f"  plot: {a.png}")
    ok = steps == list(range(1, steps[-1] + 1)) and not bad_draws and not bad_idx
    print(f"  [{'PASS' if ok else 'FAIL'}] data order and resume continuity")


if __name__ == "__main__":
    main()
