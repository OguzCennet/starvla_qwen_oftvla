"""Compare outputs of oftvla/flow_heldout_eval.py (validation, STEP 5).

  * every summary: sample 0 vs the model's own predict_action (--check), max|diff| must be 0;
  * pairs given with --same: identical predictions (rerun of one run x checkpoint must be bitwise identical);
  * all summaries: identical noise md5 (same frames -> same initial noise, in every run and both heads).

Usage: python scripts/checks/flow_heldout_eval_check.py DIR1/stepN_summary.json DIR2/... [--same A.json B.json]
"""

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--same", nargs=2, action="append", default=[], metavar=("A", "B"))
    a = ap.parse_args()
    ok = True
    sums = {p: json.load(open(p)) for p in a.summaries + [x for pair in a.same for x in pair]}
    for p, s in sums.items():
        d = s.get("check_predict_action_max_abs_diff")
        ok &= d == 0.0
        print(f"{s['head']:5s} {s['run_dir'].rsplit('/', 1)[-1]:24s} step {s['step']:>6} frames {s['frames']} "
              f"noise {s['noise_md5'][:12]} predict_action max|diff| {d}  s/1000 frames {s['seconds_per_1000_frames']}"
              f"  l1_s0 {s['l1_s0']:.4f} l1_meanK {s['l1_meanK']:.4f} l1_bestK {s['l1_bestK']:.4f} "
              f"spread {s['spread']:.4f} grip_s0 {s['grip_s0']:.4f}")
    noise = {s["noise_md5"] for s in sums.values()}
    ok &= len(noise) == 1
    print(f"noise md5 identical across all: {len(noise) == 1}")
    for x, y in a.same:
        px, py = (np.load(p.replace("_summary.json", "_samples.npz")) for p in (x, y))
        same = all(np.array_equal(px[k], py[k]) for k in ("pred", "gt"))
        cx, cy = (open(p.replace("_summary.json", "_frames.csv")).read() for p in (x, y))
        ok &= same and cx == cy
        print(f"rerun identical (pred, gt, frames.csv): {same and cx == cy}  {x} vs {y}")
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
