"""Compare runs (e.g. e3v5_A0 / e3v5_AS / e3v5_MAS) on the held-out test frames scored by heldout_eval.py.

The unit of inference is the TASK (24 test tasks): frames of one demo are strongly correlated, so per-frame
p-values would be far too optimistic. For every metric and frame subset:
  * mean per task for each run, then paired differences between runs over the tasks
    (mean difference, tasks better / worse, Wilcoxon signed-rank p, task-bootstrap 95 % CI);
  * frame-weighted means are reported too, for orientation only.
Frame subsets: all frames; "near" = within NEAR raw steps of a MAS step change (|dist_MAS| <= NEAR, from
gold_test.csv; the prompt label switches 3 steps after the event); "far" = the rest (incl. frames without dist).
Also: L1 per MAS step type (first word of cur_MAS).

Usage: python -m oftvla.heldout_analysis --runs RUN_A0 RUN_AS RUN_MAS --step 30000 \
           --gold /hnvme/.../step2_gold/gold_test.csv --out_dir DIR
"""

import argparse
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEYS = ["suite", "task", "demo", "t"]
METRICS = ["l1", "l1_arm", "l1_first", "grip_agree"]
NEAR = 8  # raw steps = one action chunk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run dirs; the name after the first '_' is the label")
    p.add_argument("--step", type=int, default=30000)
    p.add_argument("--gold", required=True, help="gold_test.csv (dist_MAS, cur_MAS, task_id)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_boot", type=int, default=10000)
    return p.parse_args()


def run_label(run_dir: str) -> str:
    """Folder name after the FIRST '_' (e3v5_AS_noAG -> AS_noAG); the whole name if there is no '_'."""
    name = os.path.basename(os.path.normpath(run_dir))
    return name.split("_", 1)[1] if "_" in name else name


def load_runs(run_dirs, step: int) -> pd.DataFrame:
    """Long table: one row per (run, frame). Refuses duplicate labels and runs scored on different frames."""
    labels = [run_label(d) for d in run_dirs]
    dups = sorted({l for l in labels if labels.count(l) > 1})
    if dups:
        raise ValueError(f"duplicate run labels {dups} from {list(run_dirs)}")
    frames, ref = [], None
    for d in run_dirs:
        f = pd.read_csv(os.path.join(d, "heldout_eval", f"step{step}_frames.csv"))
        keyset = set(map(tuple, f[KEYS].astype(str).values))
        if ref is None:
            ref = keyset
        elif keyset != ref:
            raise ValueError(f"{d}: scored frames differ from the first run (cannot compare)")
        f["run"] = run_label(d)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def add_gold(df: pd.DataFrame, gold: pd.DataFrame, near: int = NEAR) -> pd.DataFrame:
    """Join task_id, dist_MAS and the MAS step type; every scored frame must be in the gold index."""
    g = gold[KEYS + ["task_id", "dist_MAS", "cur_MAS"]].copy()
    out = df.merge(g, on=KEYS, how="left", validate="many_to_one", indicator=True)
    missing = (out["_merge"] != "both").sum()
    if missing:
        raise ValueError(f"{missing} scored frames are not in the gold index")
    out = out.drop(columns="_merge")
    out["band"] = np.where(out["dist_MAS"].abs() <= near, "near", "far")  # NaN dist -> far
    out["mas_type"] = out["cur_MAS"].astype(str).str.split().str[0]
    return out


def paired_stats(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int = 0, higher_is_better: bool = False) -> dict:
    """Per-task values of run A and run B (same task order). diff = B - A."""
    diff = np.asarray(b, float) - np.asarray(a, float)
    better = diff > 0 if higher_is_better else diff < 0
    worse = diff < 0 if higher_is_better else diff > 0
    nonzero = diff[diff != 0]
    p = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 1 and np.any(nonzero) else 1.0
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    return {"n_tasks": int(len(diff)), "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
            "mean_diff": float(diff.mean()), "rel_diff": float(diff.mean() / np.mean(a)),
            "b_better": int(better.sum()), "b_worse": int(worse.sum()), "equal": int((diff == 0).sum()),
            "wilcoxon_p": p, "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def compare(df: pd.DataFrame, n_boot: int) -> list:
    rows = []
    runs = list(dict.fromkeys(df["run"]))
    for subset, part in [("all", df), ("near", df[df.band == "near"]), ("far", df[df.band == "far"])]:
        for metric in METRICS:
            per_task = part.pivot_table(index="task_id", columns="run", values=metric, aggfunc="mean")[runs]
            per_task = per_task.dropna()
            for a, b in combinations(runs, 2):
                s = paired_stats(per_task[a].values, per_task[b].values, n_boot,
                                 higher_is_better=(metric == "grip_agree"))
                rows.append({"subset": subset, "metric": metric, "a": a, "b": b,
                             "frames": int(len(part) / len(runs)), **s})
    return rows


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = load_runs(args.runs, args.step)
    gold = pd.read_csv(args.gold, usecols=KEYS + ["task_id", "dist_MAS", "cur_MAS"])
    df = add_gold(df, gold)
    runs = list(dict.fromkeys(df["run"]))
    n = len(df) // len(runs)
    print(f"[data] step {args.step} | runs {runs} | {n} frames per run | "
          f"near (|dist_MAS| <= {NEAR}) {int((df.band == 'near').sum() / len(runs))} | "
          f"tasks {df.task_id.nunique()}", flush=True)

    # Frame-weighted means (orientation only)
    fw = df.groupby(["band", "run"])[METRICS].mean().unstack("run").round(4)
    print("\nFrame-weighted means by band:\n", fw.to_string(), flush=True)
    print("\nFrame-weighted L1 by MAS step type:\n",
          df.pivot_table(index="mas_type", columns="run", values="l1", aggfunc="mean")[runs].round(4).to_string())

    # Per task (all frames) and paired comparisons over tasks
    per_task = df.pivot_table(index="task_id", columns="run", values="l1", aggfunc="mean")[runs]
    per_task.round(5).to_csv(os.path.join(args.out_dir, f"step{args.step}_per_task_l1.csv"))
    rows = compare(df, args.n_boot)
    with open(os.path.join(args.out_dir, f"step{args.step}_paired.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\nPaired over tasks (diff = b - a; for L1 negative = b better):")
    for r in rows:
        if r["metric"] in ("l1", "grip_agree"):
            print(f"  {r['subset']:4s} {r['metric']:10s} {r['b']:>4s} vs {r['a']:<4s} "
                  f"diff {r['mean_diff']:+.5f} ({100 * r['rel_diff']:+.1f} %) "
                  f"better/worse/equal {r['b_better']}/{r['b_worse']}/{r['equal']} "
                  f"p={r['wilcoxon_p']:.3g} CI [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]")


if __name__ == "__main__":
    main()
