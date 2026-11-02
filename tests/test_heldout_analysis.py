"""CPU tests for oftvla.heldout_analysis on synthetic runs with a planted effect.

Run:  python -m pytest -q tests/test_heldout_analysis.py
"""

import shutil

import numpy as np
import pandas as pd
import pytest

from oftvla.heldout_analysis import add_gold, compare, load_runs, paired_stats, run_label

KEYS = ["suite", "task", "demo", "t"]


def _make(tmp_path, near_gain=0.02, n_tasks=24, length=40):
    """Runs A0 / MAS on identical frames; MAS has lower L1 only near a MAS step change (dist in [-8, 8])."""
    rng = np.random.default_rng(0)
    gold_rows, frames = [], {"A0": [], "MAS": []}
    for k in range(n_tasks):
        for d in range(2):
            for t in range(length):
                dist = t - 20  # one step change at t = 20
                key = dict(suite="s", task=f"task{k}", demo=f"demo_{d}", t=t)
                gold_rows.append({**key, "task_id": f"T{k:02d}", "dist_MAS": dist,
                                  "cur_MAS": "reach for x" if t < 23 else "grasp x"})
                base = 0.1 + 0.01 * k + rng.normal(0, 0.002)
                for run in frames:
                    l1 = base - (near_gain if run == "MAS" and abs(dist) <= 8 else 0.0)
                    frames[run].append({**key, "l1": l1, "l1_arm": l1, "l1_first": l1, "grip_agree": 0.9})
    for run, rows in frames.items():
        (tmp_path / f"e3v5_{run}" / "heldout_eval").mkdir(parents=True)
        pd.DataFrame(rows).to_csv(tmp_path / f"e3v5_{run}" / "heldout_eval" / "step30000_frames.csv", index=False)
    return [str(tmp_path / "e3v5_A0"), str(tmp_path / "e3v5_MAS")], pd.DataFrame(gold_rows)


def test_planted_effect_found_only_near(tmp_path):
    runs, gold = _make(tmp_path)
    df = add_gold(load_runs(runs, 30000), gold)
    assert set(df.run) == {"A0", "MAS"} and set(df.mas_type) == {"reach", "grasp"}
    rows = {(r["subset"], r["metric"]): r for r in compare(df, n_boot=2000)}
    near, far = rows[("near", "l1")], rows[("far", "l1")]
    assert near["mean_diff"] == pytest.approx(-0.02, abs=1e-3) and near["b_better"] == 24 and near["wilcoxon_p"] < 1e-4
    assert near["ci95"][1] < 0
    assert far["mean_diff"] == pytest.approx(0.0, abs=1e-12) and far["equal"] == 24 and far["wilcoxon_p"] == 1.0


def test_no_effect_gives_no_difference(tmp_path):
    runs, gold = _make(tmp_path, near_gain=0.0)
    rows = compare(add_gold(load_runs(runs, 30000), gold), n_boot=500)
    assert all(r["mean_diff"] == pytest.approx(0.0, abs=1e-12) for r in rows)


def test_different_frames_refused(tmp_path):
    runs, gold = _make(tmp_path)
    p = tmp_path / "e3v5_MAS" / "heldout_eval" / "step30000_frames.csv"
    pd.read_csv(p).iloc[1:].to_csv(p, index=False)
    with pytest.raises(ValueError, match="differ"):
        load_runs(runs, 30000)


def test_frame_missing_in_gold_refused(tmp_path):
    runs, gold = _make(tmp_path)
    with pytest.raises(ValueError, match="not in the gold index"):
        add_gold(load_runs(runs, 30000), gold.iloc[1:])


def test_paired_stats_direction():
    s = paired_stats(np.array([1.0, 2.0, 3.0]), np.array([0.5, 2.5, 3.0]), n_boot=100)
    assert (s["b_better"], s["b_worse"], s["equal"]) == (1, 1, 1)
    g = paired_stats(np.array([0.8, 0.8]), np.array([0.9, 0.7]), n_boot=100, higher_is_better=True)
    assert (g["b_better"], g["b_worse"]) == (1, 1)


def test_run_label_after_first_underscore():
    names = ["e3v5_A0", "e3v5_AS", "e3v5_MAS", "e3v5_AS_noAG", "e3v5_MAS_noAG", "e3v5_A0_seed2"]
    assert [run_label(f"/runs/oftvla/{n}/") for n in names] == ["A0", "AS", "MAS", "AS_noAG", "MAS_noAG", "A0_seed2"]


def test_duplicate_label_refused(tmp_path):
    runs, gold = _make(tmp_path)
    other = tmp_path / "other" / "e3_MAS"                 # different folder, same label "MAS"
    shutil.copytree(runs[1], other)
    with pytest.raises(ValueError, match="duplicate run labels"):
        load_runs(runs + [str(other)], 30000)
