"""Reference loss levels for the flow heads on a training set (actions only, no images).

Flow loss = mean over (B*R, 8, 7) of (v_pred - (a - eps))^2, eps ~ N(0, 1) (gr00t_head.py:98-134, pi_head.py:109-147).
  * zero-output level  : v_pred = 0                         -> 1 + E[a^2]
  * best-constant level: v_pred = E[a] per (position, dim)  -> 1 + mean over (pos, dim) of Var(a)
Chunks are built exactly as LiberoLangDataset.__getitem__ (first_last end padding, gripper remap, min_max on the
arm dims with the run's dataset_statistics.json, float16 cast); every sample start counts once (uniform sampling).

Usage: python scripts/checks/flow_zero_level.py --stats RUN_DIR/dataset_statistics.json [--setup MAS]
"""

import argparse
import os

import h5py
import numpy as np

from oftvla.data import LiberoLangDataset, chunk_indices, load_stats, normalize_actions, remap_gripper

H = os.path.expanduser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/hnvme/workspace/y307bd13-vgv/data/libero_lang/v1")
    ap.add_argument("--train_ids", default=H("~/VLA_FT/DataSamples/Libero/Splits/e3_train_ids.csv"))
    ap.add_argument("--forbidden_ids", default=H("~/VLA_FT/DataSamples/Libero/Splits/e3_testdemo_ids.csv"))
    ap.add_argument("--text_csv", default=H("~/VLA_FT/DataSamples/Libero/A0ASMAS/train.csv"))
    ap.add_argument("--setup", default="MAS")
    ap.add_argument("--stats", required=True)
    a = ap.parse_args()
    stats = load_stats(a.stats)
    ds = LiberoLangDataset(a.data_root, [a.train_ids], None, a.setup, stats=stats,
                           forbidden_id_files=[a.forbidden_ids], text_csv=a.text_csv)
    n = len(ds)
    s1 = np.zeros((8, 7)); s2 = np.zeros((8, 7)); cnt = 0
    files = {}
    for d in ds.demos:
        if d["path"] not in files:
            files[d["path"]] = h5py.File(d["path"], "r")
        raw = files[d["path"]][d["key"][2]]["action"][:]
        L = d["length"]
        idx = np.stack([chunk_indices(t, L) for t in range(L)])          # [L, 8]
        ch = normalize_actions(remap_gripper(raw[idx.reshape(-1)]), stats).astype(np.float16)
        ch = ch.astype(np.float64).reshape(L, 8, 7)
        s1 += ch.sum(0); s2 += (ch ** 2).sum(0); cnt += L
    assert cnt == n, (cnt, n)
    m1, m2 = s1 / cnt, s2 / cnt
    var = m2 - m1 ** 2
    print(f"samples {n} (demos {len(ds.demos)}), stats {a.stats}")
    print(f"E[a^2] = {m2.mean():.5f}  -> zero-output level      1 + E[a^2]   = {1 + m2.mean():.5f}")
    print(f"Var(a) = {var.mean():.5f}  -> best-constant level   1 + mean Var = {1 + var.mean():.5f}")
    print("per dim (x y z roll pitch yaw gripper), averaged over the 8 chunk positions:")
    print("  E[a^2]  " + " ".join(f"{v:.4f}" for v in m2.mean(0)))
    print("  E[a]    " + " ".join(f"{v:+.4f}" for v in m1.mean(0)))
    print("  Var(a)  " + " ".join(f"{v:.4f}" for v in var.mean(0)))


if __name__ == "__main__":
    main()
