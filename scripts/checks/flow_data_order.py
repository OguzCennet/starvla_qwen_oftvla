"""Same seed -> same sample order for the A0 / AS / MAS runs of a head (first N samples of the global stream).

Builds the three training datasets (E3 v5 train split, A0ASMAS train.csv), maps the first N stream positions
(oftvla.flow_data, seed 42) to (demo key, t) and checks that the three lists are identical while the texts
differ. Also checks the per-rank split for a layout (world x per_device) and a resume at draw d.
Usage: python scripts/checks/flow_data_order.py --data_root $DATA --train_ids $SPLITS/e3_train_ids.csv \
           --text_csv $TEXTS/train.csv --stats $RUNS/e3v5_A0/dataset_statistics.json [--n 1000]
"""
import argparse
import hashlib

from oftvla.data import LiberoLangDataset, load_stats
from oftvla.flow_data import GlobalStream, StreamBatchSampler

ap = argparse.ArgumentParser()
for k in ("data_root", "train_ids", "text_csv", "stats"):
    ap.add_argument(f"--{k}", required=True)
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
stats = load_stats(a.stats)
keys, texts = {}, {}
for setup in ("A0", "AS", "MAS"):
    ds = LiberoLangDataset(a.data_root, [a.train_ids], None, setup, stats=stats, text_csv=a.text_csv)
    st = GlobalStream(len(ds), a.seed)
    order = [st.index(k) for k in range(a.n)]
    keys[setup] = [(*ds.demos[ds.starts[i][0]]["key"], ds.starts[i][1]) for i in order]
    texts[setup] = [ds.texts[ds.demos[ds.starts[i][0]]["text_idx"][ds.starts[i][1]]] for i in order]
    h = hashlib.md5(repr(keys[setup]).encode()).hexdigest()
    print(f"{setup}: samples {len(ds)}, first {a.n} (demo, t) md5 {h}; first 3: {keys[setup][:3]}")
same = keys["A0"] == keys["AS"] == keys["MAS"]
diff_text = sum(x != y for x, y in zip(texts["A0"], texts["MAS"]))
print(f"[{'PASS' if same else 'FAIL'}] first {a.n} samples identical for A0 / AS / MAS (frames and order); "
      f"texts differ A0 vs MAS in {diff_text}/{a.n}")
# layout: GR00T 8 x 16, PI 16 x 8 -> rank r of draw d holds stream positions (d*W + r)*P ...
for W, P in ((8, 16), (16, 8)):
    flat = [i for d in range(a.n // (W * P) + 1) for r in range(W) for i in StreamBatchSampler(len(ds), a.seed, W, r, P, d).__iter__().__next__()]
    print(f"[{'PASS' if flat[:a.n] == [st.index(k) for k in range(a.n)] else 'FAIL'}] layout {W} x {P}: ranks together cover the stream in order")
