"""Preflight for the 6 flow runs: first N samples per head, in the head's GPU layout, for A0 / AS / MAS.

Datasets are built exactly as train_flow.make_ds (E3 train ids, test demos forbidden, A0ASMAS train.csv); the order is
walked draw by draw over ranks (GR00T 8 x 16, PI 16 x 8). Checks indices and (demo, t) identical across setups, texts
different, and the rank-0 indices against the smoke runs' logs (MAS). CPU only, ~30 s.
Usage: run_py scripts/checks/flow_data_order_preflight.py
"""
import hashlib, json, os, sys
from oftvla.data import LiberoLangDataset, load_stats
from oftvla.flow_data import StreamBatchSampler
H = os.path.expanduser
DATA = "/hnvme/workspace/y307bd13-vgv/data/libero_lang/v1"; SPL = H("~/VLA_FT/DataSamples/Libero/Splits")
CSV = H("~/VLA_FT/DataSamples/Libero/A0ASMAS/train.csv"); RUNS = "/hnvme/workspace/y307bd13-vgv/runs/oftvla"
N = 1000
stats = load_stats(f"{RUNS}/smoke2_gr00t_qwen3vl4b_MAS/dataset_statistics.json")
ds = {s: LiberoLangDataset(DATA, [f"{SPL}/e3_train_ids.csv"], None, s, stats=stats,
                           forbidden_id_files=[f"{SPL}/e3_testdemo_ids.csv"], text_csv=CSV) for s in ("A0", "AS", "MAS")}
for s, d in ds.items():
    print(f"{s}: demos {len(d.demos)} samples {len(d)} skipped_not_in_text {d.n_skipped_not_in_text}")
for head, W, Pd, smoke in (("gr00t", 8, 16, "smoke2_gr00t_qwen3vl4b_MAS"), ("pi", 16, 8, "smoke3_pi_qwen3vl4b_MAS")):
    res = {}
    for s, d in ds.items():
        samp = [iter(StreamBatchSampler(len(d), 42, W, r, Pd)) for r in range(W)]
        idx = []
        while len(idx) < N:                       # draw d: ranks 0..W-1 in order
            for it in samp:
                idx += next(it)
        idx = idx[:N]
        frames = [(*d.demos[d.starts[i][0]]["key"], d.starts[i][1]) for i in idx]
        txt = [d.texts[d.demos[d.starts[i][0]]["text_idx"][d.starts[i][1]]] for i in idx]
        res[s] = (idx, frames, txt)
        print(f"  {head} {W}x{Pd} {s}: first {N} indices md5 {hashlib.md5(repr(idx).encode()).hexdigest()[:12]}, "
              f"(demo,t) md5 {hashlib.md5(repr(frames).encode()).hexdigest()[:12]}, first 5 {idx[:5]}")
    same = res["A0"][:2] == res["AS"][:2] == res["MAS"][:2]
    dt = [sum(a != b for a, b in zip(res[x][2], res[y][2])) for x, y in (("A0", "AS"), ("A0", "MAS"), ("AS", "MAS"))]
    print(f"[{'PASS' if same else 'FAIL'}] {head}: first {N} indices and (demo,t) identical for A0/AS/MAS; "
          f"texts differ A0-AS {dt[0]}, A0-MAS {dt[1]}, AS-MAS {dt[2]} of {N}")
    # cross-check vs the smoke run's logged rank-0 indices (MAS, same layout)
    logged = [json.loads(l) for l in open(f"{RUNS}/{smoke}/log.jsonl")]
    logged = {r["draws"]: r["idx_rank0"] for r in logged if "idx_rank0" in r}
    k = max(1, N // (W * Pd) + 1)
    ok = all(logged.get(dd + 1) == res["MAS"][0][dd * W * Pd: dd * W * Pd + Pd] for dd in range(min(k, N // (W * Pd))))
    print(f"[{'PASS' if ok else 'FAIL'}] {head}: rank-0 indices of draws 1..{N // (W * Pd)} match {smoke}/log.jsonl")
