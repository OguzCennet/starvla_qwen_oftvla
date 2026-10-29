"""CPU tests for oftvla.data on a tiny synthetic HDF5 dataset.

Run:  python -m pytest -q tests/test_data.py
"""

import csv

import h5py
import numpy as np
import pytest

from oftvla.data import (
    LiberoLangDataset,
    chunk_indices,
    normalize_actions,
    remap_gripper,
    unnormalize_actions,
)


def test_chunk_end_padding_repeats_last_step():
    assert chunk_indices(0, 20).tolist() == list(range(8))
    assert chunk_indices(15, 20).tolist() == [15, 16, 17, 18, 19, 19, 19, 19]


def test_gripper_remap():
    a = np.zeros((3, 7))
    a[:, -1] = [-1.0, 1.0, 0.3]  # LIBERO: -1 open, +1 close
    assert remap_gripper(a)[:, -1].tolist() == pytest.approx([1.0, 0.0, 0.7])


def test_normalize_roundtrip_and_range():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 3, (100, 7)).astype(np.float32)
    x[:, -1] = rng.integers(0, 2, 100)
    stats = {"min": x.min(0).tolist(), "max": x.max(0).tolist()}
    y = normalize_actions(x, stats)
    assert y[:, :6].min() == pytest.approx(-1) and y[:, :6].max() == pytest.approx(1)
    assert np.array_equal(y[:, 6], x[:, 6])  # gripper untouched
    assert np.allclose(unnormalize_actions(y, stats), x, atol=1e-5)


def _make_dataset(tmp_path):
    """Two tasks: 'train_task' (2 demos, one not ok) and 'test_task' (1 demo)."""
    root = tmp_path / "v1" / "libero_90"
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for task, demos in (("train_task", ["demo_0", "demo_1"]), ("test_task", ["demo_0"])):
        with h5py.File(root / f"{task}.hdf5", "w") as f:
            for i, d in enumerate(demos):
                T = 12 + i
                g = f.create_group(d)
                g["action"] = rng.uniform(-1, 1, (T, 7)).astype(np.float32)
                g["agentview"] = rng.integers(0, 255, (T, 256, 256, 3), dtype=np.uint8)
                g["wrist"] = rng.integers(0, 255, (T, 256, 256, 3), dtype=np.uint8)
    rows = [("libero_90", "train_task", "demo_0", "ok"), ("libero_90", "train_task", "demo_1", "bad"),
            ("libero_90", "test_task", "demo_0", "ok")]
    with open(tmp_path / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "task", "demo", "A0", "status"])
        for s, t, d, st in rows:
            w.writerow([s, t, d, f"do {t}", st])
    for name, sel in (("train.csv", rows[:2]), ("test.csv", rows[2:])):
        with open(tmp_path / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["suite", "task", "demo"])
            for s, t, d, _ in sel:
                w.writerow([s, t, d])
    return tmp_path


def test_dataset_sample_format(tmp_path):
    p = _make_dataset(tmp_path)
    ds = LiberoLangDataset(str(p / "v1"), [str(p / "train.csv")], str(p / "labels.csv"),
                           forbidden_id_files=[str(p / "test.csv")])
    assert len(ds.demos) == 1 and ds.n_skipped_not_ok == 1
    assert len(ds) == 12  # every step of the ok demo is a start
    s = ds[11]  # last start: fully padded chunk
    assert s["lang"] == "do train_task"
    assert [im.size for im in s["image"]] == [(224, 224), (224, 224)]
    assert s["action"].shape == (8, 7) and s["action"].dtype == np.float16
    assert np.all(s["action"] == s["action"][0])  # all 8 rows = last action
    assert np.abs(s["action"][:, :6]).max() <= 1.0 + 1e-3


def test_forbidden_task_raises(tmp_path):
    p = _make_dataset(tmp_path)
    with pytest.raises(RuntimeError):
        LiberoLangDataset(str(p / "v1"), [str(p / "train.csv"), str(p / "test.csv")], str(p / "labels.csv"),
                          forbidden_id_files=[str(p / "test.csv")])


def test_only_a0_supported(tmp_path):
    p = _make_dataset(tmp_path)
    with pytest.raises(ValueError):
        LiberoLangDataset(str(p / "v1"), [str(p / "train.csv")], str(p / "labels.csv"), setup="A1")


# ──────────────────────────────────────────────────────────────────────
# Per-frame texts from the A0ASMAS CSV (text_csv)
# ──────────────────────────────────────────────────────────────────────
def _write_text_csv(path, rows):
    cols = ["split", "suite", "task_id", "task", "demo", "t", "raw_step", "text_A0", "text_AS", "text_MAS"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _frame_rows(suite, task, demo, length, as_switch, mas_switches):
    """AS: 2 steps switching at as_switch; MAS: one step index per switch. Multi-line texts as in the CSV."""
    a0 = f"Instruction: do {task}"
    rows = []
    for t in range(length):
        k_as = int(t >= as_switch)
        k_mas = sum(t >= s for s in mas_switches)
        rows.append(dict(split="train", suite=suite, task_id="T1", task=task, demo=demo, t=t, raw_step=t,
                         text_A0=a0, text_AS=f"{a0}\nIn-Progress Step: as{k_as}",
                         text_MAS=f"{a0}\nIn-Progress Step: mas{k_mas}"))
    return rows


def _make_text_dataset(tmp_path):
    """train_task demo_0 (12 steps) is in the text CSV; demo_1 (13 steps) is not (a v5-dropped demo)."""
    p = _make_dataset(tmp_path)
    with open(p / "train_both.csv", "w", newline="") as f:  # split file with both demos (status ignored here)
        w = csv.writer(f)
        w.writerow(["suite", "task", "demo"])
        w.writerow(["libero_90", "train_task", "demo_0"])
        w.writerow(["libero_90", "train_task", "demo_1"])
    _write_text_csv(p / "a0asmas.csv", _frame_rows("libero_90", "train_task", "demo_0", 12, 5, [3, 5, 9]))
    return p


def _text_ds(p, setup, **kw):
    return LiberoLangDataset(str(p / "v1"), [str(p / "train_both.csv")], None, setup=setup,
                             text_csv=str(p / "a0asmas.csv"), **kw)


def test_text_csv_per_frame_text(tmp_path):
    p = _make_text_dataset(tmp_path)
    ds = _text_ds(p, "MAS")
    assert len(ds.demos) == 1 and ds.n_skipped_not_in_text == 1  # demo_1 not in the CSV -> skipped
    assert len(ds) == 12
    langs = [ds[t]["lang"] for t in range(12)]
    expected = [f"mas{sum(t >= s for s in (3, 5, 9))}" for t in range(12)]
    assert [s.split("In-Progress Step: ")[1] for s in langs] == expected
    assert langs[0].startswith("Instruction: do train_task\n")  # instruction line kept, real newline
    assert len(ds.texts) == 4  # distinct texts stored once


def test_text_csv_same_samples_for_all_setups(tmp_path):
    """Fairness: A0 / AS / MAS see identical demos, starts, images and actions; only the text differs."""
    p = _make_text_dataset(tmp_path)
    dss = {s: _text_ds(p, s) for s in ("A0", "AS", "MAS")}
    ref = dss["A0"]
    assert len({d["text_idx"].size for d in ref.demos}) == 1
    for name, ds in dss.items():
        assert ds.starts == ref.starts and [d["key"] for d in ds.demos] == [d["key"] for d in ref.demos]
        assert ds.stats == ref.stats
        for i in (0, 4, 5, 11):
            a, b = ds[i], ref[i]
            assert np.array_equal(a["action"], b["action"])
            assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a["image"], b["image"]))
    assert {dss["A0"][t]["lang"] for t in range(12)} == {"Instruction: do train_task"}
    assert dss["AS"][4]["lang"].endswith("as0") and dss["AS"][5]["lang"].endswith("as1")


def test_text_csv_missing_frame_raises(tmp_path):
    p = _make_text_dataset(tmp_path)
    rows = _frame_rows("libero_90", "train_task", "demo_0", 12, 5, [3, 5, 9])
    _write_text_csv(p / "a0asmas.csv", rows[:7] + rows[8:])  # frame 7 missing
    with pytest.raises(ValueError, match="HDF5 demo has 12 steps"):
        _text_ds(p, "AS")


def test_text_csv_bad_setup_raises(tmp_path):
    p = _make_text_dataset(tmp_path)
    with pytest.raises(ValueError):
        _text_ds(p, "A1")
