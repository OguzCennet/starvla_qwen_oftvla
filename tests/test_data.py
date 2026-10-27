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
