"""CPU tests for the pure parts of oftvla.heldout_eval (metrics, setup guard, test-CSV path).

Run:  python -m pytest -q tests/test_heldout_eval.py
"""

import json

import numpy as np
import pytest

from oftvla.heldout_eval import frame_metrics, resolve_setup, resolve_text_csv


def test_frame_metrics_values():
    gt = np.zeros((2, 8, 7), np.float32)
    gt[:, :, 6] = 1.0                       # gripper open in the demo
    pred = gt.copy()
    pred[0, :, :6] += 0.1                   # sample 0: arm off by 0.1 everywhere, gripper right
    pred[1, :4, 6] = 0.0                    # sample 1: arm right, gripper closed in the first 4 steps
    m = frame_metrics(pred, gt)
    assert m["l1"] == pytest.approx([0.1 * 6 / 7, 4 / 56])
    assert m["l1_arm"] == pytest.approx([0.1, 0.0])
    assert m["l1_first"] == pytest.approx([0.1 * 6 / 7, 1 / 7])
    assert m["grip_agree"] == pytest.approx([1.0, 0.5])


def test_frame_metrics_mean_equals_training_loss():
    rng = np.random.default_rng(0)
    pred, gt = rng.normal(size=(5, 8, 7)), rng.normal(size=(5, 8, 7))
    assert frame_metrics(pred, gt)["l1"].mean() == pytest.approx(np.abs(pred - gt).mean(), rel=1e-5)


def test_resolve_setup_uses_training_setup(tmp_path, capsys):
    (tmp_path / "args.json").write_text(json.dumps({"setup": "MAS"}))
    assert resolve_setup(str(tmp_path), None) == "MAS"
    assert resolve_setup(str(tmp_path), "AS") == "AS"  # explicit override is allowed ...
    assert "trained with 'MAS'" in capsys.readouterr().out  # ... but warned about


def test_resolve_text_csv_from_training_csv(tmp_path):
    (tmp_path / "args.json").write_text(json.dumps({"text_csv": "/data/A0ASMAS_noAG/train.csv"}))
    assert resolve_text_csv(str(tmp_path), None) == "/data/A0ASMAS_noAG/test.csv"
    assert resolve_text_csv(str(tmp_path), "/other/test.csv") == "/other/test.csv"  # explicit wins


def test_resolve_text_csv_missing_raises(tmp_path):
    (tmp_path / "args.json").write_text(json.dumps({"labels_csv": "x.csv", "text_csv": None}))
    with pytest.raises(ValueError, match="--text_csv"):
        resolve_text_csv(str(tmp_path), None)
