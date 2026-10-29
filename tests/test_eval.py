"""CPU tests for the evaluation path: IPC encoding, client pre/post-processing, A0 texts, socket round trip.

Run:  python -m pytest -q tests/test_eval.py
"""

import csv
import threading

import numpy as np
import pytest

from oftvla import ipc
from oftvla.libero_client import a0_texts, binarize_gripper, preprocess


def test_pack_unpack_roundtrip_keeps_dtype_and_shape():
    msg = {"images": [np.arange(12, dtype=np.uint8).reshape(2, 2, 3), np.ones((3,), np.float16)],
           "lang": "pick up the bowl", "n": 3}
    out = ipc.unpack(ipc.pack(msg))
    assert out["lang"] == "pick up the bowl" and out["n"] == 3
    for a, b in zip(msg["images"], out["images"]):
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def test_packed_message_contains_no_numpy_objects():
    packed = ipc.pack({"a": np.zeros((2, 2), np.float32)})
    assert type(packed["a"]["__nd__"][0]) is bytes  # portable across numpy 1.x / 2.x


def test_binarize_gripper_convention():
    assert binarize_gripper(0.9) == -1.0  # model "open" -> LIBERO open
    assert binarize_gripper(0.1) == 1.0   # model "close" -> LIBERO close
    assert binarize_gripper(0.5) == 1.0   # threshold is strict ">" as in StarVLA


def test_preprocess_rotates_180_and_resizes():
    img = np.zeros((256, 256, 3), np.uint8)
    img[:128, :128] = 255  # top-left white
    out = preprocess(img)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    assert out[200, 200].mean() > 250 and out[20, 20].mean() < 5  # now bottom-right


def test_a0_texts_most_common_ok(tmp_path):
    p = tmp_path / "labels.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "task", "demo", "A0", "status"])
        w.writerow(["s", "t", "d0", "open the drawer", "ok"])
        w.writerow(["s", "t", "d1", "open the drawer", "ok"])
        w.writerow(["s", "t", "d2", "Open drawer", "ok"])
        w.writerow(["s", "t", "d3", "WRONG", "bad"])
    assert a0_texts(str(p)) == {("s", "t"): "open the drawer"}


def test_socket_round_trip():
    """A fake server answers like oftvla.policy.serve; checks the transport end to end."""
    port = 10931
    listener = ipc.listen(port)

    def fake_server():
        conn = listener.accept()
        msg = ipc.unpack(conn.recv())
        conn.send(ipc.pack({"actions": np.full((8, 7), float(msg["images"][0].mean()), np.float32)}))
        conn.close()
        listener.close()

    th = threading.Thread(target=fake_server)
    th.start()
    conn = ipc.connect(port)
    out = ipc.request(conn, {"images": [np.full((4, 4, 3), 7, np.uint8)], "lang": "x"})
    conn.close()
    th.join(timeout=5)
    assert out["actions"].shape == (8, 7) and out["actions"][0, 0] == pytest.approx(7.0)
