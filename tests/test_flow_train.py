"""CPU tests for the flow trainer's data stream and optimiser setup (no GPU, no VLM).

The config-level comparison with the fork's real trainer code runs in scripts/checks/flow_trainer_config.py.
"""

import torch
import torch.nn as nn

from oftvla.flow_data import GlobalStream, StreamBatchSampler
import json
import os

from oftvla.train_flow import build_optimizer_and_scheduler, lost_progress, parse_args, recover_resume_swap

BASE = ["--run_dir", "x", "--data_root", "x", "--train_ids", "x", "--text_csv", "x", "--setup", "MAS", "--base_vlm", "x"]


def _take(sampler, n):
    it = iter(sampler)
    return [next(it) for _ in range(n)]


def test_stream_is_a_permutation_per_epoch_and_reshuffles():
    s = GlobalStream(10, seed=3)
    e0 = [s.index(k) for k in range(10)]
    e1 = [s.index(k) for k in range(10, 20)]
    assert sorted(e0) == sorted(e1) == list(range(10))
    assert e0 != e1  # new order each epoch
    assert e0 == [GlobalStream(10, seed=3).index(k) for k in range(10)]  # deterministic
    assert e0 != [GlobalStream(10, seed=4).index(k) for k in range(10)]


def test_resume_continues_exactly():
    full = _take(StreamBatchSampler(1000, 42, world=2, rank=1, per_device=4), 50)
    resumed = _take(StreamBatchSampler(1000, 42, world=2, rank=1, per_device=4, start_draw=30), 20)
    assert resumed == full[30:]


def test_ranks_partition_each_draw_and_layouts_cover_the_same_stream():
    n, seed = 997, 7
    s = GlobalStream(n, seed)
    for world, per in ((8, 16), (16, 8), (1, 128)):
        draws = [_take(StreamBatchSampler(n, seed, world, r, per), 3) for r in range(world)]
        flat = [i for d in range(3) for r in range(world) for i in draws[r][d]]
        assert flat == [s.index(k) for k in range(3 * world * per)]


def test_head_defaults():
    g, p = parse_args(["--head", "gr00t"] + BASE), parse_args(["--head", "pi"] + BASE)
    assert (g.per_device_batch, g.max_train_steps) == (16, 80000)
    assert (p.per_device_batch, p.max_train_steps) == (8, 100000)
    assert (g.lr_base, p.lr_base) == (2.5e-5, 3e-5)  # fork yaml / released PI run
    assert parse_args(["--head", "pi", "--lr_base", "2.5e-5"] + BASE).lr_base == 2.5e-5
    for a in (g, p):
        assert (a.global_batch, a.stop_at, a.warmup_steps) == (128, 30000, 5000)
        assert (a.lr_vlm, a.lr_head, a.min_lr) == (1e-5, 1e-4, 1e-6)
        assert (a.weight_decay, tuple(a.betas), a.eps) == (1e-8, (0.9, 0.95), 1e-8)
        assert a.save_steps == [10000, 20000, 30000]


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = nn.Linear(3, 3)
        self.action_head = nn.Linear(3, 2)


def test_groups_order_and_schedule():
    a = parse_args(["--head", "gr00t"] + BASE)
    opt, sched = build_optimizer_and_scheduler(_Toy(), a)
    assert [g["name"] for g in opt.param_groups] == ["qwen_vl_interface", "action_model"]  # no 'base' group
    lrs = {}
    for s in range(30001):
        if s in (0, 5000, 30000):
            lrs[s] = sched.get_last_lr()
        sched.step()
    assert lrs[0] == [0.0, 0.0]
    assert lrs[5000] == [1e-5, 1e-4]
    assert abs(lrs[30000][0] / 1e-5 - 0.76) < 1e-9  # 80K cosine: 0.04 + 0.96 * 0.5 * (1 + cos(pi / 3))


def _run_dir(tmp_path, resume=False, old=False, log_steps=(), ckpt=False):
    d = tmp_path / "run"
    d.mkdir()
    for name, on in (("resume", resume), ("resume.old", old)):
        if on:
            (d / name).mkdir()
            (d / name / "trainer_state.json").write_text(json.dumps({"from": name}))
    if log_steps:
        (d / "log.jsonl").write_text("".join(json.dumps({"step": s, "loss": 1.0}) + "\n" for s in log_steps))
    if ckpt:
        (d / "steps_10000_model.pt").write_bytes(b"x")
    paths = {k: str(d / v) for k, v in dict(log="log.jsonl", resume="resume", resume_old="resume.old").items()}
    return str(d), paths


def test_recover_resume_old_and_resume(tmp_path):  # crash while deleting resume.old
    d, p = _run_dir(tmp_path, resume=True, old=True, log_steps=[10, 20])
    assert "deleted" in recover_resume_swap(p)
    assert not os.path.exists(p["resume_old"])
    assert json.load(open(os.path.join(p["resume"], "trainer_state.json"))) == {"from": "resume"}
    assert lost_progress(p, d) is None


def test_recover_only_resume_old(tmp_path):  # crash between the two renames
    d, p = _run_dir(tmp_path, old=True, log_steps=[10, 20])
    assert "renamed" in recover_resume_swap(p)
    assert not os.path.exists(p["resume_old"])
    assert json.load(open(os.path.join(p["resume"], "trainer_state.json"))) == {"from": "resume.old"}
    assert lost_progress(p, d) is None


def test_no_resume_but_logged_steps(tmp_path):
    d, p = _run_dir(tmp_path, log_steps=[0, 10, 20])
    assert recover_resume_swap(p) is None
    assert "up to step 20" in lost_progress(p, d)


def test_no_resume_but_weights(tmp_path):
    d, p = _run_dir(tmp_path, ckpt=True)
    assert recover_resume_swap(p) is None
    assert "steps_10000_model.pt" in lost_progress(p, d)


def test_fresh_run_dir_starts(tmp_path):
    d, p = _run_dir(tmp_path)
    assert recover_resume_swap(p) is None and lost_progress(p, d) is None
    (tmp_path / "y").mkdir()
    d2, p2 = _run_dir(tmp_path / "y", log_steps=[0])
    assert lost_progress(p2, d2) is None  # a log with only step 0 is not progress
