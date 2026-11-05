"""CPU tests for the flow trainer's data stream and optimiser setup (no GPU, no VLM).

The config-level comparison with the fork's real trainer code runs in scripts/checks/flow_trainer_config.py.
"""

import torch
import torch.nn as nn

from oftvla.flow_data import GlobalStream, StreamBatchSampler
from oftvla.train_flow import build_optimizer_and_scheduler, parse_args

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
    for a in (g, p):
        assert (a.global_batch, a.stop_at, a.warmup_steps) == (128, 30000, 5000)
        assert (a.lr_base, a.lr_vlm, a.lr_head, a.min_lr) == (2.5e-5, 1e-5, 1e-4, 1e-6)
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
