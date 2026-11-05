"""Bitwise tests: oftvla flow-matching components vs the StarVLA fork modules @ 985c223 (CPU, fp32).

Each comparison loads the fork module's state_dict into ours, feeds identical inputs (incl. injected noise
and t), dropout off, and requires torch.equal on the loss / predictions AND on every parameter gradient
(and the input gradients) after backward.

Run:  python -m pytest -q tests/test_flow_components.py
Needs the fork (env OFTVLA_STARVLA_ROOT, default ~/VLA_FT/starVLA); skipped if absent.

Negative-control demonstration: OFTVLA_NEGCTRL=weight|mask|t applies that perturbation to OUR side in the
equality tests, which must then FAIL. (test_negative_controls_detected checks the same automatically.)
"""

import copy
import os
import sys
from unittest import mock

import pytest
import torch

from oftvla.heads.flow_matching import (
    GR00T_REPEATED_DIFFUSION_STEPS,
    PI_REPEATED_DIFFUSION_STEPS,
    FlowmatchingActionHead,
    LayerwiseFlowmatchingActionHead,
    gr00t_action_config,
    pi_action_config,
    repeat_for_diffusion,
)

FORK = os.path.expanduser(os.environ.get("OFTVLA_STARVLA_ROOT", "~/VLA_FT/starVLA"))
pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(FORK, "starVLA")), reason="StarVLA fork not found")
NEGCTRL = os.environ.get("OFTVLA_NEGCTRL")  # None in normal runs

B, L, D = 3, 10, 64  # batch before repeat, VLM tokens, VLM hidden size (small, same code paths)


def _fork():
    if FORK not in sys.path:
        sys.path.insert(0, FORK)
    from omegaconf import OmegaConf
    from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead as ForkGR00T
    from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
        LayerwiseFlowmatchingActionHead as ForkPI,
    )
    return OmegaConf, ForkGR00T, ForkPI


def _small_cfg(kind):
    if kind == "gr00t":
        return gr00t_action_config(D, hidden_size=64, num_target_vision_tokens=4, max_seq_len=16,
                                   diffusion_model_cfg={"num_layers": 4, "output_dim": 64})
    return pi_action_config(D, 4, use_canonical_forward=False, num_target_vision_tokens=4, max_seq_len=16,
                            diffusion_model_cfg={"attention_head_dim": 16})


def _build_pair(kind):
    """(fork head, our head) with identical weights, both in eval mode (dropout off)."""
    OmegaConf, ForkGR00T, ForkPI = _fork()
    cfg = _small_cfg(kind)
    fork_cfg = OmegaConf.create({"framework": {"action_model": copy.deepcopy(vars(cfg))}})
    torch.manual_seed(0)
    if kind == "gr00t":
        fork = ForkGR00T(fork_cfg)
        ours = FlowmatchingActionHead(copy.deepcopy(cfg))
    else:
        fork = ForkPI(global_config=fork_cfg)
        ours = LayerwiseFlowmatchingActionHead(copy.deepcopy(cfg))
    ours.load_state_dict(fork.state_dict(), strict=True)
    return fork.eval(), ours.eval()


def _inputs(kind, mask_kind, seed=1):
    g = torch.Generator().manual_seed(seed)
    n_vl = 1 if kind == "gr00t" else 4  # PI: one tensor per DiT block
    vl = [torch.randn(B, L, D, generator=g) for _ in range(n_vl)]
    actions = torch.rand(B, 8, 7, generator=g) * 2 - 1
    if mask_kind == "none":
        mask = None
    else:
        mask = torch.ones(B, L, dtype=torch.long)
        if mask_kind == "padded":  # left padding of different lengths
            mask[0, :3] = 0
            mask[1, :1] = 0
    R = GR00T_REPEATED_DIFFUSION_STEPS if kind == "gr00t" else PI_REPEATED_DIFFUSION_STEPS
    vl_rep, act_rep, mask_rep, _ = repeat_for_diffusion(vl if kind == "pi" else vl[0], actions, mask, None, R)
    noise = torch.randn(act_rep.shape, generator=g)
    t = torch.rand(act_rep.shape[0], generator=g)
    return vl_rep, act_rep, mask_rep, noise, t


def _perturb(ours, mask, t, perturb):
    if perturb == "weight":
        with torch.no_grad():
            ours.action_encoder.layer1.weight[0, 0] += 1e-3
    elif perturb == "mask":
        mask = torch.ones(t.shape[0], L, dtype=torch.bool) if mask is None else mask.clone()
        mask[0, 0] = ~mask[0, 0]
    elif perturb == "t":
        t = t.clone()
        t[0] = t[0] + 0.01
    return mask, t


def _leaf(x):
    return [h.clone().requires_grad_(True) for h in x] if isinstance(x, list) else x.clone().requires_grad_(True)


def _flat(x):
    return x if isinstance(x, list) else [x]


def compare_forward_backward(kind, mask_kind, perturb=None):
    """Returns the list of mismatching quantities (empty = bitwise identical)."""
    fork, ours = _build_pair(kind)
    vl, actions, mask, noise, t = _inputs(kind, mask_kind)
    vl_f, vl_o = _leaf(vl), _leaf(vl)

    with mock.patch.object(torch, "randn", side_effect=lambda *a, **k: noise.clone()) as rn:
        fork.sample_time = lambda batch_size, device, dtype: t.clone()
        loss_f = fork(vl_f, actions.clone(), None, encoder_attention_mask=mask)
        assert rn.call_count == 1
    mask_o, t_o = _perturb(ours, mask, t, perturb)
    loss_o = ours(vl_o, actions.clone(), None, encoder_attention_mask=mask_o, noise=noise.clone(), t=t_o.clone())

    bad = [] if torch.equal(loss_f, loss_o) else ["loss"]
    loss_f.backward()
    loss_o.backward()
    gf, go = dict(fork.named_parameters()), dict(ours.named_parameters())
    assert gf.keys() == go.keys()
    for name in gf:
        a, b = gf[name].grad, go[name].grad
        if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
            bad.append(f"grad:{name}")
    for i, (a, b) in enumerate(zip(_flat(vl_f), _flat(vl_o))):
        if not torch.equal(a.grad, b.grad):
            bad.append(f"grad:vl[{i}]")
    return bad


def compare_predict(kind, mask_kind, perturb=None):
    fork, ours = _build_pair(kind)
    vl, _, mask, _, t = _inputs(kind, mask_kind)
    init = torch.randn(t.shape[0], 8, 7, generator=torch.Generator().manual_seed(7))
    with mock.patch.object(torch, "randn", side_effect=lambda *a, **k: init.clone()):
        pred_f = fork.predict_action(vl, None, encoder_attention_mask=mask)
    mask_o, _ = _perturb(ours, mask, t, perturb if perturb != "t" else None)
    pred_o = ours.predict_action(vl, None, encoder_attention_mask=mask_o, init_noise=init.clone())
    return [] if torch.equal(pred_f, pred_o) else ["predict_action"]


MODES = [("gr00t", "none"), ("gr00t", "full"), ("gr00t", "padded"), ("pi", "none"), ("pi", "full"), ("pi", "padded")]


@pytest.mark.parametrize("kind,mask_kind", MODES)
def test_forward_backward_bitwise(kind, mask_kind):
    assert compare_forward_backward(kind, mask_kind, perturb=NEGCTRL) == []


@pytest.mark.parametrize("kind,mask_kind", MODES)
def test_predict_action_bitwise(kind, mask_kind):
    assert compare_predict(kind, mask_kind, perturb=NEGCTRL) == []


@pytest.mark.parametrize("kind", ["gr00t", "pi"])
@pytest.mark.parametrize("perturb", ["weight", "mask", "t"])
def test_negative_controls_detected(kind, perturb):
    """Each perturbation must break the bitwise comparison (else the comparison could not detect it)."""
    assert compare_forward_backward(kind, "padded", perturb=perturb) != []
    if perturb != "t":
        assert compare_predict(kind, "padded", perturb=perturb) != []


@pytest.mark.parametrize("kind", ["gr00t", "pi"])
@pytest.mark.parametrize("train_mode", [False, True])
def test_internal_sampling_same_rng_order(kind, train_mode):
    """No injection: same seed -> identical loss, i.e. our head draws noise, t (and dropout masks in
    train mode) in the fork's order."""
    fork, ours = _build_pair(kind)
    fork.train(train_mode)
    ours.train(train_mode)
    vl, actions, mask, _, _ = _inputs(kind, "padded")
    torch.manual_seed(123)
    loss_f = fork(vl, actions, None, encoder_attention_mask=mask)
    torch.manual_seed(123)
    loss_o = ours(vl, actions, None, encoder_attention_mask=mask)
    assert torch.equal(loss_f, loss_o)


def test_time_sampling_clamp_gr00t_only():
    fork_g, ours_g = _build_pair("gr00t")
    fork_p, ours_p = _build_pair("pi")
    for fork, ours in [(fork_g, ours_g), (fork_p, ours_p)]:
        torch.manual_seed(5)
        a = fork.sample_time(1000, "cpu", torch.float32)
        torch.manual_seed(5)
        assert torch.equal(a, ours.sample_time(1000, "cpu", torch.float32))
    big = torch.full((4,), 0.9995)
    for h in (ours_g, ours_p):
        h.beta_dist = mock.Mock(sample=lambda shape: big.clone())
    assert torch.equal(ours_g.sample_time(4, "cpu", torch.float32), torch.zeros(4))  # clamp at noise_s
    assert (ours_p.sample_time(4, "cpu", torch.float32) < 0).all()  # PI: no clamp (kept quirk)


def test_repeat_matches_fork_framework_code():
    vl = torch.randn(B, L, D)
    actions = torch.randn(B, 8, 7)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[0, :2] = 0
    R = 8
    v, a, m, s = repeat_for_diffusion(vl, actions, mask, None, R)
    # QwenGR00T.py:204-209 verbatim
    assert torch.equal(a, actions.repeat(R, 1, 1)) and torch.equal(v, vl.repeat(R, 1, 1))
    assert torch.equal(m, mask.repeat(R, 1).to(dtype=torch.bool)) and s is None
    assert torch.equal(v[B + 1], vl[1])  # copy r of sample b is row r*B + b


def test_full_size_config_and_parameters_match_fork():
    """Full LIBERO sizes (Qwen3-VL-4B: hidden 2560, 36 layers): our configs equal the fork's effective configs
    (yaml -> apply_config_compat -> framework defaults -> runtime dims), and state_dict names/shapes match."""
    OmegaConf, ForkGR00T, ForkPI = _fork()
    from starVLA.model.framework.share_tools import apply_config_compat, merge_framework_config, populate_layerwise_dit_cfg
    from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
    from starVLA.model.framework.VLM4A.QwenPI import QwenPIDefaultConfig

    yaml = os.path.join(FORK, "examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml")
    compat_only = {"future_action_window_size", "past_action_window_size"}  # added by apply_config_compat

    def fork_cfg(name, defaults):
        c = OmegaConf.load(yaml)
        c.framework.name = name
        return merge_framework_config(defaults, apply_config_compat(c))

    fg = fork_cfg("QwenGR00T", QwenGR00TDefaultConfig)
    fg.framework.action_model.diffusion_model_cfg.cross_attention_dim = 2560  # QwenGR00T.py:155-157
    fp = fork_cfg("QwenPI", QwenPIDefaultConfig)
    populate_layerwise_dit_cfg(fp, dit_hidden_dim=2560, num_dit_layers=36)  # QwenPI.py:153-157
    fp.framework.action_model.diffusion_model_cfg.use_canonical_forward = False  # run_policy_server.sh:29

    og, op = gr00t_action_config(2560), pi_action_config(2560, 36, use_canonical_forward=False)
    for fork_c, our_c in [(fg, og), (fp, op)]:
        f = {k: v for k, v in OmegaConf.to_container(fork_c.framework.action_model).items() if k not in compat_only}
        assert f == vars(our_c)

    torch.distributions.Distribution.set_default_validate_args(False)  # Beta validation calls .item()
    try:
        with torch.device("meta"):
            pairs = [(ForkGR00T(fg), FlowmatchingActionHead(og)), (ForkPI(global_config=fp), LayerwiseFlowmatchingActionHead(op))]
    finally:
        torch.distributions.Distribution.set_default_validate_args(True)
    for fork, ours in pairs:
        sf, so = fork.state_dict(), ours.state_dict()
        assert sf.keys() == so.keys()
        assert all(sf[k].shape == so[k].shape for k in sf)
    assert sum(p.numel() for p in pairs[0][1].parameters()) == 161_472_775  # spec §5
    assert sum(p.numel() for p in pairs[1][1].parameters()) == 3_352_072_737
    # legacy switch: every PI block cross-attends; GR00T interleaves
    assert all(b.attn1.is_cross_attention for b in pairs[1][1].model.transformer_blocks)
    assert [b.attn1.is_cross_attention for b in pairs[0][1].model.transformer_blocks] == [True, False] * 8
