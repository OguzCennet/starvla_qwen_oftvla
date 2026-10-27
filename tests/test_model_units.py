"""CPU unit tests for the parts of QwenOFT that do not need the VLM.

Run:  pytest -q tests/test_model_units.py
"""

import pytest
import torch

from oftvla.model import (
    L1RegressionActionHead,
    build_prompt,
    gather_query_states,
    masked_l1_loss,
)

Q = 99  # fake query-token id
PAD = 0


def _hidden_encoding_positions(b: int, l: int, h: int) -> torch.Tensor:
    """hidden[b, t, :] = t, so a gathered vector tells us which position it came from."""
    return torch.arange(l, dtype=torch.float32).view(1, l, 1).expand(b, l, h).clone()


def test_gather_left_padding_picks_correct_positions():
    # sample 0: 3 pads; sample 1: no pads -> query positions differ per row
    ids = torch.tensor([
        [PAD, PAD, PAD, 5, 6, Q, Q, Q, 7],
        [5, 6, 8, 9, 4, Q, Q, Q, 7],
    ])
    ids[1, 2:5] = torch.tensor([1, Q, 2])  # an EARLIER query token in row 1 that must be ignored
    hidden = _hidden_encoding_positions(2, ids.shape[1], 4)
    out = gather_query_states(hidden, ids, Q, n=3)
    assert out.shape == (2, 3, 4)
    assert out[0, :, 0].tolist() == [5, 6, 7]
    assert out[1, :, 0].tolist() == [5, 6, 7]  # last 3 occurrences, not the one at position 3


def test_gather_raises_if_too_few_queries():
    ids = torch.tensor([[5, Q, Q, 7]])
    with pytest.raises(RuntimeError):
        gather_query_states(torch.zeros(1, 4, 2), ids, Q, n=3)


def test_head_shape_and_weight_sharing_over_time():
    torch.manual_seed(0)
    head = L1RegressionActionHead(input_dim=16, action_dim=7)
    x = torch.randn(2, 8, 16)
    x[:, 5] = x[:, 1]  # identical query states at two time steps ...
    y = head(x)
    assert y.shape == (2, 8, 7)
    # ... give identical actions: time information can only come from the VLM, not the head.
    assert torch.allclose(y[:, 5], y[:, 1])


def test_head_param_names_match_starvla():
    head = L1RegressionActionHead(input_dim=16, action_dim=7)
    names = set(head.state_dict().keys())
    expected = {
        "model.layer_norm1.weight", "model.layer_norm1.bias",
        "model.fc1.weight", "model.fc1.bias",
        "model.layer_norm2.weight", "model.layer_norm2.bias",
        "model.fc2.weight", "model.fc2.bias",
    }
    for i in range(2):
        for j in (0, 1):  # ffn.0 = LayerNorm, ffn.1 = Linear
            expected |= {f"model.mlp_resnet_blocks.{i}.ffn.{j}.weight", f"model.mlp_resnet_blocks.{i}.ffn.{j}.bias"}
    assert names == expected
    assert head.model.fc1.out_features == 32  # hidden = 2 * input_dim


def test_masked_l1():
    pred = torch.zeros(1, 2, 2)
    target = torch.tensor([[[1.0, 1.0], [3.0, 3.0]]])
    assert masked_l1_loss(pred, target).item() == pytest.approx(2.0)
    mask = torch.tensor([[[True, True], [False, False]]])
    assert masked_l1_loss(pred, target, mask).item() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        masked_l1_loss(pred, target, torch.zeros_like(mask))


def test_prompt_has_contiguous_queries():
    p = build_prompt("pick up the bowl", 8, "🔍")
    assert p.startswith("pick up the bowl ")
    assert "<action>" + "🔍" * 8 + "<action>." in p
    assert p.count("🔍") == 8
