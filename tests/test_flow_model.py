"""CPU tests for the GR00T / PI framework layer that need no VLM weights.

The full-size forward / gradient equivalence against the fork runs on a GPU:
scripts/checks/flow_equivalence.py (see scripts/checks/README.md).
"""

import os

import pytest
import torch

from oftvla.flow_model import COT_PROMPT, FlowConfig, remap_starvla_state_dict

FORK = os.path.expanduser(os.environ.get("OFTVLA_STARVLA_ROOT", "~/VLA_FT/starVLA"))
YAML = os.path.join(FORK, "examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml")


@pytest.mark.skipif(not os.path.exists(YAML), reason="StarVLA fork not found")
def test_cot_prompt_is_the_fork_yaml_value():
    from omegaconf import OmegaConf

    assert COT_PROMPT == OmegaConf.load(YAML).datasets.vla_data.CoT_prompt


def test_defaults_follow_decisions():
    for head in ("gr00t", "pi"):
        c = FlowConfig(head=head)
        assert c.cot_prompt == COT_PROMPT  # decision 11.4
        assert c.attn_implementation == "sdpa"  # QWen3.py:52-53
        assert c.vlm_dtype == "bfloat16"
        assert c.obs_image_size is None  # LIBERO yaml: no obs_image_size
    assert FlowConfig(head="pi").pi_use_canonical_forward is False  # decision 11.1 (legacy switch)


def test_state_dict_remap():
    sd = {"qwen_vl_interface.model.model.x": torch.zeros(1), "action_model.model.y": torch.ones(1), "other": torch.ones(2)}
    out = remap_starvla_state_dict(sd)
    assert set(out) == {"vlm.model.x", "action_head.model.y", "other"}
