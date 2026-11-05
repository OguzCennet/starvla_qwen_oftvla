"""CPU tests for the VLM registry / Qwen3-VL adapter (no weights; processor only where needed).

Run:  python -m pytest -q tests/test_vlm.py
The processor-based tests need the Qwen3-VL-4B-Instruct folder (env OFTVLA_BASE_VLM, default the
path in scripts/common.sh) and are skipped if it is absent.
"""

import os

import pytest
from PIL import Image

from oftvla.vlm import get_vlm_adapter, vlm_tag
from oftvla.vlm.qwen3vl import Qwen3VLAdapter

BASE_VLM = os.environ.get("OFTVLA_BASE_VLM", "/hnvme/workspace/y307bd13-vgv/models/Qwen3-VL-4B-Instruct")
needs_processor = pytest.mark.skipif(not os.path.isdir(BASE_VLM), reason="Qwen3-VL processor files not available")

# Verbatim from starvla_cotrain_libero.yaml:40 @ 985c223
COT = ("Your task is {instruction}. To identify the key objects for your task. "
       "Locate their bounding boxes in [x1,y1,x2,y2] format.")


def test_wrap_text_matches_fork_replace():
    text = "Instruction: put the bowl on the plate\nCurrent Goal: the bowl has to be secured"
    assert Qwen3VLAdapter.wrap_text(text, None) == text  # key absent -> raw text
    assert Qwen3VLAdapter.wrap_text(text, COT) == COT.replace("{instruction}", text)  # whole text inserted
    assert Qwen3VLAdapter.wrap_text(text, COT).startswith("Your task is Instruction: put the bowl")


def test_registry_and_tag():
    with pytest.raises(NotImplementedError):
        get_vlm_adapter("/models/Qwen2.5-VL-3B-Instruct", load_model=False)
    assert vlm_tag("/hnvme/x/models/Qwen3-VL-4B-Instruct/") == "qwen3vl4b"
    assert vlm_tag("Qwen/Qwen3-VL-30B-A3B-Instruct") == "qwen3vl30ba3b"
    assert vlm_tag("Qwen3-VL-8B-Thinking") == "qwen3vl8b"
    with pytest.raises(NotImplementedError):
        vlm_tag("/models/gemma-4-12B-it")


@needs_processor
def test_special_tokens_and_prompt_ids():
    a = get_vlm_adapter(BASE_VLM, load_model=False)
    assert a.token_id("🔍") == 146663  # single token in plain Qwen3-VL-4B-Instruct (M2)
    with pytest.raises(ValueError):
        a.token_id("two words")
    imgs = [[Image.new("RGB", (224, 224)) for _ in range(2)] for _ in range(2)]
    texts = ["pick up the bowl", "open the top drawer of the cabinet and put the bowl inside"]
    raw = a.build_inputs(imgs, texts)
    cot = a.build_inputs(imgs, texts, cot_prompt=COT)
    # left padding: shorter sample is padded at the start
    assert raw["attention_mask"][0, 0] == 0 and raw["attention_mask"][1, 0] == 1
    assert (raw["input_ids"] == a.image_token_id).sum(1).tolist() == [128, 128]  # 2 images x 64 tokens at 224
    # the wrapper only adds text tokens; the image tokens are unchanged
    assert (cot["input_ids"] == a.image_token_id).sum(1).tolist() == [128, 128]
    assert cot["attention_mask"].sum(1).gt(raw["attention_mask"].sum(1)).all()
    rendered = a.processor.apply_chat_template(a.build_messages(imgs[:1], texts[:1], COT), tokenize=False,
                                               add_generation_prompt=True)[0]
    assert "Your task is pick up the bowl. To identify the key objects" in rendered
