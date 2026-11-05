"""VLM registry: base_vlm path/name -> adapter.

Mirrors StarVLA's ``get_vlm_model`` (starVLA/model/modules/vlm/__init__.py:1-45 @ 985c223), which picks
the backbone wrapper by a substring of ``base_vlm``. Only the Qwen3-VL family is implemented here.

Adapter interface (what action heads may rely on; nothing else):
  adapter.model                         the HF model (frameworks register it as their ``vlm`` submodule)
  adapter.processor                     the HF processor (tokenizer padding side: left)
  adapter.hidden_size, adapter.num_layers   read from the loaded config
  adapter.token_id(token)               id of a single-token string (e.g. the OFT action query)
  adapter.image_token_id                id of the image placeholder token
  adapter.wrap_text(text, cot_prompt)   the fork's CoT_prompt wrapper (None = no wrapper)
  adapter.build_inputs(images, texts, cot_prompt=None)  tokenised batch (input_ids, attention_mask, pixels, ...)
  adapter.forward(inputs)               -> (all hidden states, attention_mask)
"""

import os
import re


def get_vlm_adapter(base_vlm: str, **kwargs):
    """Adapter for ``base_vlm`` (a local path or hub id), chosen by family name as in the fork."""
    if "Qwen3-VL" in base_vlm:
        from oftvla.vlm.qwen3vl import Qwen3VLAdapter

        return Qwen3VLAdapter(base_vlm, **kwargs)
    raise NotImplementedError(f"VLM {base_vlm!r} not implemented (only Qwen3-VL)")


def vlm_tag(base_vlm: str) -> str:
    """Short backbone tag for run-directory names, e.g. '.../Qwen3-VL-4B-Instruct' -> 'qwen3vl4b'.

    Runs with different backbones must never share a label in the analysis (labels come from folder
    names), so the tag is derived from the backbone name and refuses names it cannot tag.
    """
    name = os.path.basename(os.path.normpath(base_vlm))
    if name.startswith("Qwen3-VL-"):
        size = name[len("Qwen3-VL-"):].split("-Instruct")[0].split("-Thinking")[0]  # "4B", "30B-A3B"
        if size:
            return "qwen3vl" + re.sub(r"[^0-9a-z]", "", size.lower())
    raise NotImplementedError(f"no VLM tag defined for {name!r}")
