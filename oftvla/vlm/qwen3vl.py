"""Qwen3-VL adapter: loading, prompt building and the backbone forward.

Follows StarVLA's ``_QWen3_VL_Interface`` (starVLA/model/modules/vlm/QWen3.py @ 985c223):
  * attention implementation: the fork forces "sdpa" whatever the yaml says (QWen3.py:52-53);
  * weights loaded in bf16 (QWen3.py:60-65; oftvla training passes float32 explicitly);
  * tokenizer padding side "left" (QWen3.py:67);
  * one user turn per sample: [image]...[image][text], chat template with the generation prompt,
    padded batch (QWen3.py:120-144);
  * optional CoT_prompt: ``CoT_prompt.replace("{instruction}", text)`` (QWen3.py:126-128). The fork
    applies it whenever the key exists in datasets.vla_data; here ``cot_prompt=None`` means "key absent".
  * forward under bf16 autocast, all hidden states returned (QWen3.py:85-95; QwenOFT.py / QwenGR00T.py /
    QwenPI.py call it with output_hidden_states=True, return_dict=True).
The fork also sets ``model.config.hidden_size = text_config.hidden_size`` (QWen3.py:74); here
``hidden_size`` / ``num_layers`` are read from ``text_config`` directly instead (same values).
"""

from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class Qwen3VLAdapter:
    family = "qwen3vl"

    def __init__(self, base_vlm: str, attn_implementation: str = "sdpa", dtype: str = "bfloat16",
                 load_model: bool = True):
        self.base_vlm = base_vlm
        self.model = None
        if load_model:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_vlm, attn_implementation=attn_implementation, dtype=getattr(torch, dtype)
            )
        self.processor = AutoProcessor.from_pretrained(base_vlm)
        self.processor.tokenizer.padding_side = "left"

    # --- sizes and special tokens --------------------------------------
    @property
    def _text_config(self):
        if self.model is None:
            raise RuntimeError("adapter built with load_model=False has no config")
        return self.model.config.text_config

    @property
    def hidden_size(self) -> int:
        return int(self._text_config.hidden_size)

    @property
    def num_layers(self) -> int:
        return int(self._text_config.num_hidden_layers)

    def token_id(self, token: str) -> int:
        """Id of a string that must be exactly one token (StarVLA silently takes ids[0]; we refuse)."""
        ids = self.processor.tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"Query token {token!r} is {len(ids)} tokens ({ids}) in {self.base_vlm}")
        return ids[0]

    @property
    def image_token_id(self) -> int:
        return int(self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))

    # --- prompt ---------------------------------------------------------
    @staticmethod
    def wrap_text(text: str, cot_prompt: Optional[str] = None) -> str:
        """QWen3.py:126-130: with a CoT_prompt the whole text replaces '{instruction}'."""
        return text if cot_prompt is None else cot_prompt.replace("{instruction}", text)

    def build_messages(self, images: Sequence[List[Image.Image]], texts: Sequence[str],
                       cot_prompt: Optional[str] = None) -> list:
        messages = []
        for imgs, text in zip(images, texts, strict=True):
            content = [{"type": "image", "image": im} for im in imgs]
            content.append({"type": "text", "text": self.wrap_text(text, cot_prompt)})
            messages.append([{"role": "user", "content": content}])
        return messages

    def build_inputs(self, images: Sequence[List[Image.Image]], texts: Sequence[str],
                     cot_prompt: Optional[str] = None):
        batch = self.processor.apply_chat_template(
            self.build_messages(images, texts, cot_prompt),
            tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        )
        return batch if self.model is None else batch.to(self.model.device)

    # --- forward --------------------------------------------------------
    def forward(self, inputs) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """All hidden states (embeddings, layer outputs, final normed output last) and the attention mask."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=inputs["input_ids"].is_cuda):
            out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        return out.hidden_states, inputs["attention_mask"]
