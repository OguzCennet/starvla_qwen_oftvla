"""Print the exact prompt QwenOFT receives for A0 / AS / MAS at one frame.

Uses the VLM adapter's own prompt building (oftvla.vlm, processor only, no weights) with
oftvla.model.build_prompt, i.e. exactly what QwenOFT.build_inputs feeds the VLM (OFT: no
CoT_prompt). Images are blank 224x224 images; the image-token count does not depend on content.

Run inside the container (no GPU needed):
  source scripts/common.sh
  run_py scripts/show_prompt.py --base_vlm $BASE_VLM \
      --csv $HOME/VLA_FT/DataSamples/Libero/A0ASMAS/train.csv --task_id L10T01 --demo demo_31 --raw_step 180
"""

import argparse
import csv
import os

import torch

from PIL import Image

from oftvla.model import OFTConfig, build_prompt
from oftvla.vlm import get_vlm_adapter


def find_row(csv_path: str, task_id: str, demo: str, raw_step: int) -> dict:
    with open(os.path.expanduser(csv_path)) as f:
        for r in csv.DictReader(f):
            if r["task_id"] == task_id and r["demo"] == demo and int(r["raw_step"]) == raw_step:
                return r
    raise KeyError(f"no row for {task_id} {demo} raw_step {raw_step}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--task_id", default="L10T01")
    ap.add_argument("--demo", default="demo_31")
    ap.add_argument("--raw_step", type=int, default=180)
    ap.add_argument("--base_vlm", default=OFTConfig().base_vlm)
    ap.add_argument("--save_ids", default=None, help="optional .pt file: {column: input_ids} of the prompts shown")
    args = ap.parse_args()

    cfg = OFTConfig(base_vlm=args.base_vlm)
    adapter = get_vlm_adapter(cfg.base_vlm, load_model=False)
    row = find_row(args.csv, args.task_id, args.demo, args.raw_step)
    images = [Image.new("RGB", cfg.obs_image_size) for _ in range(2)]  # front, wrist
    pad_id = adapter.image_token_id

    saved = {}
    for col in ("text_A0", "text_AS", "text_MAS"):
        text = build_prompt(row[col], cfg.chunk_len, cfg.query_token)
        # Same path as QwenOFT.build_inputs (cot_prompt=None), once as text and once tokenised.
        messages = adapter.build_messages([images], [text], cot_prompt=None)
        as_text = adapter.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[0]
        ids = adapter.build_inputs([images], [text], cot_prompt=None)["input_ids"][0]
        n_img = int((ids == pad_id).sum())
        print(f"===== {col}  ({args.task_id} {args.demo} raw_step {args.raw_step}, v1 t={row['t']})")
        print(as_text, end="" if as_text.endswith("\n") else "\n")
        print(f"----- tokens: total {len(ids)} | image-pad tokens {n_img} (both images) | "
              f"text+template {len(ids) - n_img}\n")
        saved[col] = ids
    if args.save_ids:
        torch.save(saved, args.save_ids)


if __name__ == "__main__":
    main()
