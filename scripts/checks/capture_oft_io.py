"""Capture QwenOFT inputs/outputs for fixed samples, to compare code versions bitwise.

Usage: python capture_oft_io.py OUT.pt BASE_VLM TEXT_CSV [HEAD_STATE.pt]
Samples: L10T01 demo_31 raw_step 180, text_A0 / text_AS / text_MAS; each alone and all three in one
batch (left padding). Images: fixed random uint8. If HEAD_STATE.pt exists it is loaded into the
action head, otherwise the freshly initialised head is saved there (so both runs use the same head).
"""
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image

from oftvla.model import OFTConfig, QwenOFT

out_path, base_vlm, text_csv = sys.argv[1:4]
head_path = sys.argv[4] if len(sys.argv) > 4 else out_path + ".head.pt"

row = None
with open(text_csv, newline="") as f:
    for r in csv.DictReader(f):
        if r["task_id"] == "L10T01" and r["demo"] == "demo_31" and r["raw_step"] == "180":
            row = r
            break
texts = [row["text_A0"], row["text_AS"], row["text_MAS"]]

rng = np.random.default_rng(0)
images = [[Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(2)] for _ in texts]
actions = [rng.uniform(-1, 1, (8, 7)).astype(np.float16) for _ in texts]

torch.manual_seed(0)
model = QwenOFT(OFTConfig(base_vlm=base_vlm))
if os.path.exists(head_path):
    model.action_head.load_state_dict(torch.load(head_path))
else:
    torch.save(model.action_head.state_dict(), head_path)
model = model.to("cuda").eval()

res = {}
groups = {"A0": [0], "AS": [1], "MAS": [2], "batch3": [0, 1, 2]}
for name, idx in groups.items():
    inp = model.build_inputs([images[i] for i in idx], [texts[i] for i in idx])
    for k, v in inp.items():
        res[f"{name}/in/{k}"] = v.detach().cpu()
    batch = [{"image": images[i], "lang": texts[i], "action": actions[i]} for i in idx]
    res[f"{name}/pred"] = torch.as_tensor(model.predict_action(batch))
    with torch.no_grad():
        res[f"{name}/loss"] = model(batch)["action_loss"].detach().cpu()
torch.save(res, out_path)
for k, v in res.items():
    print(k, tuple(v.shape), v.dtype)
