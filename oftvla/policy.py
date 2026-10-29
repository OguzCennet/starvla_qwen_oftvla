"""Inference policy: checkpoint -> unnormalised action chunks, plus a small socket server.

Matches StarVLA's LIBERO deployment (server_policy.py + policy_wrapper.py @ 985c223):
the model predicts a normalised (8, 7) chunk; arm dims are un-normalised with the TRAINING
min/max statistics; the gripper stays in model convention (1 = open, 0 = close).
Binarising the gripper and executing the chunk is the client's job (oftvla/libero_client.py).

Server usage (inside starvla.sif):
  python -m oftvla.policy --run_dir RUN --step 30000 --base_vlm ... --port 10093 [--ready_file F]
"""

import argparse
import os
from typing import List

import numpy as np
import torch

from oftvla import ipc
from oftvla.data import load_stats, unnormalize_actions
from oftvla.model import OFTConfig, QwenOFT


class Policy:
    def __init__(self, run_dir: str, step: int, base_vlm: str, device: str = "cuda"):
        self.stats = load_stats(os.path.join(run_dir, "dataset_statistics.json"))
        self.model = QwenOFT(OFTConfig(base_vlm=base_vlm)).to(device).eval()
        sd = torch.load(os.path.join(run_dir, f"steps_{step}_model.pt"), map_location="cpu")
        self.model.load_state_dict(sd, strict=True)
        del sd
        self.chunk_len = self.model.cfg.chunk_len

    def act(self, images: List[np.ndarray], lang: str) -> np.ndarray:
        """images: [main, wrist] uint8 HWC (already rotated/resized by the client) -> (8, 7) actions."""
        normalized = self.model.predict_action([{"image": images, "lang": lang}])[0]
        return unnormalize_actions(normalized, self.stats)


def serve(policy: Policy, port: int, ready_file: str = None) -> None:
    """Serve requests {'images': [...], 'lang': str} -> {'actions': (8, 7)} until a client sends {'quit': True}."""
    listener = ipc.listen(port)
    print(f"[server] listening on 127.0.0.1:{port}, chunk_len={policy.chunk_len}", flush=True)
    if ready_file:
        open(ready_file, "w").write("ready\n")
    while True:
        conn = listener.accept()
        try:
            while True:
                msg = ipc.unpack(conn.recv())
                if msg.get("quit"):
                    conn.close()
                    listener.close()
                    print("[server] quit", flush=True)
                    return
                if msg.get("meta"):
                    conn.send(ipc.pack({"chunk_len": policy.chunk_len}))
                    continue
                conn.send(ipc.pack({"actions": policy.act(msg["images"], msg["lang"])}))
        except EOFError:
            conn.close()  # client disconnected; wait for the next one


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--base_vlm", required=True)
    p.add_argument("--port", type=int, default=10093)
    p.add_argument("--ready_file", default=None)
    args = p.parse_args()
    serve(Policy(args.run_dir, args.step, args.base_vlm), args.port, args.ready_file)


if __name__ == "__main__":
    main()
