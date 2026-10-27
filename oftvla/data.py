"""LIBERO-lang HDF5 dataset producing StarVLA-format samples.

Data source: our regenerated LIBERO data (one HDF5 per task, 180°-rotated 256x256 images),
demo selection by split-ID files, A0 text = original task sentence at every step.

Sample processing follows StarVLA's LIBERO pipeline (Libero4in1DataConfig + LeRobot loader @ 985c223):
  * every kept step t is a sample start; the 8-step chunk is padded at the episode end by
    repeating the last action ("first_last" padding)
  * arm dims 0..5: min_max normalisation to [-1, 1] with dataset min/max; gripper NOT normalised
  * images: PIL default resize 256 -> 224, no augmentation
  * no proprioceptive state
  * actions cast to float16 (StarVLA's state_action_dtype default)
Gripper convention (shared with our OFT loader): LIBERO -1 open/+1 close -> 1 open / 0 close.
"""

import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

CHUNK_LEN = 8
NUM_ARM_DIMS = 6  # x, y, z, roll, pitch, yaw are normalised; dim 6 (gripper) is not


# ──────────────────────────────────────────────────────────────────────
# Pure helpers
# ──────────────────────────────────────────────────────────────────────
def read_ids(path: str) -> List[Tuple[str, str, str]]:
    """Split file with columns suite, task, demo -> list of keys."""
    with open(path) as f:
        return [(r["suite"], r["task"], r["demo"]) for r in csv.DictReader(f)]


def read_a0_labels(path: str) -> Dict[Tuple[str, str, str], Tuple[str, bool]]:
    """switch_labels_v1.csv -> {(suite, task, demo): (A0 sentence, status ok)}."""
    with open(path) as f:
        return {(r["suite"], r["task"], r["demo"]): (r["A0"], r["status"] == "ok") for r in csv.DictReader(f)}


def remap_gripper(actions: np.ndarray) -> np.ndarray:
    """LIBERO -1 open / +1 close  ->  1 open / 0 close."""
    out = actions.astype(np.float32).copy()
    out[:, -1] = 1.0 - np.clip(out[:, -1], 0.0, 1.0)
    return out


def chunk_indices(t: int, length: int, chunk_len: int = CHUNK_LEN) -> np.ndarray:
    """Indices t..t+chunk_len-1, clipped to the last step ("first_last" end padding)."""
    return np.minimum(np.arange(t, t + chunk_len), length - 1)


def normalize_actions(a: np.ndarray, stats: dict) -> np.ndarray:
    """min_max on arm dims: 2 (x - min) / (max - min) - 1; dims with min == max -> 0. Gripper unchanged."""
    out = a.astype(np.float32).copy()
    lo = np.asarray(stats["min"], np.float32)[:NUM_ARM_DIMS]
    hi = np.asarray(stats["max"], np.float32)[:NUM_ARM_DIMS]
    arm = out[..., :NUM_ARM_DIMS]
    valid = hi != lo
    norm = np.zeros_like(arm)
    norm[..., valid] = 2.0 * (arm[..., valid] - lo[valid]) / (hi[valid] - lo[valid]) - 1.0
    out[..., :NUM_ARM_DIMS] = norm
    return out


def unnormalize_actions(a: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of normalize_actions on arm dims: (x + 1) / 2 (max - min) + min. Gripper unchanged."""
    out = a.astype(np.float32).copy()
    lo = np.asarray(stats["min"], np.float32)[:NUM_ARM_DIMS]
    hi = np.asarray(stats["max"], np.float32)[:NUM_ARM_DIMS]
    out[..., :NUM_ARM_DIMS] = (out[..., :NUM_ARM_DIMS] + 1.0) / 2.0 * (hi - lo) + lo
    return out


def compute_action_stats(demos: Sequence[dict]) -> dict:
    """Per-dim statistics of all kept (remapped) actions of the given demos."""
    acts = []
    for d in demos:
        with h5py.File(d["path"], "r") as f:
            acts.append(remap_gripper(f[d["key"][2]]["action"][()]))
    x = np.concatenate(acts)
    return dict(
        min=x.min(0).tolist(), max=x.max(0).tolist(), mean=x.mean(0).tolist(), std=x.std(0).tolist(),
        q01=np.quantile(x, 0.01, axis=0).tolist(), q99=np.quantile(x, 0.99, axis=0).tolist(),
        num_transitions=int(len(x)), num_trajectories=len(acts),
    )


def save_stats(stats: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_stats(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────
class LiberoLangDataset(Dataset):
    """Returns {image: [PIL main, PIL wrist], lang: str, action: float16 (8, 7) normalised}."""

    def __init__(
        self,
        data_root: str,
        id_files: Sequence[str],
        labels_csv: str,
        setup: str = "A0",
        stats: Optional[dict] = None,
        forbidden_id_files: Sequence[str] = (),
        image_size: Tuple[int, int] = (224, 224),
    ):
        if setup != "A0":
            raise ValueError(f"Only setup 'A0' is implemented, got {setup!r}")
        labels = read_a0_labels(labels_csv)

        # Guard: no training demo may come from a forbidden (test) task. Does not change the data.
        forbidden_tasks = {(s, t) for path in forbidden_id_files for s, t, _ in read_ids(path)}

        keys = [k for path in id_files for k in read_ids(path)]
        leaked = sorted({(s, t) for s, t, _ in keys} & forbidden_tasks)
        if leaked:
            raise RuntimeError(f"{len(leaked)} training tasks are forbidden (test) tasks, e.g. {leaked[:3]}")

        by_path = defaultdict(list)
        n_not_ok = 0
        for key in keys:
            text, ok = labels[key]
            if not ok:
                n_not_ok += 1
                continue
            by_path[f"{data_root}/{key[0]}/{key[1]}.hdf5"].append((key, text))

        self.demos, self.starts = [], []
        for path, items in by_path.items():
            with h5py.File(path, "r") as f:
                for key, text in items:
                    length = f[key[2]]["action"].shape[0]
                    i = len(self.demos)
                    self.demos.append(dict(key=key, path=path, lang=text, length=length))
                    self.starts += [(i, t) for t in range(length)]
        self.n_skipped_not_ok = n_not_ok
        self.stats = stats if stats is not None else compute_action_stats(self.demos)
        self.image_size = tuple(image_size)
        self._files: Dict[str, h5py.File] = {}  # opened lazily, once per worker process

    def __len__(self) -> int:
        return len(self.starts)

    def _group(self, d: dict):
        if d["path"] not in self._files:
            self._files[d["path"]] = h5py.File(d["path"], "r")
        return self._files[d["path"]][d["key"][2]]

    def __getitem__(self, i: int) -> dict:
        di, t = self.starts[i]
        d = self.demos[di]
        g = self._group(d)
        idx = chunk_indices(t, d["length"])
        raw = g["action"][idx[0]: idx[-1] + 1][idx - idx[0]]  # contiguous read, then end padding
        action = normalize_actions(remap_gripper(raw), self.stats).astype(np.float16)
        images = [Image.fromarray(g[k][t]).resize(self.image_size) for k in ("agentview", "wrist")]
        return {"image": images, "lang": d["lang"], "action": action}


def identity_collate(batch: List[dict]) -> List[dict]:
    return batch
