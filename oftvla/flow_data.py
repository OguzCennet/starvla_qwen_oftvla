"""Deterministic, resumable sample order for multi-GPU training of the flow heads.

The fork shuffles with its LeRobot loader (per-epoch reshuffle, train_starvla.py:337-350); our data is
the HDF5 E3 split (experiment-intrinsic, spec D9/D10), so the order cannot match the fork's anyway.
What we need instead:
  * the order depends only on (seed, number of samples): A0 / AS / MAS datasets index the SAME frames in
    the same order (only the text column differs), so runs of one head see identical frames in
    identical order;
  * a resume continues exactly where the run stopped;
  * independent of the GPU layout: every draw takes the next world*per_device samples of one global
    stream and rank r gets slice r.

Global stream: position k -> epoch k // N, permutation randperm(N, seed + epoch)[k % N].
One "draw" = one micro-batch on every rank. A training step with accumulation A uses A draws; the
periodic eval (as in the fork, which consumes a batch, train_starvla.py:386-388) uses one draw.
"""

from typing import Dict, Iterator, List

import torch
from torch.utils.data import Sampler


class GlobalStream:
    def __init__(self, num_samples: int, seed: int):
        self.n = int(num_samples)
        self.seed = int(seed)
        self._perms: Dict[int, torch.Tensor] = {}

    def _perm(self, epoch: int) -> torch.Tensor:
        if epoch not in self._perms:
            if len(self._perms) > 2:
                self._perms.pop(min(self._perms))
            g = torch.Generator().manual_seed(self.seed + epoch)
            self._perms[epoch] = torch.randperm(self.n, generator=g)
        return self._perms[epoch]

    def index(self, k: int) -> int:
        return int(self._perm(k // self.n)[k % self.n])

    def draw(self, d: int, world: int, per_device: int, rank: int) -> List[int]:
        """Dataset indices of rank ``rank`` in draw ``d``."""
        start = (d * world + rank) * per_device
        return [self.index(start + j) for j in range(per_device)]


class StreamBatchSampler(Sampler):
    """Endless batch sampler for one rank, starting at draw ``start_draw`` (use with DataLoader(batch_sampler=...))."""

    def __init__(self, num_samples: int, seed: int, world: int, rank: int, per_device: int, start_draw: int = 0):
        self.stream = GlobalStream(num_samples, seed)
        self.world, self.rank, self.per_device = world, rank, per_device
        self.start_draw = start_draw

    def __iter__(self) -> Iterator[List[int]]:
        d = self.start_draw
        while True:
            yield self.stream.draw(d, self.world, self.per_device, self.rank)
            d += 1
