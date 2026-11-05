"""Multi-node NCCL bandwidth preflight (launched with torchrun on every node).

Times all_reduce and reduce_scatter (ZeRO-2's gradient collective) of 1 GiB bf16 tensors and prints the
bus bandwidth on rank 0, so the transport (InfiniBand vs TCP sockets) can be compared before training.
"""
import os
import time

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
n = 512 * 1024 * 1024  # 1 GiB in bf16
x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
out = torch.empty(n // world, dtype=torch.bfloat16, device="cuda")
for name, fn, factor in (("all_reduce", lambda: dist.all_reduce(x), 2 * (world - 1) / world),
                         ("reduce_scatter", lambda: dist.reduce_scatter_tensor(out, x), (world - 1) / world)):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 5
    if rank == 0:
        print(f"[nccl] {name}: {dt * 1e3:.1f} ms per 1 GiB, bus bandwidth {2 * n * factor / dt / 1e9:.1f} GB/s (world {world})", flush=True)
dist.destroy_process_group()
