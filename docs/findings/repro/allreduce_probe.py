"""NCCL all-reduce latency across the message sizes tensor-parallel decode actually uses.

This is the metric the P2P patch is supposed to move. A TP=2 decode step issues one
all-reduce per layer, per token, and at batch 1 those are small enough that latency
dominates and bandwidth is irrelevant. Sweeping the range makes it visible where —
if anywhere — the interconnect stops being the bottleneck.

Launch with:  torchrun --nproc_per_node=2 allreduce_probe.py
"""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist

# 8 KiB is roughly one layer of a 27B model at batch 1 (hidden 5120, bf16);
# the top of the range covers large-batch prefill.
SIZES_BYTES = [8 << 10, 32 << 10, 128 << 10, 512 << 10, 2 << 20, 8 << 20]
ITERS = 500
WARMUP = 50


def bench(nbytes: int, device: torch.device) -> float:
    """Mean microseconds per all-reduce."""
    tensor = torch.ones(nbytes // 2, dtype=torch.bfloat16, device=device)

    for _ in range(WARMUP):
        dist.all_reduce(tensor)
    torch.cuda.synchronize(device)
    dist.barrier()

    start = time.perf_counter()
    for _ in range(ITERS):
        dist.all_reduce(tensor)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    return elapsed / ITERS * 1e6


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    results = {}
    for nbytes in SIZES_BYTES:
        micros = bench(nbytes, device)
        results[nbytes] = round(micros, 2)

    if rank == 0:
        # Bus bandwidth for a ring all-reduce moves 2(n-1)/n of the buffer.
        world = dist.get_world_size()
        factor = 2 * (world - 1) / world
        print(json.dumps({
            "world_size": world,
            "peer_access_0_1": torch.cuda.can_device_access_peer(0, 1),
            "allreduce": [
                {
                    "bytes": b,
                    "latency_us": us,
                    "bus_gb_s": round(b * factor / (us * 1e-6) / 1e9, 2),
                }
                for b, us in results.items()
            ],
        }, indent=2))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
