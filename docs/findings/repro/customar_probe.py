"""vLLM's custom all-reduce against NCCL, at the sizes tensor-parallel decode uses.

The driver patch does not speed NCCL up in the small-message regime — NCCL's floor there is
launch and synchronisation overhead rather than data movement, and at this topology it
declines the P2P path entirely unless told otherwise. What the patch actually unlocks is
vLLM's own one-shot kernel, which is gated on `cudaDeviceCanAccessPeer` and is built for
exactly this regime. Measuring it directly predicts the serving experiment before it is
paid for.

Launch with:  torchrun --nproc_per_node=2 customar_probe.py
"""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist

# 8 KiB is roughly one layer of a 27B model at batch 1 (hidden 5120, bf16); the rest walk
# up through the batch sizes a concurrency sweep will actually reach.
SIZES_BYTES = [8 * 1024, 32 * 1024, 128 * 1024, 512 * 1024, 2 * 1024 * 1024]
ITERS = 500
WARMUP = 50


def timed(fn, device) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()
    start = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / ITERS * 1e6


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:" + str(local_rank))
    dist.init_process_group(backend="nccl")

    # CustomAllreduce gathers physical device ids over CPU tensors, so it needs a group
    # that can carry them.
    cpu_group = dist.new_group(backend="gloo")

    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    car = CustomAllreduce(group=cpu_group, device=device)

    results = []
    for nbytes in SIZES_BYTES:
        t = torch.ones(nbytes // 2, dtype=torch.bfloat16, device=device)
        nccl_us = timed(lambda t=t: dist.all_reduce(t), device)
        row = {"bytes": nbytes, "nccl_us": round(nccl_us, 2)}
        if not car.disabled and car.should_custom_ar(t):
            car_us = timed(lambda t=t: car.all_reduce(t), device)
            row["custom_ar_us"] = round(car_us, 2)
            row["speedup_x"] = round(nccl_us / car_us, 2)
        else:
            row["custom_ar_us"] = None
            row["speedup_x"] = None
        results.append(row)

    if rank == 0:
        print(
            json.dumps(
                {
                    "peer_access": torch.cuda.can_device_access_peer(0, 1),
                    "custom_ar_disabled": car.disabled,
                    "results": results,
                },
                indent=2,
            )
        )

    car.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
