"""Is vLLM's custom all-reduce numerically correct on this driver?

Gate 2 found the serving path emitting token 0 repeatedly — the signature of corrupt
logits — with custom all-reduce enabled, and coherent text with it disabled, on the same
driver in the same session. This isolates the kernel from the rest of the stack: an
all-reduce of ones across two ranks must produce exactly 2.0, elementwise, at every size.

Launch with:  torchrun --nproc_per_node=2 customar_correctness.py
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

SIZES_BYTES = [8 * 1024, 32 * 1024, 128 * 1024, 512 * 1024, 2 * 1024 * 1024]


def check(tag: str, got: torch.Tensor, expect: float) -> dict:
    finite = bool(torch.isfinite(got).all().item())
    exact = bool((got == expect).all().item())
    return {
        "path": tag,
        "all_finite": finite,
        "all_exact": exact,
        "min": float(got.min().item()) if finite else None,
        "max": float(got.max().item()) if finite else None,
        "n_nan": int(torch.isnan(got).sum().item()),
        "n_inf": int(torch.isinf(got).sum().item()),
        "first8": [float(x) for x in got[:8].tolist()],
    }


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:" + str(local_rank))
    dist.init_process_group(backend="nccl")
    cpu_group = dist.new_group(backend="gloo")

    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    car = CustomAllreduce(group=cpu_group, device=device)
    world = dist.get_world_size()

    results = []
    for nbytes in SIZES_BYTES:
        n = nbytes // 2

        # NCCL, as the reference. In-place, so use a fresh tensor.
        ref = torch.ones(n, dtype=torch.bfloat16, device=device)
        dist.all_reduce(ref)
        ref_row = check("nccl", ref, float(world))

        car_row = {"path": "custom_ar", "skipped": True}
        if not car.disabled:
            src = torch.ones(n, dtype=torch.bfloat16, device=device)
            if car.should_custom_ar(src):
                out = car.all_reduce(src)
                torch.cuda.synchronize(device)
                car_row = check("custom_ar", out, float(world))

        results.append({"bytes": nbytes, "nccl": ref_row, "custom_ar": car_row})

    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": world,
                    "peer_access": torch.cuda.can_device_access_peer(0, 1),
                    "custom_ar_disabled": car.disabled,
                    "expected_value": float(world),
                    "results": results,
                },
                indent=2,
            )
        )

    car.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
