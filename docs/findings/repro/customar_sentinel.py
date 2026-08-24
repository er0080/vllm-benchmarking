"""Does the custom all-reduce compute wrong values, or write nothing at all?

The correctness check reads a buffer that `all_reduce` allocates with `torch.empty_like`,
so uniform NaN has two very different explanations: the kernel computed NaN, or the kernel
never wrote and we are reading uninitialised memory. Those are different bugs and only one
of them is a numerical fault.

Pre-filling the output with a sentinel separates them. All sentinel back means the kernel
did nothing; NaN back means it wrote NaN.

Launch with:  torchrun --nproc_per_node=2 customar_sentinel.py
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

SENTINEL = -7.0
SIZES_BYTES = [8 * 1024, 128 * 1024]


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:" + str(local_rank))
    dist.init_process_group(backend="nccl")
    cpu_group = dist.new_group(backend="gloo")

    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    car = CustomAllreduce(group=cpu_group, device=device)
    rows = []

    for nbytes in SIZES_BYTES:
        n = nbytes // 2
        src = torch.ones(n, dtype=torch.bfloat16, device=device)
        out = torch.full((n,), SENTINEL, dtype=torch.bfloat16, device=device)

        if car.disabled or not car.should_custom_ar(src):
            rows.append({"bytes": nbytes, "skipped": True})
            continue

        car.all_reduce(src, out=out, registered=False)
        torch.cuda.synchronize(device)

        n_sentinel = int((out == SENTINEL).sum().item())
        n_nan = int(torch.isnan(out).sum().item())
        n_correct = int((out == 2.0).sum().item())
        rows.append(
            {
                "bytes": nbytes,
                "elements": n,
                "still_sentinel": n_sentinel,
                "nan": n_nan,
                "correct_2.0": n_correct,
                "verdict": (
                    "kernel wrote nothing" if n_sentinel == n
                    else "kernel wrote NaN" if n_nan == n
                    else "correct" if n_correct == n
                    else "mixed"
                ),
                "first8": [float(x) for x in out[:8].tolist()],
            }
        )

        # And confirm the input survived — a kernel scribbling on its source would be a
        # third failure mode again.
        rows[-1]["input_intact"] = bool((src == 1.0).all().item())

    if rank == 0:
        print(json.dumps({"sentinel": SENTINEL, "results": rows}, indent=2))

    car.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
