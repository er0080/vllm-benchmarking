"""Peer-access facts and raw device-to-device transfer performance.

Run identically on the stock and patched drivers. Nothing here imports vLLM: the
point is to establish whether P2P works *at all* before any benchmark is spent, so
that a null result from the serving benchmark can be told apart from a treatment
that never took.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import torch


def link_state() -> list[dict[str, str]]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pcie.link.gen.current,pcie.link.width.current",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    rows = []
    for line in out.splitlines():
        idx, gen, width = (p.strip() for p in line.split(","))
        rows.append({"gpu": idx, "gen": gen, "width": width})
    return rows


def copy_bandwidth(src: int, dst: int, mib: int = 256, iters: int = 20) -> float:
    """Unidirectional GB/s for a large contiguous device-to-device copy."""
    a = torch.empty(mib * 1024 * 1024, dtype=torch.uint8, device=f"cuda:{src}")
    b = torch.empty_like(a, device=f"cuda:{dst}")
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)

    for _ in range(3):  # warm up
        b.copy_(a)
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)

    start = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)
    elapsed = time.perf_counter() - start

    total_bytes = a.numel() * iters
    return total_bytes / elapsed / 1e9


def copy_latency(src: int, dst: int, nbytes: int = 4096, iters: int = 2000) -> float:
    """Round-trip microseconds for a small copy — the latency-bound regime."""
    a = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{src}")
    b = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{dst}")
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)

    for _ in range(50):
        b.copy_(a)
        a.copy_(b)
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)

    start = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
        a.copy_(b)
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)
    elapsed = time.perf_counter() - start

    return elapsed / iters * 1e6


def main() -> int:
    n = torch.cuda.device_count()
    report: dict = {
        "torch": torch.__version__,
        "device_count": n,
        "devices": [torch.cuda.get_device_name(i) for i in range(n)],
        "driver_reports_peer_access": {},
        "link_state_idle": link_state(),
    }
    if n < 2:
        print(json.dumps(report, indent=2))
        return 1

    for i in range(n):
        for j in range(n):
            if i != j:
                report["driver_reports_peer_access"][f"{i}->{j}"] = (
                    torch.cuda.can_device_access_peer(i, j)
                )

    report["copy_bandwidth_gb_s_0_to_1"] = round(copy_bandwidth(0, 1), 2)
    report["link_state_under_load"] = link_state()
    report["copy_bandwidth_gb_s_1_to_0"] = round(copy_bandwidth(1, 0), 2)
    report["roundtrip_latency_us_4kb"] = round(copy_latency(0, 1), 2)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
