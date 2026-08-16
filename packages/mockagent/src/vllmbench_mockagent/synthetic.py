"""Plausible benchmark numbers, derived rather than random.

The mock exists so the control plane and the charts can be built without a GPU. Numbers
that are merely random would make every chart look like noise, and a UI tuned against
noise is a UI that has never been seen working. So these follow the shapes real serving
actually produces:

- throughput rises with concurrency, then flattens as the engine saturates
- TTFT is flat while there is headroom, then climbs sharply once requests queue
- inter-token latency degrades gradually with load

They are still fabricated, and every run carrying them is flagged synthetic at creation
(invariant 7). The point is to be *shaped* like real data, never to be mistaken for it.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

# A notional engine: how many concurrent requests it serves before saturating, and the
# ceiling it saturates at. Chosen to sit in the range a small model on one datacentre
# GPU actually reaches, so charts have a realistic knee.
SATURATION_CONCURRENCY = 32.0
PEAK_OUTPUT_TOKENS_PER_SEC = 2400.0
BASE_TTFT_MS = 42.0
BASE_ITL_MS = 9.0


def _jitter(seed_material: str, spread: float) -> float:
    """Deterministic pseudo-jitter in [1 - spread, 1 + spread].

    Deterministic so tests are stable and so re-running a sweep point in the mock
    reproduces its result — but varied across configs, so replicate spread is visible
    in the UI rather than every bar being identical.
    """
    digest = hashlib.sha256(seed_material.encode()).digest()
    unit = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return 1.0 + (unit * 2 - 1) * spread


def synthesize_bench_result(
    *,
    model: str,
    num_prompts: int,
    max_concurrency: int | None,
    request_rate: float | None,
    input_len: int,
    output_len: int,
    config_hash: str,
    replicate_seed: str = "",
    tensor_parallel_size: int = 1,
) -> dict[str, Any]:
    """Produce a payload with the same field names real vLLM emits.

    The field names matter as much as the values: code paths that read these are the
    same ones that will read production results, so a mock using different names would
    let a flattening bug through.
    """
    concurrency = float(max_concurrency or 64)
    seed = f"{config_hash}:{replicate_seed}:{concurrency}:{output_len}"

    # Throughput saturates rather than growing without bound. tanh gives a smooth knee
    # around SATURATION_CONCURRENCY, and TP widens the ceiling sublinearly — which is
    # the entire question a TP sweep is asked to answer.
    tp_scaling = tensor_parallel_size**0.8
    utilization = math.tanh(concurrency / SATURATION_CONCURRENCY)
    output_throughput = PEAK_OUTPUT_TOKENS_PER_SEC * tp_scaling * utilization
    output_throughput *= _jitter(seed + ":throughput", 0.04)

    # Queueing: below saturation TTFT is nearly flat; above it, waiting dominates.
    queue_pressure = max(0.0, concurrency / SATURATION_CONCURRENCY - 1.0)
    ttft_mean = BASE_TTFT_MS * (1.0 + 2.5 * queue_pressure) * _jitter(seed + ":ttft", 0.08)
    # p99 pulls away from the mean under load — the spread is the interesting part.
    ttft_p99 = ttft_mean * (2.2 + 3.0 * queue_pressure)

    itl_mean = BASE_ITL_MS * (1.0 + 0.35 * utilization) * _jitter(seed + ":itl", 0.05)

    total_output_tokens = num_prompts * output_len
    total_input_tokens = num_prompts * input_len
    duration = total_output_tokens / max(output_throughput, 1.0)

    return {
        "backend": "vllm",
        "endpoint_type": "vllm",
        "model_id": model,
        "tokenizer_id": model,
        "num_prompts": num_prompts,
        "request_rate": "inf" if request_rate is None else request_rate,
        "max_concurrency": max_concurrency,
        "burstiness": 1.0,
        "completed": num_prompts,
        "failed": 0,
        "duration": duration,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "request_throughput": num_prompts / max(duration, 0.001),
        "output_throughput": output_throughput,
        "total_token_throughput": (total_input_tokens + total_output_tokens) / max(duration, 0.001),
        "max_output_tokens_per_s": output_throughput * 1.08,
        "max_concurrent_requests": concurrency,
        "mean_ttft_ms": ttft_mean,
        "median_ttft_ms": ttft_mean * 0.72,
        "p99_ttft_ms": ttft_p99,
        "std_ttft_ms": ttft_mean * 0.6,
        "mean_tpot_ms": itl_mean,
        "median_tpot_ms": itl_mean * 0.97,
        "p99_tpot_ms": itl_mean * 1.9,
        "std_tpot_ms": itl_mean * 0.18,
        "mean_itl_ms": itl_mean,
        "median_itl_ms": itl_mean * 0.93,
        "p99_itl_ms": itl_mean * 2.6,
        "std_itl_ms": itl_mean * 0.31,
        "request_goodput": None,
        "rtfx": 0.0,
        "label": None,
        "date": "synthetic",
    }
