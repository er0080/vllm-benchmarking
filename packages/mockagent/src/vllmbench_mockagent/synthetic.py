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
    served_model_name: str | None = None,
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

    ``model_id`` and ``tokenizer_id`` both carry the *weights* identifier, matching what
    upstream writes for ``--model``. Keeping that faithful is the point: while the mock
    echoed whatever single name it was handed, an alias arriving in ``--model`` looked
    perfectly correct here and only failed against a real engine.
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
        "served_model_name": served_model_name or model,
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


# Telemetry shape. A benchmark does not start at steady state: requests arrive, the
# queue builds, the KV cache fills, and everything plateaus. Modelling that ramp is the
# point — a flat line would let a timeline chart look finished while never having shown
# a transition, which is the only part of the picture that is hard to render well.
TELEMETRY_RAMP_FRACTION = 0.25


def synthesize_telemetry(
    *,
    duration_seconds: float,
    interval_seconds: float,
    device_indices: list[int],
    max_concurrency: int | None,
    config_hash: str,
    tensor_parallel_size: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Produce engine and per-device samples spanning one benchmark window.

    Per device rather than per host, and deliberately *asymmetric* across devices: on a
    real tensor-parallel run device 0 carries the sampling and detokenization work and
    runs a little hotter. A mock that emitted identical series for every GPU would make
    a per-device chart look correct while it was in fact plotting one line on top of
    another — which is the one bug per-device sampling exists to expose.
    """
    concurrency = float(max_concurrency or 64)
    seed = f"{config_hash}:{concurrency}"
    steady_running = min(concurrency, SATURATION_CONCURRENCY * tensor_parallel_size**0.8)
    steady_waiting = max(0.0, concurrency - steady_running)

    engine: list[dict[str, Any]] = []
    gpu: list[dict[str, Any]] = []

    ticks = max(1, int(duration_seconds / interval_seconds))
    ramp_ticks = max(1.0, ticks * TELEMETRY_RAMP_FRACTION)
    queries = 0
    hits = 0

    for tick in range(ticks):
        offset = tick * interval_seconds
        # tanh ramp to steady state, so the first samples show the engine filling up.
        progress = math.tanh(tick / ramp_ticks)

        running = steady_running * progress
        waiting = steady_waiting * progress
        kv = min(0.97, 0.85 * progress * _jitter(f"{seed}:kv:{tick}", 0.03))

        # Counters only ever increase — differencing them is the whole reason they are
        # stored raw, and a counter that went backwards would produce negative rates.
        queries += int(64 * progress)
        hits += int(18 * progress)

        engine.append(
            {
                "offset_seconds": offset,
                "num_requests_running": int(running),
                "num_requests_waiting": int(waiting),
                "kv_cache_usage_fraction": kv,
                "num_preemptions_total": int(max(0.0, (progress - 0.9) * 20)),
                "prefix_cache_queries_total": queries,
                "prefix_cache_hits_total": hits,
            }
        )

        for position, index in enumerate(device_indices or [0]):
            # Device 0 runs hotter; the rest track it a few points lower.
            lead = 1.0 if position == 0 else 0.93
            util = min(99.0, 96.0 * progress * lead * _jitter(f"{seed}:sm:{index}:{tick}", 0.02))
            gpu.append(
                {
                    "offset_seconds": offset,
                    "gpu_index": index,
                    "sm_utilization_pct": util,
                    "memory_used_bytes": int((6.0 + 14.0 * progress * lead) * 1024**3),
                    "power_watts": 90.0 + 230.0 * progress * lead,
                    "temperature_c": 38.0 + 34.0 * progress * lead,
                    "sm_clock_mhz": int(1200 + 600 * progress),
                    "memory_clock_mhz": 9501,
                }
            )

    return engine, gpu
