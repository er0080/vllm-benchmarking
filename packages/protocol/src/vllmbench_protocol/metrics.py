"""The vLLM ``/metrics`` contract.

Names verified against a live vLLM 0.25.1 rather than taken from documentation, for the
same reason as ``bench_result``: two of them are not what a reasonable reading of the
docs would suggest.

The one that matters most: **there is no prefix-cache hit-rate gauge.** vLLM exposes two
counters, ``vllm:prefix_cache_queries_total`` and ``vllm:prefix_cache_hits_total``. A
schema that stored only a scraped "hit rate" would have recorded nothing at all. Storing
the counters and deriving the rate also follows "raw before derived": counters are
monotonic and can be differenced across any window, whereas a rate sampled at one instant
cannot be re-derived once written.
"""

from __future__ import annotations

# Gauges: instantaneous state, meaningful at the moment sampled.
GAUGE_METRICS: dict[str, str] = {
    "vllm:num_requests_running": "num_requests_running",
    "vllm:num_requests_waiting": "num_requests_waiting",
    # Not "gpu_cache_usage_perc", which is what the older docs imply.
    "vllm:kv_cache_usage_perc": "kv_cache_usage_pct",
}

# Counters: monotonic totals. Stored raw and differenced later, never converted to a
# rate at write time — a rate discards the information needed to recompute it.
COUNTER_METRICS: dict[str, str] = {
    "vllm:num_preemptions_total": "num_preemptions_total",
    "vllm:prefix_cache_queries_total": "prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total": "prefix_cache_hits_total",
}

ALL_METRICS: dict[str, str] = {**GAUGE_METRICS, **COUNTER_METRICS}


def prefix_cache_hit_rate(queries: float | None, hits: float | None) -> float | None:
    """Derive the hit rate from the two counters.

    Returns None rather than 0.0 when no queries have been made. Zero would claim a
    measured 0% hit rate, which is a different statement from "nothing was asked yet" —
    and averaging the two together across a sweep would quietly drag the number down.
    """
    if queries is None or hits is None or queries <= 0:
        return None
    return hits / queries
