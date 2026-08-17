"""The vLLM ``/metrics`` contract.

Names and *values* verified against a live vLLM 0.25.1 rather than taken from
documentation, for the same reason as ``bench_result``. The captured payloads are in
``tests/fixtures/metrics_vllm_0_25_1_{idle,loaded}.txt``; the loaded one was taken with
128 concurrent requests in flight, because a payload full of zeros cannot show what a
value looks like when it is real.

Three things the real payload settles, none of which the docs would have:

**There is no prefix-cache hit-rate gauge.** vLLM exposes two counters,
``vllm:prefix_cache_queries_total`` and ``vllm:prefix_cache_hits_total``. A schema
storing only a scraped "hit rate" would have recorded nothing at all. Storing counters
and deriving the rate also follows "raw before derived": counters are monotonic and can
be differenced across any window, whereas a rate sampled at an instant cannot be
recovered.

**``kv_cache_usage_perc`` is a fraction, not a percent.** Under load it reads
``0.11223878211531546``, meaning 11.2%. The name says otherwise. It is stored exactly as
emitted and converted only for display — hence ``kv_cache_usage_fraction`` rather than
inheriting vLLM's misleading suffix into our own schema.

**Metric names collide on prefixes.** The payload contains both::

    vllm:num_requests_waiting{engine="0",model_name="..."} 64.0
    vllm:num_requests_waiting_by_reason{...,reason="capacity"} 0.0
    vllm:num_requests_waiting_by_reason{...,reason="deferred"} 0.0

A ``startswith`` match takes all three, and since ``_by_reason`` comes *later* in the
file, last-write-wins records a queue depth of 0 while 64 requests are actually queued.
Counters bring the same hazard via Prometheus's companion ``_created`` series. So names
are matched exactly, terminated by ``{`` or whitespace, and never by prefix.
"""

from __future__ import annotations

import re

# Gauges: instantaneous state, meaningful at the moment sampled.
GAUGE_METRICS: dict[str, str] = {
    "vllm:num_requests_running": "num_requests_running",
    "vllm:num_requests_waiting": "num_requests_waiting",
    # Not "gpu_cache_usage_perc", which is what the older docs imply. And despite the
    # "perc" suffix it is a 0..1 fraction — see the module docstring.
    "vllm:kv_cache_usage_perc": "kv_cache_usage_fraction",
}

# Counters: monotonic totals. Stored raw and differenced later, never converted to a
# rate at write time — a rate discards the information needed to recompute it.
COUNTER_METRICS: dict[str, str] = {
    "vllm:num_preemptions_total": "num_preemptions_total",
    "vllm:prefix_cache_queries_total": "prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total": "prefix_cache_hits_total",
}

ALL_METRICS: dict[str, str] = {**GAUGE_METRICS, **COUNTER_METRICS}


# How to combine a metric that appears more than once. vLLM labels every series with
# `engine="N"`, so a data-parallel deployment emits one series per engine and the right
# combination is not the same for all of them: request counts add up, a cache-occupancy
# fraction does not. Summing fractions would happily report 180% KV utilization.
MEAN_METRICS: frozenset[str] = frozenset({"vllm:kv_cache_usage_perc"})

# A sample line: `name{label="v",...} 12.5` or `name 12.5`. The name is anchored and
# terminated by `{` or whitespace, which is what stops `num_requests_waiting` from
# swallowing `num_requests_waiting_by_reason`.
_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?[ \t]+(?P<value>.+)$"
)


def parse_metrics(text: str) -> dict[str, float]:
    """Parse a Prometheus exposition payload into our column names.

    Only the metrics in :data:`ALL_METRICS` are extracted; the real payload carries 86
    metric families and we store six. Unknown names are ignored rather than rejected,
    because vLLM adding a metric is not an error — but a name we *do* want changing
    shape is, and that is what the tier 2 contract test exists to catch.
    """
    totals: dict[str, list[float]] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None:
            continue
        name = match.group("name")
        if name not in ALL_METRICS:
            continue
        try:
            value = float(match.group("value").split()[0])
        except (ValueError, IndexError):
            # A malformed value is skipped rather than recorded as zero: a zero here is
            # indistinguishable from a genuinely idle engine.
            continue
        # NaN is Prometheus's way of saying "no value", and it would poison any average.
        if value != value:
            continue
        totals.setdefault(name, []).append(value)

    result: dict[str, float] = {}
    for name, values in totals.items():
        if not values:
            continue
        combined = sum(values) / len(values) if name in MEAN_METRICS else sum(values)
        result[ALL_METRICS[name]] = combined
    return result


def prefix_cache_hit_rate(queries: float | None, hits: float | None) -> float | None:
    """Derive the hit rate from the two counters.

    Returns None rather than 0.0 when no queries have been made. Zero would claim a
    measured 0% hit rate, which is a different statement from "nothing was asked yet" —
    and averaging the two together across a sweep would quietly drag the number down.
    """
    if queries is None or hits is None or queries <= 0:
        return None
    return hits / queries
