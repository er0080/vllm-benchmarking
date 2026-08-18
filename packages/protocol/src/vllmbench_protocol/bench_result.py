"""The ``vllm bench serve --save-result`` contract, and the flattening layer over it.

This is the highest-consequence code in the repository. Everything downstream — every
chart, every comparison, every tuning decision — reads the columns this module produces.
A mistake here does not crash; it writes NULLs or plausible-but-wrong numbers, and the
error surfaces months later as a conclusion nobody can reproduce.

Two consequences shape the design.

**The field names are pinned to observed reality, not to documentation.** The published
docs describe fields like ``successful_requests``, ``benchmark_duration_sec`` and
``ttft_ms_p99``. vLLM 0.25.1 actually emits ``completed``, ``duration`` and
``p99_ttft_ms``. Flattening against the documented names would have written NULL for
every metric while reporting success. The names below were read off a real
``--save-result`` payload, and ``tests/fixtures/bench_serve_v0.25.1.json`` is that
payload. CI tier 2 re-derives them from a live vLLM so upstream drift fails a test rather
than corrupting a run.

**Nothing is silently dropped.** Any key not mapped here is preserved verbatim in
``extra``, and the whole payload is stored separately as ``Run.raw_result``. If this
mapping turns out to be wrong, the raw record is what allows recomputation instead of
re-running weeks of GPU time.
"""

from __future__ import annotations

from typing import Any

# vLLM's field name -> our column name.
#
# Only fields we promote to typed columns appear here. Everything else survives in
# `extra`. Adding a row is cheap; guessing at one is not — verify against a real payload.
SUMMARY_FIELD_MAP: dict[str, str] = {
    # Counts and duration
    "completed": "successful_requests",
    "failed": "failed_requests",
    "duration": "benchmark_duration_sec",
    "total_input_tokens": "total_input_tokens",
    "total_output_tokens": "total_generated_tokens",
    # Throughput
    "request_throughput": "request_throughput_req_sec",
    "output_throughput": "output_token_throughput_tok_sec",
    "total_token_throughput": "total_token_throughput_tok_sec",
    "max_output_tokens_per_s": "peak_output_token_throughput_tok_sec",
    "max_concurrent_requests": "peak_concurrent_requests",
    # Time to first token
    "mean_ttft_ms": "ttft_ms_mean",
    "median_ttft_ms": "ttft_ms_median",
    "p99_ttft_ms": "ttft_ms_p99",
    "std_ttft_ms": "ttft_ms_std",
    # Time per output token, excluding the first
    "mean_tpot_ms": "tpot_ms_mean",
    "median_tpot_ms": "tpot_ms_median",
    "p99_tpot_ms": "tpot_ms_p99",
    "std_tpot_ms": "tpot_ms_std",
    # Inter-token latency
    "mean_itl_ms": "itl_ms_mean",
    "median_itl_ms": "itl_ms_median",
    "p99_itl_ms": "itl_ms_p99",
    "std_itl_ms": "itl_ms_std",
}

# Fields that describe what was run rather than what was measured. Kept out of
# run_summary because the run row already records provenance, and duplicating it invites
# the two copies to disagree.
CONTEXT_FIELDS: frozenset[str] = frozenset(
    {
        "backend",
        "endpoint_type",
        "model_id",
        "tokenizer_id",
        "date",
        "label",
        "num_prompts",
        "request_rate",
        "max_concurrency",
        "burstiness",
    }
)

# Absence of these means the payload is not a benchmark result at all — a truncated file,
# a crashed run, or a future version that reorganized everything. Better to fail loudly
# than to write a row of NULLs that looks like a completed benchmark.
REQUIRED_FIELDS: frozenset[str] = frozenset(
    {"completed", "duration", "total_input_tokens", "total_output_tokens"}
)


class BenchResultError(ValueError):
    """The payload is not a usable ``vllm bench serve`` result."""


class EmptyBenchResult(BenchResultError):
    """The benchmark ran to completion and measured nothing.

    Separated from its parent because the two mean different things and have different
    fixes. A schema mismatch means vLLM changed its output and the work is in this
    repository; this means the benchmark itself failed — a model name that is not served,
    an engine that fell over mid-run — and the work is on the host.
    """


def flatten_bench_result(payload: dict[str, Any], *, gpu_count: int = 1) -> dict[str, Any]:
    """Map a raw ``--save-result`` payload onto ``run_summary`` columns.

    ``gpu_count`` drives the per-device normalization required by invariant 8. It is
    computed here, once, rather than at read time by each consumer — otherwise the UI,
    the MCP tools and the CSV export each divide by their own idea of the device count,
    and they will eventually disagree.

    Raises ``BenchResultError`` when required fields are missing. Silence would be worse:
    a summary row full of NULLs is indistinguishable from a benchmark that legitimately
    measured nothing.
    """
    if not isinstance(payload, dict):
        raise BenchResultError(f"expected a JSON object, got {type(payload).__name__}")

    missing = REQUIRED_FIELDS - payload.keys()
    if missing:
        raise BenchResultError(
            "benchmark result is missing required fields: "
            + ", ".join(sorted(missing))
            + ". This usually means the vLLM version changed its --save-result schema; "
            "check SUMMARY_FIELD_MAP against a real payload."
        )

    if gpu_count < 1:
        raise BenchResultError(f"gpu_count must be at least 1, got {gpu_count}")

    _reject_empty(payload)

    flattened: dict[str, Any] = {}
    for source, column in SUMMARY_FIELD_MAP.items():
        value = payload.get(source)
        # vLLM emits null for metrics it did not compute (request_goodput without an SLO,
        # for instance). Null is the honest answer and is preserved as such.
        flattened[column] = value

    # Invariant 8. Aggregate throughput is not comparable across parallelism topologies:
    # a TP=4 run trivially beats TP=1 on these numbers while being worse per device.
    for aggregate, per_gpu in (
        ("output_token_throughput_tok_sec", "output_token_throughput_per_gpu"),
        ("total_token_throughput_tok_sec", "total_token_throughput_per_gpu"),
    ):
        value = flattened.get(aggregate)
        flattened[per_gpu] = None if value is None else value / gpu_count

    known = set(SUMMARY_FIELD_MAP) | CONTEXT_FIELDS
    flattened["extra"] = {k: v for k, v in payload.items() if k not in known}

    return flattened


def _reject_empty(payload: dict[str, Any]) -> None:
    """Refuse a benchmark in which nothing completed.

    This is not defensive programming; it is a real payload, captured from vLLM 0.25.1
    and checked in as ``bench_serve_all_requests_failed_v0.25.1.json``. Point
    ``vllm bench serve`` at a model name the server does not serve and every request
    404s — and the client **exits 0**, writes a result file, and reports:

        completed: 0, failed: 8, mean_ttft_ms: 0.0, output_throughput: 0.0

    Every field the contract requires is present, so nothing upstream of here notices.
    Without this check the run is recorded as succeeded with a full summary row of zeros,
    and those zeros are not neutral: on a latency axis 0 ms is the *best* value there is,
    so a benchmark that measured nothing renders as the fastest configuration ever
    tested. It would sit on the Pareto frontier.

    Zero completions is therefore not a measurement of zero. It is the absence of a
    measurement, and the only honest thing to record is a failure.

    A run with *some* completions is kept: those requests really did complete and the
    figures describe them. `failed_requests` is flattened alongside, and the analysis
    layer warns when it is non-zero, because throughput divided by the whole duration
    understates a partially failed run rather than invalidating it.
    """
    completed = payload.get("completed")
    if not isinstance(completed, int) or completed > 0:
        return

    failed = payload.get("failed")
    detail = f"{failed} request(s) failed" if failed else "no requests were issued"
    raise EmptyBenchResult(
        f"the benchmark completed 0 requests ({detail}), so its metrics are zeros rather "
        "than measurements — on a latency axis that would read as the fastest result "
        "ever recorded. Usually the served model name does not match what the engine is "
        "answering to, or the engine failed during the run; check the benchmark's output."
    )
