"""The sweep report: one sweep written out, for a reader who will act on it.

Shared by the MCP resource and the HTTP download, so there is one report rather than two
that drift. The same reason the MCP tools call the router functions: a second
implementation is a second thing to be wrong.

Markdown rather than JSON because the things worth saying about a sweep — that these two
groups may not be compared, that a difference sits inside the spread — are sentences, not
fields. The JSON is still there in the analysis endpoints for anything that wants to
compute.

Partitioned exactly as the charts are: a section per host / GPU / vLLM version /
bench-client location. A reader who skips the headings still cannot find two GPU models in
one table, because they are not in one table.
"""

from __future__ import annotations

from vllmbench_api.analysis import PARETO_Y
from vllmbench_api.schemas import AnalysisOut, DurationEstimateOut, PointOut, SweepOut

#: What the report tabulates. Deliberately per-GPU first: an agent skimming a table reads
#: the leftmost number, and the aggregate is the one that is not comparable across
#: tensor-parallel widths (invariant 8).
REPORT_METRICS = (
    ("total_token_throughput_per_gpu", "tok/s per GPU"),
    (PARETO_Y, "per-user tok/s"),
    ("ttft_ms_p99", "TTFT p99 ms"),
    ("total_token_throughput_tok_sec", "tok/s aggregate"),
)


def _cell(point: PointOut, key: str) -> str:
    """One metric, with its spread, or an em dash.

    The spread travels with the number rather than sitting in a footnote, because the
    number alone invites a comparison the spread would have forbidden.
    """
    spread = point.metrics.get(key)
    if spread is None:
        return "—"
    if spread.n < 2:
        return f"{spread.median:,.1f}"
    return f"{spread.median:,.1f} ±{(spread.max - spread.min) / 2:,.1f}"


def _duration(seconds: float) -> str:
    """Seconds as something a reader can act on.

    Rounded to the minute above an hour, because an estimate extrapolated from a handful
    of runs does not have second-level precision and printing it as though it does invites
    more trust than it has earned.
    """
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.1f} h"


def _render_estimate(estimate: DurationEstimateOut) -> list[str]:
    """How much longer, with the arithmetic visible.

    The two components are shown separately because they behave differently: benchmark
    time scales with the runs left, engine-load time with the *config changes* left. A
    reader deciding whether to wait needs to know which of those is the bulk of it.
    """
    remaining = (
        f"{estimate.runs_remaining} run(s) left, {estimate.engine_loads_remaining} of them "
        "restarting the engine."
    )
    if estimate.seconds_remaining is None:
        lines = [f"**Remaining:** {remaining} No time estimate yet.", ""]
    else:
        lines = [
            f"**Remaining:** ~{_duration(estimate.seconds_remaining)} — {remaining}",
            "",
        ]
        if estimate.median_run_seconds is not None:
            detail = f"Based on {estimate.sample_size} completed run(s): "
            detail += f"{_duration(estimate.median_run_seconds)} per benchmark"
            if estimate.median_engine_load_seconds is not None:
                detail += f", {_duration(estimate.median_engine_load_seconds)} per engine load"
            lines += [detail + ".", ""]
    for caveat in estimate.caveats:
        lines += [f"> {caveat}", ""]
    return lines


def render_sweep_report(sweep: SweepOut, analysis: AnalysisOut) -> str:
    """A sweep as prose and tables.

    Written for a reader that will act on it. That means the caveats are inline rather
    than appended: the synthetic banner is the first thing on the page, each comparability
    group is its own section with its own heading, and a group's warnings sit above its
    table rather than below it.
    """
    out: list[str] = [f"# {sweep.name}", ""]

    if sweep.is_synthetic:
        out += [
            "> **Synthetic. These are not measurements of any real hardware.** They come "
            "from the mock agent or the CPU backend and exist to exercise the framework. "
            "Do not compare them with real results or draw a tuning conclusion from them.",
            "",
        ]

    if sweep.description:
        out += [sweep.description, ""]

    progress = sweep.progress
    out += [
        f"**Status:** {sweep.status} — {progress.succeeded} succeeded, "
        f"{progress.failed} failed, {progress.cancelled} cancelled, of {progress.total}.",
        "",
        f"**Plan:** {sweep.replicates} replicate(s), {sweep.replicate_order} order, "
        f"{sweep.engine_starts} engine load(s). Started by `{sweep.initiated_by}`.",
        "",
    ]
    if sweep.error:
        out += [f"**Error:** {sweep.error}", ""]

    if progress.failures:
        # The one line that turns "11 failed" into something to act on. Eleven runs of
        # one cause and eleven runs of eleven causes are different situations, and the
        # count alone cannot tell them apart.
        out += [
            "**Failures:** "
            + ", ".join(f"{count} {kind}" for kind, count in progress.failures.items())
            + ".",
            "",
        ]

    if sweep.estimated_remaining is not None:
        out += _render_estimate(sweep.estimated_remaining)

    if not analysis.groups:
        out += [
            "No charted results. " + _no_results_reason(sweep, analysis),
            "",
        ]
        return "\n".join(out)

    excluded = analysis.excluded
    if any((excluded.failed, excluded.cancelled, excluded.unfinished)):
        out += [
            f"{excluded.failed} failed, {excluded.cancelled} cancelled and "
            f"{excluded.unfinished} unfinished run(s) are counted here but not charted.",
            "",
        ]

    if len(analysis.groups) > 1:
        out += [
            f"Results fall into {len(analysis.groups)} sets that cannot be compared with "
            "each other — they differ in host, GPU, vLLM version or where the benchmark "
            "client ran. Each has its own table below. Do not read across them.",
            "",
        ]

    for group in analysis.groups:
        # The label already names host, GPU and vLLM version — it is what partitioned
        # these points in the first place. Only what it leaves out goes underneath it.
        out += [
            f"## {group.label}",
            "",
            f"Benchmark client: {group.bench_client_location}. {group.run_count} run(s).",
            "",
        ]
        for warning in group.warnings:
            out += [f"> {warning}", ""]

        headers = (
            ["", "Config", "Workload", "TP"] + [label for _, label in REPORT_METRICS] + ["reps"]
        )
        out += ["| " + " | ".join(headers) + " |"]
        out += ["| " + " | ".join("---" for _ in headers) + " |"]
        for point in group.points:
            # The frontier marker is the whole reason to read the table in order: these
            # are the configurations nothing else beats on both axes at once.
            row = [
                "★" if point.on_pareto_frontier else "",
                point.config_name,
                point.workload_name,
                str(point.tensor_parallel_size),
                *(_cell(point, key) for key, _ in REPORT_METRICS),
                str(point.replicates),
            ]
            out += ["| " + " | ".join(row) + " |"]
        out += [""]

        frontier = [p for p in group.points if p.on_pareto_frontier]
        if frontier:
            out += [
                f"★ marks the {len(frontier)} point(s) on the Pareto frontier of per-GPU "
                "throughput against per-user output rate — nothing measured here beats "
                "them on both at once. Everything else is dominated: some starred point "
                "is better on both axes, so there is no reason to run it.",
                "",
            ]
        spread_notes = {p.spread_note for p in group.points if p.replicates > 1}
        for note in sorted(spread_notes):
            out += [note, ""]
        out += [
            "± is half the observed range across replicates. A difference smaller than "
            "that is not a result.",
            "",
        ]

    return "\n".join(out)


def _no_results_reason(sweep: SweepOut, analysis: AnalysisOut) -> str:
    """Why an empty report is empty.

    An agent that gets a blank report and no reason will retry, or worse conclude the
    configuration performed at zero.
    """
    if sweep.progress.total == 0:
        return "This sweep has no runs."
    if sweep.progress.succeeded == 0:
        return "No run in it succeeded."
    if analysis.excluded.succeeded_without_summary:
        return (
            f"{analysis.excluded.succeeded_without_summary} run(s) finished but recorded no "
            "summary metrics, which means the benchmark output did not parse."
        )
    return "Its runs carry no charted metrics."
