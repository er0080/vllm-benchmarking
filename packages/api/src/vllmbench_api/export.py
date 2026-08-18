"""CSV and JSON export of a result set.

**A row that leaves this service must be able to state what produced it.** Invariant 6
does not stop at the database: an exported CSV is the form in which results travel to
people who cannot see the filters that produced them, get pasted into a spreadsheet next
to somebody else's numbers, and are charted by a tool that knows nothing about vLLM
versions. So every row carries its full provenance — host, GPU model, vLLM version,
benchmark-client location, tensor-parallel width — even though that repeats the same
values down a column. The redundancy is the point: it is what makes a stray row
self-describing.

For the same reason the population is a column, not just a filter. A synthetic row that
escapes into a spreadsheet has no flag to consult and no chart to warn it, and invariant 7
is a promise about where those numbers can appear, which has to survive the export.

Spreads travel with their medians. A CSV of medians alone is an invitation to compare two
numbers whose own run-to-run variation is larger than the difference between them.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable

from vllmbench_api.analysis import METRICS
from vllmbench_api.schemas import AnalysisOut, GroupOut, PointOut

#: Provenance carried on every exported row, in the order a reader scans them. First the
#: population, because it is the one value that determines whether the row may be compared
#: with anything at all.
PROVENANCE_COLUMNS = (
    "source",
    "gpu_host",
    "gpu_model",
    "vllm_version",
    "bench_client_location",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "gpu_count",
)

IDENTITY_COLUMNS = (
    "config_name",
    "config_hash",
    "workload_name",
    "workload_hash",
    "max_concurrency",
    "request_rate",
    "num_prompts",
)

#: What the numbers mean, kept adjacent to them rather than in a footnote a spreadsheet
#: will drop.
CONTEXT_COLUMNS = (
    "replicates",
    "spread_basis",
    "on_pareto_frontier",
    "latest_finished_at",
    "comparability_group",
    "run_ids",
)


def _metric_columns() -> list[str]:
    """Three columns per metric: the median and the range it came from.

    Not a single averaged column. A difference smaller than a point's own spread is not a
    result, and a reader holding only the median has no way to know that.
    """
    columns: list[str] = []
    for metric in METRICS:
        columns += [metric.key, f"{metric.key}__min", f"{metric.key}__max"]
    return columns


def analysis_columns() -> list[str]:
    return [
        *PROVENANCE_COLUMNS,
        *IDENTITY_COLUMNS,
        *CONTEXT_COLUMNS,
        *_metric_columns(),
    ]


def _row(analysis: AnalysisOut, group: GroupOut, point: PointOut) -> dict[str, object]:
    row: dict[str, object] = {
        "source": analysis.source,
        "gpu_host": group.gpu_host_name,
        "gpu_model": group.gpu_model or "",
        "vllm_version": group.vllm_version or "",
        "bench_client_location": group.bench_client_location,
        "tensor_parallel_size": point.tensor_parallel_size,
        "pipeline_parallel_size": point.pipeline_parallel_size,
        "gpu_count": point.gpu_count,
        "config_name": point.config_name,
        "config_hash": point.config_hash,
        "workload_name": point.workload_name,
        "workload_hash": point.workload_hash,
        # Null means unbounded, genuinely — no --max-concurrency, or --request-rate inf.
        # Written as empty rather than 0, which would mean the opposite.
        "max_concurrency": "" if point.max_concurrency is None else point.max_concurrency,
        "request_rate": "" if point.request_rate is None else point.request_rate,
        "num_prompts": point.num_prompts,
        "replicates": point.replicates,
        "spread_basis": point.spread_basis,
        "on_pareto_frontier": point.on_pareto_frontier,
        "latest_finished_at": (
            point.latest_finished_at.isoformat() if point.latest_finished_at else ""
        ),
        # The group a reader must not read across. Carried so that a sorted or filtered
        # spreadsheet still says which rows belong together.
        "comparability_group": group.label,
        # The way back to the raw records, which is what makes an exported figure
        # checkable rather than merely quotable.
        "run_ids": " ".join(str(run_id) for run_id in point.run_ids),
    }
    for metric in METRICS:
        spread = point.metrics.get(metric.key)
        row[metric.key] = "" if spread is None else spread.median
        row[f"{metric.key}__min"] = "" if spread is None else spread.min
        row[f"{metric.key}__max"] = "" if spread is None else spread.max
    return row


def analysis_rows(analysis: AnalysisOut) -> Iterable[dict[str, object]]:
    """One row per measurement point, flattened across comparability groups.

    Flattened, but never silently: `comparability_group` rides on every row. The grouping
    is what the charts use to refuse an invalid overlay, and a flat file cannot refuse
    anything — so the only honest thing it can do is say which rows belong together.
    """
    for group in analysis.groups:
        for point in group.points:
            yield _row(analysis, group, point)


def to_csv(columns: list[str], rows: Iterable[dict[str, object]]) -> str:
    """Rows as CSV text.

    `\\r\\n` line endings and full quoting, per RFC 4180: config names contain commas and
    spreadsheet imports are unforgiving about it.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def filename(kind: str, source: str, extension: str) -> str:
    """A download name that says what the file holds.

    The population is in the name as well as in a column, because the filename is the part
    that survives being forwarded, and "these are synthetic" is the one thing a recipient
    must not have to open the file to discover.
    """
    return f"vllmbench-{kind}-{source}.{extension}"
