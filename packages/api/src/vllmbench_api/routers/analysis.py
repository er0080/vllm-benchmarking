"""Comparable run sets, for every analysis view.

One endpoint feeds the Pareto frontier, the tensor-parallel scaling view, the
latency-versus-concurrency curves and the saturation curves, because they are the same
question asked with different axes: which measurement points exist, what did each one
measure, and which of them may honestly be drawn together.

Giving each view its own query would mean four places that decide what "comparable"
means, and they would drift. The rules live in :mod:`vllmbench_api.analysis`; this module
only turns rows into that module's input and its output into JSON.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Annotated, Any, TypeVar

from fastapi import APIRouter, Query
from sqlalchemy import Select, func, select

from vllmbench_api.analysis import (
    METRICS,
    METRICS_BY_KEY,
    PARETO_X,
    PARETO_Y,
    Group,
    Point,
    RunRecord,
    RunSource,
    Spread,
    build_groups,
    derive_per_user_rates,
    pareto_frontier,
    scaling_curves,
    spread_note,
)
from vllmbench_api.deps import SessionDep
from vllmbench_api.hashing import config_hash
from vllmbench_api.schemas import (
    AnalysisOut,
    ExcludedOut,
    GroupOut,
    MetricOut,
    PointOut,
    ScalingCurveOut,
    ScalingGroupOut,
    ScalingOut,
    ScalingStepOut,
    SpreadOut,
)
from vllmbench_api.sweep_plan import config_family_text
from vllmbench_db.enums import RunStatus
from vllmbench_db.models import GpuHost, Run, RunSummary, ServerConfig, Sweep, Workload

#: Any SELECT the filters can narrow — the row query and the count queries alike.
#: Bound to ``Select[Any]`` because ``Select`` is invariant in its row type, so a
#: concrete bound would reject every caller.
_S = TypeVar("_S", bound=Select[Any])

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

#: Runs considered in one query. High enough that a real sweep matrix arrives whole, and
#: capped so a UI filter mistake cannot pull the entire history into memory. Whether the
#: cap actually bit is reported on the response rather than left for the reader to
#: notice: a chart quietly missing its last twenty points looks exactly like a chart.
DEFAULT_RUN_LIMIT = 500
MAX_RUN_LIMIT = 2000


@dataclass(frozen=True, slots=True)
class _Filters:
    """One description of "which runs are in scope", applied everywhere.

    A struct rather than loose arguments so the result query and the exclusion tally
    cannot drift apart. Counts computed over a different population than the chart would
    be worse than no counts at all: they would explain away points that were never in
    scope in the first place.
    """

    source: RunSource
    host_id: uuid.UUID | None = None
    sweep_ids: list[uuid.UUID] | None = None
    config_hashes: list[str] | None = None
    workload_hashes: list[str] | None = None
    tensor_parallel_sizes: list[int] | None = None
    vllm_versions: list[str] | None = None
    since: dt.datetime | None = None

    def apply(self, stmt: _S) -> _S:
        stmt = stmt.where(Run.is_synthetic == (self.source is RunSource.SYNTHETIC))
        if self.host_id is not None:
            stmt = stmt.where(Run.gpu_host_id == self.host_id)
        if self.sweep_ids:
            stmt = stmt.where(Run.sweep_id.in_(self.sweep_ids))
        if self.config_hashes:
            stmt = stmt.where(Run.config_hash.in_(self.config_hashes))
        if self.workload_hashes:
            stmt = stmt.where(Run.workload_hash.in_(self.workload_hashes))
        if self.tensor_parallel_sizes:
            stmt = stmt.where(Run.tensor_parallel_size.in_(self.tensor_parallel_sizes))
        if self.vllm_versions:
            stmt = stmt.where(Run.vllm_version.in_(self.vllm_versions))
        if self.since is not None:
            stmt = stmt.where(Run.finished_at >= self.since)
        return stmt

    def flipped(self) -> _Filters:
        """The same scope, over the other population.

        Used only to count what the caller is not being shown; the two results are never
        merged, which is the whole point of :class:`RunSource` (invariant 7).
        """
        other = RunSource.SYNTHETIC if self.source is RunSource.REAL else RunSource.REAL
        return replace(self, source=other)


@lru_cache(maxsize=512)
def _family_hash(config_yaml: str) -> str:
    """Identity of the config with its tensor-parallel width normalized away.

    Cached because a sweep's runs share a handful of configs and the query re-derives it
    per row; the cache is keyed on the exact text, so it can only ever return the answer
    for that text.
    """
    return config_hash(config_family_text(config_yaml))


def _to_record(
    run: Run,
    summary: RunSummary,
    config_name: str,
    config_yaml: str,
    workload: Workload,
    host_name: str,
    sweep: Sweep | None,
) -> RunRecord:
    metrics: dict[str, float | None] = {
        spec.key: getattr(summary, spec.key, None) for spec in METRICS
    }
    # Derived here, in one place, rather than by each consumer. Two views disagreeing
    # about what "per-user tok/s" means is the kind of difference nobody notices until a
    # decision has already been made on it.
    metrics.update(derive_per_user_rates(summary.tpot_ms_mean, summary.tpot_ms_p99))

    return RunRecord(
        run_id=run.id,
        finished_at=run.finished_at,
        gpu_host_id=run.gpu_host_id,
        gpu_host_name=host_name,
        gpu_model=run.gpu_model,
        vllm_version=run.vllm_version,
        bench_client_location=str(run.bench_client_location),
        driver_version=run.driver_version,
        cuda_version=run.cuda_version,
        config_hash=run.config_hash,
        config_name=config_name,
        config_family=_family_hash(config_yaml),
        workload_hash=run.workload_hash,
        workload_name=workload.name,
        gpu_count=run.gpu_count,
        tensor_parallel_size=run.tensor_parallel_size,
        pipeline_parallel_size=run.pipeline_parallel_size,
        device_indices=tuple(run.device_indices or ()),
        max_concurrency=workload.max_concurrency,
        request_rate=workload.request_rate,
        num_prompts=workload.num_prompts,
        sweep_id=run.sweep_id,
        sweep_name=sweep.name if sweep else None,
        replicate_idx=run.replicate_idx,
        replicate_order=sweep.replicate_order if sweep else None,
        metrics=metrics,
    )


def _point_out(point: Point, on_frontier: bool) -> PointOut:
    return PointOut(
        point_id=_point_id(point),
        config_hash=point.key.config_hash,
        config_family=point.family,
        config_name=point.config_name,
        workload_hash=point.key.workload_hash,
        workload_name=point.workload_name,
        tensor_parallel_size=point.tensor_parallel_size,
        pipeline_parallel_size=point.pipeline_parallel_size,
        gpu_count=point.gpu_count,
        max_concurrency=point.max_concurrency,
        request_rate=point.request_rate,
        num_prompts=point.num_prompts,
        replicates=point.replicates,
        run_ids=list(point.run_ids),
        sweep_ids=list(point.sweep_ids),
        spread_basis=point.basis,
        spread_note=spread_note(point.basis),
        latest_finished_at=point.latest_finished_at,
        on_pareto_frontier=on_frontier,
        metrics={key: _spread_out(spread) for key, spread in point.metrics.items()},
    )


def _group_header(group: Group) -> dict[str, object]:
    """The comparability facts, identical across every analysis endpoint.

    Shared so two views cannot label the same group differently — a reader switching
    between them has to be able to tell it is the same set of runs.
    """
    return {
        "group_id": f"{group.key.gpu_host_id}:{group.key.vllm_version}:"
        f"{group.key.gpu_model}:{group.key.bench_client_location}",
        "label": group.key.label,
        "gpu_host_id": group.key.gpu_host_id,
        "gpu_host_name": group.key.gpu_host_name,
        "gpu_model": group.key.gpu_model,
        "vllm_version": group.key.vllm_version,
        "bench_client_location": group.key.bench_client_location,
        "warnings": list(group.warnings),
        "run_count": group.run_count,
    }


def _point_id(point: Point) -> str:
    return f"{point.key.config_hash}:{point.key.workload_hash}"


def _group_out(group: Group, x_key: str, y_key: str) -> GroupOut:
    frontier = pareto_frontier(group.points, METRICS_BY_KEY[x_key], METRICS_BY_KEY[y_key])
    on_frontier = {p.key for p in frontier}
    return GroupOut(
        **_group_header(group),  # type: ignore[arg-type]
        points=[_point_out(p, p.key in on_frontier) for p in group.points],
        pareto_point_ids=[_point_id(p) for p in frontier],
    )


async def _load_records(session: SessionDep, filters: _Filters, capped: int) -> list[RunRecord]:
    """Succeeded runs with metrics, newest first, as the rules module's input.

    Shared by every analysis endpoint so they see the same population. A view built on a
    subtly different query would disagree with its neighbour about which runs exist,
    which is worse than either being wrong alone.
    """
    stmt = (
        select(Run, RunSummary, ServerConfig.name, ServerConfig.yaml, Workload, GpuHost.name, Sweep)
        .join(RunSummary, RunSummary.run_id == Run.id)
        .join(ServerConfig, ServerConfig.id == Run.server_config_id)
        .join(Workload, Workload.id == Run.workload_id)
        .join(GpuHost, GpuHost.id == Run.gpu_host_id)
        .outerjoin(Sweep, Sweep.id == Run.sweep_id)
        .where(Run.status == RunStatus.SUCCEEDED)
        .order_by(Run.finished_at.desc().nulls_last())
        .limit(capped)
    )
    rows = (await session.execute(filters.apply(stmt))).all()
    return [
        _to_record(run, summary, config_name, config_yaml, workload, host_name, sweep)
        for run, summary, config_name, config_yaml, workload, host_name, sweep in rows
    ]


def _metrics_out() -> list[MetricOut]:
    return [
        MetricOut(
            key=m.key,
            label=m.label,
            unit=m.unit,
            better=m.better,
            per_gpu=m.per_gpu,
            description=m.description,
        )
        for m in METRICS
    ]


@router.get("/points", response_model=AnalysisOut)
async def analysis_points(
    session: SessionDep,
    source: RunSource = RunSource.REAL,
    host_id: uuid.UUID | None = None,
    sweep_id: Annotated[list[uuid.UUID] | None, Query()] = None,
    config_hash: Annotated[list[str] | None, Query()] = None,
    workload_hash: Annotated[list[str] | None, Query()] = None,
    tensor_parallel_size: Annotated[list[int] | None, Query()] = None,
    vllm_version: Annotated[list[str] | None, Query()] = None,
    since: dt.datetime | None = None,
    pareto_x: str = PARETO_X,
    pareto_y: str = PARETO_Y,
    limit: int = DEFAULT_RUN_LIMIT,
) -> AnalysisOut:
    """Measurement points, partitioned into sets that may be charted together.

    ``source`` selects one population — real or synthetic — and there is no value that
    means both. Invariant 7 is a property of the request shape rather than a check that
    could be forgotten (see :class:`~vllmbench_api.analysis.RunSource`).
    """
    x_key = pareto_x if pareto_x in METRICS_BY_KEY else PARETO_X
    y_key = pareto_y if pareto_y in METRICS_BY_KEY else PARETO_Y
    capped = max(1, min(limit, MAX_RUN_LIMIT))

    filters = _Filters(
        source=source,
        host_id=host_id,
        sweep_ids=sweep_id,
        config_hashes=config_hash,
        workload_hashes=workload_hash,
        tensor_parallel_sizes=tensor_parallel_size,
        vllm_versions=vllm_version,
        since=since,
    )

    records = await _load_records(session, filters, capped)
    groups = build_groups(records)

    return AnalysisOut(
        source=str(source),
        run_count=len(records),
        truncated=len(records) >= capped,
        limit=capped,
        pareto_x=x_key,
        pareto_y=y_key,
        metrics=_metrics_out(),
        excluded=await _excluded(session, filters),
        groups=[_group_out(g, x_key, y_key) for g in groups],
    )


async def _excluded(session: SessionDep, filters: _Filters) -> ExcludedOut:
    """What the same filters matched but the chart is not showing.

    Reported because a chart cannot distinguish "this configuration was not tried" from
    "every replicate of it failed", and those lead to opposite conclusions. The
    other-population count is here for the same reason: a developer who has been working
    against the mock agent should be told their real-run chart is empty *because* their
    runs are synthetic, not left to wonder.
    """

    async def scalar(stmt: Select[tuple[int]]) -> int:
        return await session.scalar(stmt) or 0

    counted = filters.apply(select(func.count(Run.id)))

    by_status = {
        status: await scalar(counted.where(Run.status == status))
        for status in (RunStatus.FAILED, RunStatus.CANCELLED)
    }
    unfinished = await scalar(
        counted.where(
            Run.status.in_([RunStatus.QUEUED, RunStatus.STARTING, RunStatus.BENCHMARKING])
        )
    )
    # A succeeded run with no summary row means the flattening layer did not produce
    # one. Silence here would look identical to a run that was never attempted.
    no_summary = await scalar(
        counted.where(Run.status == RunStatus.SUCCEEDED).where(
            ~Run.id.in_(select(RunSummary.run_id))
        )
    )
    other = await scalar(filters.flipped().apply(select(func.count(Run.id))))

    return ExcludedOut(
        failed=by_status[RunStatus.FAILED],
        cancelled=by_status[RunStatus.CANCELLED],
        unfinished=unfinished,
        succeeded_without_summary=no_summary,
        other_source=other,
        other_source_name="synthetic" if filters.source is RunSource.REAL else "real",
    )


def _spread_out(spread: Spread) -> SpreadOut:
    return SpreadOut(
        n=spread.n,
        median=spread.median,
        mean=spread.mean,
        min=spread.minimum,
        max=spread.maximum,
        values=list(spread.values),
        relative_range=spread.relative_range,
    )


@router.get("/scaling", response_model=ScalingOut)
async def analysis_scaling(
    session: SessionDep,
    source: RunSource = RunSource.REAL,
    host_id: uuid.UUID | None = None,
    sweep_id: Annotated[list[uuid.UUID] | None, Query()] = None,
    workload_hash: Annotated[list[str] | None, Query()] = None,
    vllm_version: Annotated[list[str] | None, Query()] = None,
    since: dt.datetime | None = None,
    metric: str = PARETO_X,
    limit: int = DEFAULT_RUN_LIMIT,
) -> ScalingOut:
    """Does TP=N earn its extra devices?

    Curves are keyed by config family — the config text with its tensor-parallel line
    normalized away — and by workload. Both are necessary: a curve whose points are
    different configurations is a comparison of configurations, and one whose points are
    different workloads measures the traffic rather than the topology. Neither is
    scaling, and both would look exactly like it.

    Deliberately no ``tensor_parallel_size`` filter. Filtering the axis under study to a
    single value is the one query that cannot produce a curve, and offering it would
    only let a caller ask for an empty chart.
    """
    # Efficiency is per-GPU throughput at one width over per-GPU throughput at the
    # baseline. Fed an aggregate figure it would compute the *speedup* and label it
    # efficiency — reporting 2.0 for a config that merely kept up. Refusing is better
    # than silently answering a different question.
    spec = METRICS_BY_KEY.get(metric)
    metric_key = metric if spec is not None and spec.per_gpu else PARETO_X

    capped = max(1, min(limit, MAX_RUN_LIMIT))
    filters = _Filters(
        source=source,
        host_id=host_id,
        sweep_ids=sweep_id,
        workload_hashes=workload_hash,
        vllm_versions=vllm_version,
        since=since,
    )

    records = await _load_records(session, filters, capped)
    groups = build_groups(records)

    single_width = 0
    out_groups: list[ScalingGroupOut] = []
    for group in groups:
        curves = scaling_curves(group.points, metric_key=metric_key)
        charted = {c.family for c in curves}
        single_width += len({p.family for p in group.points} - charted)
        out_groups.append(
            ScalingGroupOut(
                **_group_header(group),  # type: ignore[arg-type]
                curves=[
                    ScalingCurveOut(
                        family=curve.family,
                        config_name=curve.config_name,
                        workload_hash=curve.workload_hash,
                        workload_name=curve.workload_name,
                        max_concurrency=curve.max_concurrency,
                        baseline_tp=curve.baseline_tp,
                        baseline_is_single_gpu=curve.baseline_is_single_gpu,
                        steps=[
                            ScalingStepOut(
                                tensor_parallel_size=step.tensor_parallel_size,
                                gpu_count=step.point.gpu_count,
                                point_id=_point_id(step.point),
                                config_name=step.point.config_name,
                                is_baseline=step.is_baseline,
                                speedup=step.speedup,
                                efficiency=step.efficiency,
                                per_gpu=(
                                    _spread_out(step.point.metrics[metric_key])
                                    if metric_key in step.point.metrics
                                    else None
                                ),
                                aggregate_median=(
                                    value * step.point.gpu_count
                                    if (value := step.point.value(metric_key)) is not None
                                    else None
                                ),
                            )
                            for step in curve.steps
                        ],
                    )
                    for curve in curves
                ],
            )
        )

    # Groups with no curve at all are dropped rather than shown empty: a comparison set
    # where nothing was measured at two widths has nothing to say about scaling.
    return ScalingOut(
        source=str(source),
        run_count=len(records),
        truncated=len(records) >= capped,
        limit=capped,
        metric=metric_key,
        metrics=_metrics_out(),
        excluded=await _excluded(session, filters),
        groups=[g for g in out_groups if g.curves],
        single_width_families=single_width,
    )
