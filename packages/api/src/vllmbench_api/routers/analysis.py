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

from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import Select, func, select

from vllmbench_api.analysis import (
    BALANCE_METRICS,
    METRICS,
    METRICS_BY_KEY,
    PARETO_X,
    PARETO_Y,
    DeviceSummary,
    Group,
    Point,
    RunBalance,
    RunRecord,
    RunSource,
    Spread,
    build_groups,
    config_diff,
    derive_per_user_rates,
    metric_delta,
    pareto_frontier,
    provenance_differences,
    scaling_curves,
    spread_note,
)
from vllmbench_api.deps import SessionDep
from vllmbench_api.export import analysis_columns, analysis_rows, filename, to_csv
from vllmbench_api.hashing import config_hash
from vllmbench_api.schemas import (
    AnalysisOut,
    BalanceMetricOut,
    ComparisonOut,
    ComparisonSideOut,
    DeviceBalanceGroupOut,
    DeviceBalanceOut,
    DeviceSummaryOut,
    DiffLineOut,
    ExcludedOut,
    GroupOut,
    MetricComparisonOut,
    MetricOut,
    PointOut,
    ProvenanceDifferenceOut,
    RunBalanceOut,
    ScalingCurveOut,
    ScalingGroupOut,
    ScalingOut,
    ScalingStepOut,
    SpreadOut,
)
from vllmbench_api.sweep_plan import config_family_text
from vllmbench_db.enums import RunStatus
from vllmbench_db.models import (
    GpuHost,
    GpuSample,
    Run,
    RunSummary,
    ServerConfig,
    Sweep,
    Workload,
)

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
        imported_from=run.imported_from,
        failed_requests=summary.failed_requests,
        successful_requests=summary.successful_requests,
        config_hash=run.config_hash,
        config_name=config_name,
        config_family=_family_hash(config_yaml),
        workload_hash=run.workload_hash,
        workload_name=workload.name,
        gpu_count=run.gpu_count,
        tensor_parallel_size=run.tensor_parallel_size,
        pipeline_parallel_size=run.pipeline_parallel_size,
        device_indices=tuple(run.device_indices or ()),
        speculative_method=run.speculative_method,
        speculative_tokens=run.speculative_tokens,
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
        speculative_method=point.speculative_method,
        speculative_tokens=point.speculative_tokens,
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
    # An unrecognised axis falls back to the default rather than erroring, and that is
    # deliberate *for this caller*. The reader is a browser holding a URL that may have
    # been bookmarked across a release; blanking their chart to punish a stale query
    # string helps nobody, and the response states which axis it actually used, which the
    # chart then labels. A person cannot miss it.
    #
    # The MCP surface must not inherit this. An agent has no axis label to read, and a
    # substitution there is a successful call answering a question that was not asked —
    # so ``vllmbench_api.mcp_server._metric_key`` refuses first. If you are here to make
    # the two consistent, that is the direction: keep this, keep that.
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


@router.get("/device-balance", response_model=DeviceBalanceOut)
async def analysis_device_balance(
    session: SessionDep,
    source: RunSource = RunSource.REAL,
    host_id: uuid.UUID | None = None,
    sweep_id: Annotated[list[uuid.UUID] | None, Query()] = None,
    config_hash: Annotated[list[str] | None, Query()] = None,
    tensor_parallel_size: Annotated[list[int] | None, Query()] = None,
    since: dt.datetime | None = None,
    limit: int = DEFAULT_RUN_LIMIT,
) -> DeviceBalanceOut:
    """How evenly a tensor-parallel run's devices shared the work.

    This is the question ``gpu_sample`` is keyed per device to answer. One GPU at 60%
    while its peer sits at 95% means a third of a device was idle for the whole run — a
    finding that the host-level average, the one number it would have been cheaper to
    store, destroys completely.

    Per run rather than per measurement point. Imbalance is a property of one execution,
    and averaging it across replicates would hide a single run that went wrong — which is
    exactly the run worth looking at.
    """
    capped = max(1, min(limit, MAX_RUN_LIMIT))
    filters = _Filters(
        source=source,
        host_id=host_id,
        sweep_ids=sweep_id,
        config_hashes=config_hash,
        tensor_parallel_sizes=tensor_parallel_size,
        since=since,
    )

    records = await _load_records(session, filters, capped)
    by_run = {record.run_id: record for record in records}

    # Aggregated in the database, grouped exactly the way the table is keyed. Pulling
    # every sample back to average it here would move tens of thousands of rows per run
    # to compute six numbers per device.
    devices: dict[uuid.UUID, list[DeviceSummary]] = {}
    if by_run:
        rows = await session.execute(
            select(
                GpuSample.run_id,
                GpuSample.gpu_index,
                func.count().label("samples"),
                func.avg(GpuSample.sm_utilization_pct),
                func.max(GpuSample.sm_utilization_pct),
                # Peak rather than mean: VRAM is claimed and held, so the high-water mark
                # is what the device actually needed. A mean would be dragged down by the
                # ramp at the start of every run.
                func.max(GpuSample.memory_used_bytes),
                func.avg(GpuSample.power_watts),
            )
            .where(GpuSample.run_id.in_(list(by_run)))
            .group_by(GpuSample.run_id, GpuSample.gpu_index)
            .order_by(GpuSample.run_id, GpuSample.gpu_index)
        )
        for run_id, gpu_index, samples, sm_mean, sm_max, memory, power in rows:
            devices.setdefault(run_id, []).append(
                DeviceSummary(
                    gpu_index=gpu_index,
                    samples=samples,
                    sm_utilization_pct=float(sm_mean) if sm_mean is not None else None,
                    sm_utilization_max=float(sm_max) if sm_max is not None else None,
                    memory_used_bytes=float(memory) if memory is not None else None,
                    power_watts=float(power) if power is not None else None,
                )
            )

    groups = build_groups(records)
    out_groups: list[DeviceBalanceGroupOut] = []
    for group in groups:
        balances: list[RunBalance] = []
        for point in group.points:
            for run_id in point.run_ids:
                found = devices.get(run_id)
                if not found:
                    continue
                record = by_run[run_id]
                balances.append(
                    RunBalance(
                        run_id=run_id,
                        config_name=point.config_name,
                        workload_name=point.workload_name,
                        tensor_parallel_size=record.tensor_parallel_size,
                        gpu_count=record.gpu_count,
                        replicate_idx=record.replicate_idx,
                        finished_at=record.finished_at,
                        devices=tuple(found),
                    )
                )
        # Worst first: the reason to open this view is to find the run that went wrong,
        # not to page through the ones that did not.
        balances.sort(key=lambda b: (-(b.worst_imbalance or -1.0), b.config_name))
        out_groups.append(
            DeviceBalanceGroupOut(
                **_group_header(group),  # type: ignore[arg-type]
                runs=[
                    RunBalanceOut(
                        run_id=balance.run_id,
                        config_name=balance.config_name,
                        workload_name=balance.workload_name,
                        tensor_parallel_size=balance.tensor_parallel_size,
                        gpu_count=balance.gpu_count,
                        replicate_idx=balance.replicate_idx,
                        finished_at=balance.finished_at,
                        devices=[
                            DeviceSummaryOut(
                                gpu_index=device.gpu_index,
                                samples=device.samples,
                                sm_utilization_pct=device.sm_utilization_pct,
                                sm_utilization_max=device.sm_utilization_max,
                                memory_used_bytes=device.memory_used_bytes,
                                power_watts=device.power_watts,
                            )
                            for device in balance.devices
                        ],
                        imbalances=balance.imbalances,
                        worst_imbalance=balance.worst_imbalance,
                        is_single_device=balance.is_single_device,
                    )
                    for balance in balances
                ],
            )
        )

    return DeviceBalanceOut(
        source=str(source),
        run_count=len(records),
        truncated=len(records) >= capped,
        limit=capped,
        metrics=[BalanceMetricOut(key=key, label=label) for key, label in BALANCE_METRICS],
        excluded=await _excluded(session, filters),
        groups=[g for g in out_groups if g.runs],
        runs_without_telemetry=sum(1 for run_id in by_run if run_id not in devices),
    )


def _comparison_side(point: Point, record: RunRecord, config_yaml: str) -> ComparisonSideOut:
    return ComparisonSideOut(
        point_id=_point_id(point),
        config_hash=point.key.config_hash,
        config_name=point.config_name,
        config_yaml=config_yaml,
        workload_name=point.workload_name,
        tensor_parallel_size=point.tensor_parallel_size,
        gpu_count=point.gpu_count,
        replicates=point.replicates,
        spread_basis=point.basis,
        spread_note=spread_note(point.basis),
        gpu_host_name=record.gpu_host_name,
        gpu_model=record.gpu_model,
        vllm_version=record.vllm_version,
        bench_client_location=record.bench_client_location,
        metrics={key: _spread_out(spread) for key, spread in point.metrics.items()},
    )


@router.get("/compare", response_model=ComparisonOut)
async def analysis_compare(
    session: SessionDep,
    left: str,
    right: str,
    source: RunSource = RunSource.REAL,
) -> ComparisonOut:
    """Two measurement points side by side, with a diff of the configs behind them.

    ``left`` and ``right`` are ``point_id`` values from ``/points`` — ``config_hash`` and
    ``workload_hash`` joined by a colon.

    Unlike every chart, this endpoint will happily compare across a comparability
    boundary, and that is deliberate. A chart must not *silently* overlay two vLLM
    versions; a side-by-side is where that comparison is the subject rather than an
    accident, the reader named both sides, and every difference is listed back to them.
    Refusing here would block the comparison the vLLM version policy exists to enable.

    The config diff is over the exact stored text. Invariant 5 makes that text the config,
    so "what is different about these two" is which bytes differ — not which parsed
    settings differ, which would need opinions about vLLM's option set, would normalize
    away the comment explaining a value, and would call two configs identical when one of
    them declares a key twice.
    """
    wanted = {left, right}
    hashes = [half for point_id in wanted for half in [point_id.split(":", 1)[0]]]

    # Scoped to the two configs, so the query stays small whatever the history holds.
    records = await _load_records(
        session,
        _Filters(source=source, config_hashes=hashes),
        MAX_RUN_LIMIT,
    )
    points: dict[str, tuple[Point, RunRecord]] = {}
    by_run = {record.run_id: record for record in records}
    for group in build_groups(records):
        for point in group.points:
            point_id = _point_id(point)
            if point_id in wanted and point_id not in points:
                points[point_id] = (point, by_run[point.run_ids[0]])

    missing = sorted(wanted - set(points))
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no {source} measurement point for {', '.join(missing)}",
        )

    left_point, left_record = points[left]
    right_point, right_record = points[right]

    # Built with an explicit loop rather than dict(rows.all()): the two-column Row
    # resolves to dict's bytes overload and the type checker rejects it.
    yamls: dict[str, str] = {}
    for digest, text in (
        await session.execute(
            select(ServerConfig.config_hash, ServerConfig.yaml).where(
                ServerConfig.config_hash.in_(
                    [left_point.key.config_hash, right_point.key.config_hash]
                )
            )
        )
    ).all():
        yamls[digest] = text
    left_yaml = yamls.get(left_point.key.config_hash, "")
    right_yaml = yamls.get(right_point.key.config_hash, "")

    diff = config_diff(left_yaml, right_yaml)
    differences = provenance_differences(
        left_point, right_point, {"left": left_record, "right": right_record}
    )

    return ComparisonOut(
        source=str(source),
        left=_comparison_side(left_point, left_record, left_yaml),
        right=_comparison_side(right_point, right_record, right_yaml),
        config_diff=[
            DiffLineOut(
                kind=line.kind, text=line.text, left_no=line.left_no, right_no=line.right_no
            )
            for line in diff
        ],
        # Content addressing makes this exact: the same hash is the same bytes.
        configs_identical=left_point.key.config_hash == right_point.key.config_hash,
        provenance_differences=[
            ProvenanceDifferenceOut(
                field=d.field,
                label=d.label,
                left=d.left,
                right=d.right,
                invalidating=d.invalidating,
            )
            for d in differences
        ],
        metrics=[
            MetricComparisonOut(
                key=spec.key,
                label=spec.label,
                unit=spec.unit,
                better=spec.better,
                left=left_value,
                right=right_value,
                change=change,
                is_improvement=improvement,
            )
            for spec in METRICS
            for left_value, right_value in [
                (left_point.value(spec.key), right_point.value(spec.key))
            ]
            for change, improvement in [metric_delta(left_value, right_value, spec)]
            # A metric neither side measured is left out rather than shown as two dashes.
            if left_value is not None or right_value is not None
        ],
    )


@router.get("/export")
async def analysis_export(
    session: SessionDep,
    fmt: Annotated[str, Query(alias="format", pattern="^(csv|json)$")] = "csv",
    source: RunSource = RunSource.REAL,
    host_id: uuid.UUID | None = None,
    sweep_id: Annotated[list[uuid.UUID] | None, Query()] = None,
    config_hash: Annotated[list[str] | None, Query()] = None,
    workload_hash: Annotated[list[str] | None, Query()] = None,
    tensor_parallel_size: Annotated[list[int] | None, Query()] = None,
    vllm_version: Annotated[list[str] | None, Query()] = None,
    since: dt.datetime | None = None,
    limit: int = DEFAULT_RUN_LIMIT,
) -> Response:
    """The current result set as a file.

    Takes the same filters as `/points` and exports exactly what they select, so what
    lands in the file is what was on screen. A separate query shape would eventually
    disagree with the charts, and an export that disagrees with the chart it was taken
    from is worse than none.

    Every row carries its full provenance and its population, because a file is where a
    result goes to be read by someone who cannot see the filters that produced it
    (invariants 6 and 7). CSV additionally carries each metric's observed range beside its
    median: a difference smaller than a point's own spread is not a result, and a lone
    median gives a reader no way to know that.
    """
    analysis = await analysis_points(
        session,
        source=source,
        host_id=host_id,
        sweep_id=sweep_id,
        config_hash=config_hash,
        workload_hash=workload_hash,
        tensor_parallel_size=tensor_parallel_size,
        vllm_version=vllm_version,
        since=since,
        limit=limit,
    )

    if fmt == "json":
        # The analysis payload verbatim, groups intact. JSON can represent the grouping
        # that CSV has to flatten, so it keeps it rather than pre-flattening for parity.
        body = analysis.model_dump_json(indent=2)
        media_type = "application/json"
    else:
        body = to_csv(analysis_columns(), analysis_rows(analysis))
        media_type = "text/csv"

    name = filename("analysis", source.value, "json" if fmt == "json" else "csv")
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
