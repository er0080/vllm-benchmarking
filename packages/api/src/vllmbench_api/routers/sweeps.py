"""Sweep authoring.

Creating a sweep **materializes every run immediately**, as QUEUED rows, rather than
storing the axes and letting the orchestrator generate points as it goes.

That choice buys four things at once. The full plan is visible the moment it is created,
so an author can see what they asked for before it costs an hour of GPU time. Progress is
a count of rows rather than an estimate. Resume after a restart is free, because the
remaining work is simply the runs that are still queued — nothing has to be recomputed,
and nothing depends on the orchestrator remembering where it was. And a run carries the
same provenance whether it came from a sweep or was triggered by hand.

The alternative — generating points lazily — puts the plan in one process's memory, which
is precisely the thing a multi-hour job must not depend on.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_api.analysis import RunSource
from vllmbench_api.deps import SessionDep
from vllmbench_api.duration import (
    DurationEstimate,
    PlannedRun,
    RunTiming,
    estimate_remaining,
    plan_engine_loads,
)
from vllmbench_api.hashing import config_hash, normalize_yaml
from vllmbench_api.reports import render_sweep_report
from vllmbench_api.routers.analysis import analysis_points
from vllmbench_api.schemas import DurationEstimateOut, SweepCreate, SweepOut, SweepProgress
from vllmbench_api.sweep_plan import (
    SweepPlanError,
    expand,
    read_tensor_parallel_size,
    tensor_parallel_variant,
    validate_tensor_parallel,
)
from vllmbench_db.enums import RunStatus, SweepStatus
from vllmbench_db.models import GpuHost, Run, ServerConfig, Sweep, Workload

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sweeps", tags=["sweeps"])


async def _config_for(session: AsyncSession, *, name: str, yaml_text: str) -> ServerConfig:
    """Fetch or create the config with this exact text.

    Content addressing means a variant that already exists is reused rather than
    duplicated — re-running the same sweep does not litter the config table, and two
    sweeps referring to "TP=4 of this config" refer to the same row.
    """
    normalized = normalize_yaml(yaml_text)
    digest = config_hash(normalized)

    existing = await session.scalar(select(ServerConfig).where(ServerConfig.config_hash == digest))
    if existing is not None:
        return existing

    config = ServerConfig(config_hash=digest, name=name[:128], yaml=normalized)
    session.add(config)
    await session.flush()
    return config


async def _resolve_configs(
    session: AsyncSession,
    request: SweepCreate,
    host: GpuHost,
) -> list[ServerConfig]:
    """The config axis, after applying the tensor-parallel axis to it.

    Derived variants are stored as ordinary configs before anything else happens, so by
    the time runs are created there is no such thing as a "generated" config — only
    configs, each hashed by its own text (see sweep_plan for why that matters).
    """
    bases: list[ServerConfig] = []
    for config_id in request.server_config_ids:
        config = await session.get(ServerConfig, config_id)
        if config is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"server config {config_id} not found",
            )
        bases.append(config)

    if request.tensor_parallel_sizes is None:
        # No TP axis: still check what each config asks for against what the host has,
        # because the failure it prevents is a wrong per-GPU number rather than an error.
        for config in bases:
            declared = read_tensor_parallel_size(config.yaml)
            if declared is not None:
                validate_tensor_parallel(
                    declared, host_gpu_count=host.gpu_count, host_name=host.name
                )
        return bases

    for size in request.tensor_parallel_sizes:
        validate_tensor_parallel(size, host_gpu_count=host.gpu_count, host_name=host.name)

    variants: list[ServerConfig] = []
    for base in bases:
        for size in request.tensor_parallel_sizes:
            variants.append(
                await _config_for(
                    session,
                    name=f"{base.name} TP{size}",
                    yaml_text=tensor_parallel_variant(base.yaml, size),
                )
            )
    return variants


async def _progress(session: AsyncSession, sweep_id: uuid.UUID) -> SweepProgress:
    rows = await session.execute(
        select(Run.status, func.count()).where(Run.sweep_id == sweep_id).group_by(Run.status)
    )
    progress = SweepProgress()
    for run_status, count in rows:
        setattr(progress, str(run_status), count)
        progress.total += count
    return progress


async def _engine_starts(session: AsyncSession, sweep_id: uuid.UUID) -> int:
    """Count config changes along the plan, as it was actually materialized."""
    hashes = list(
        (
            await session.execute(
                select(Run.config_hash).where(Run.sweep_id == sweep_id).order_by(Run.sweep_seq)
            )
        ).scalars()
    )
    starts = 0
    previous: str | None = None
    for digest in hashes:
        if digest != previous:
            starts += 1
            previous = digest
    return starts


#: Sweep states with work still ahead of them. Anything else has nothing to estimate, so
#: the query below is skipped rather than run to produce a zero.
UNFINISHED_SWEEP_STATES = (SweepStatus.QUEUED, SweepStatus.RUNNING)


async def _estimate(session: AsyncSession, sweep_id: uuid.UUID) -> DurationEstimate:
    """Time remaining, extrapolated from this sweep's own completed runs.

    One query for the whole plan, in execution order, because the estimate needs both
    halves at once: which runs are done and how long they took, and which are left and
    whether each will restart the engine. Splitting it would risk the two halves
    disagreeing about the order, which is exactly what the engine-load accounting depends
    on.
    """
    rows = (
        await session.execute(
            select(Run.status, Run.config_hash, Run.workload_hash, Run.started_at, Run.finished_at)
            .where(Run.sweep_id == sweep_id)
            .order_by(Run.sweep_seq)
        )
    ).all()

    loads = plan_engine_loads([row.config_hash for row in rows])
    observed: list[RunTiming] = []
    remaining: list[PlannedRun] = []
    for row, needs_load in zip(rows, loads, strict=True):
        if row.status is RunStatus.SUCCEEDED and row.started_at and row.finished_at:
            observed.append(
                RunTiming(
                    workload_hash=row.workload_hash,
                    seconds=(row.finished_at - row.started_at).total_seconds(),
                    included_engine_load=needs_load,
                )
            )
        elif row.status not in (RunStatus.FAILED, RunStatus.CANCELLED):
            # A run already starting or benchmarking is counted whole. Overshooting by
            # part of one run is a better error than a countdown that runs out early.
            remaining.append(
                PlannedRun(workload_hash=row.workload_hash, needs_engine_load=needs_load)
            )

    return estimate_remaining(observed, remaining)


async def _to_out(session: AsyncSession, sweep: Sweep) -> SweepOut:
    out = SweepOut.model_validate(sweep)
    out.progress = await _progress(session, sweep.id)
    out.engine_starts = await _engine_starts(session, sweep.id)
    if sweep.status in UNFINISHED_SWEEP_STATES:
        out.estimated_remaining = DurationEstimateOut.model_validate(
            await _estimate(session, sweep.id), from_attributes=True
        )
    return out


#: Runs one sweep may materialize. Generous enough that no real matrix hits it — a
#: 4-config x 4-workload x 3-replicate sweep is 48 — and low enough that a caller which
#: got its product wrong finds out at authoring time rather than after filling the queue
#: with days of GPU work.
MAX_SWEEP_RUNS = 500

#: Sweep states that own a host. A sweep still queued has runs an orchestrator may claim
#: at any moment, so it holds the host just as firmly as a running one.
ACTIVE_SWEEP_STATES = (SweepStatus.QUEUED, SweepStatus.RUNNING)


async def _refuse_if_busy(session: AsyncSession, host: GpuHost) -> None:
    """One active sweep per host.

    Enforced here rather than in any one interface, because the reason is physical: two
    sweeps against one host interleave their engine restarts, so each pays for the other's
    config changes and neither measures what it planned to. It is also how two callers
    that cannot see each other — a person in the UI and an agent over MCP — would collide.
    """
    existing = await session.scalar(
        select(Sweep)
        .where(Sweep.gpu_host_id == host.id)
        .where(Sweep.status.in_(ACTIVE_SWEEP_STATES))
        .limit(1)
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"{host.name} is already running sweep {existing.name!r} ({existing.id}); "
                "cancel it or wait for it to finish"
            ),
        )


@router.post("", response_model=SweepOut, status_code=status.HTTP_201_CREATED)
async def create_sweep(request: SweepCreate, session: SessionDep) -> SweepOut:
    host = await session.get(GpuHost, request.gpu_host_id)
    if host is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="gpu host not found")

    workloads: list[Workload] = []
    for workload_id in request.workload_ids:
        workload = await session.get(Workload, workload_id)
        if workload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"workload {workload_id} not found"
            )
        workloads.append(workload)

    await _refuse_if_busy(session, host)

    try:
        configs = await _resolve_configs(session, request, host)
        plan = expand(
            config_count=len(configs),
            workload_count=len(workloads),
            replicates=request.replicates,
            order=request.replicate_order,
        )
        if len(plan) > MAX_SWEEP_RUNS:
            raise SweepPlanError(
                f"this sweep would create {len(plan)} runs, over the {MAX_SWEEP_RUNS} limit; "
                "narrow an axis or split it"
            )
    except SweepPlanError as exc:
        # 422: the request was well-formed but describes a sweep that cannot be run.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    sweep = Sweep(
        name=request.name,
        description=request.description,
        # Queued rather than draft: materializing the runs *is* committing to them, and a
        # sweep whose runs are claimable while it claims to be a draft would be lying.
        status=SweepStatus.QUEUED,
        gpu_host_id=host.id,
        replicates=request.replicates,
        replicate_order=request.replicate_order,
        initiated_by=request.initiated_by,
        initiated_by_client=request.initiated_by_client,
        # Invariant 7's chain of custody: the host's own declaration decides, and every
        # run below inherits the same answer. Nothing infers it later.
        is_synthetic=host.synthetic_source is not None,
    )
    session.add(sweep)
    await session.flush()

    for point in plan:
        config = configs[point.config_index]
        workload = workloads[point.workload_index]
        session.add(
            Run(
                sweep_id=sweep.id,
                sweep_seq=point.seq,
                replicate_idx=point.replicate_idx,
                gpu_host_id=host.id,
                server_config_id=config.id,
                workload_id=workload.id,
                status=RunStatus.QUEUED,
                config_hash=config.config_hash,
                workload_hash=workload.workload_hash,
                gpu_count=max(1, host.gpu_count),
                # What the config asks for. The orchestrator overwrites this with what
                # NVML observed once the engine is up — request and outcome are kept as
                # separate facts on purpose.
                tensor_parallel_size=read_tensor_parallel_size(config.yaml) or 1,
                is_synthetic=host.synthetic_source is not None,
                synthetic_source=host.synthetic_source,
                initiated_by=request.initiated_by,
                initiated_by_client=request.initiated_by_client,
            )
        )

    await session.commit()
    log.info(
        "sweep %s: %d runs across %d configs x %d workloads x %d replicates (%s)",
        sweep.id,
        len(plan),
        len(configs),
        len(workloads),
        request.replicates,
        request.replicate_order,
    )
    return await _to_out(session, sweep)


@router.get("", response_model=list[SweepOut])
async def list_sweeps(session: SessionDep, limit: int = 50) -> list[SweepOut]:
    sweeps = list(
        (
            await session.execute(
                select(Sweep).order_by(Sweep.created_at.desc()).limit(min(limit, 200))
            )
        ).scalars()
    )
    return [await _to_out(session, sweep) for sweep in sweeps]


@router.get("/{sweep_id}", response_model=SweepOut)
async def get_sweep(sweep_id: uuid.UUID, session: SessionDep) -> SweepOut:
    sweep = await session.get(Sweep, sweep_id)
    if sweep is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="sweep not found")
    return await _to_out(session, sweep)


@router.post("/{sweep_id}/cancel", response_model=SweepOut)
async def cancel_sweep(sweep_id: uuid.UUID, session: SessionDep) -> SweepOut:
    """Stop a sweep from doing any more work.

    Cancels every run that has not started. A run already in flight is left alone here —
    it is executing on another host, and reaching in to mark it cancelled from this side
    would leave the orchestrator writing results for a run the database says was
    cancelled. Stopping the in-flight one is the orchestrator's job.
    """
    sweep = await session.get(Sweep, sweep_id)
    if sweep is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="sweep not found")

    if sweep.status in (SweepStatus.SUCCEEDED, SweepStatus.FAILED, SweepStatus.CANCELLED):
        # Terminal already. Not an error — cancelling twice should be safe.
        return await _to_out(session, sweep)

    queued = list(
        (
            await session.execute(
                select(Run).where(Run.sweep_id == sweep_id, Run.status == RunStatus.QUEUED)
            )
        ).scalars()
    )
    for run in queued:
        run.status = RunStatus.CANCELLED
        run.error = "cancelled with its sweep before it started"

    sweep.status = SweepStatus.CANCELLED
    await session.commit()
    log.info("sweep %s cancelled; %d queued runs cancelled with it", sweep_id, len(queued))
    return await _to_out(session, sweep)


@router.get("/{sweep_id}/report")
async def sweep_report(
    sweep_id: uuid.UUID,
    session: SessionDep,
    download: bool = False,
) -> Response:
    """One sweep written out as markdown, for handing to someone who does not run this.

    The same report the MCP resource serves — one implementation, so the two cannot come
    to disagree about what a sweep measured.

    Shareable means self-contained: the caveats are inline rather than appended, the
    synthetic banner precedes any number, and each comparability group is its own section
    with its own heading. A recipient reading only the tables still cannot put two GPU
    models in one comparison, because they were never in one table.
    """
    sweep = await get_sweep(sweep_id, session)
    analysis = await analysis_points(
        session,
        source=RunSource.SYNTHETIC if sweep.is_synthetic else RunSource.REAL,
        sweep_id=[sweep_id],
    )
    body = render_sweep_report(sweep, analysis)

    headers = {}
    if download:
        stem = "".join(c if c.isalnum() or c in "-_" else "-" for c in sweep.name.lower())
        headers["Content-Disposition"] = (
            f'attachment; filename="{stem.strip("-") or "sweep"}-{str(sweep_id)[:8]}.md"'
        )
    # text/markdown so a browser shows it and a client can render it; the bytes are the
    # same either way.
    return Response(content=body, media_type="text/markdown; charset=utf-8", headers=headers)
