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

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_api.deps import SessionDep
from vllmbench_api.hashing import config_hash, normalize_yaml
from vllmbench_api.schemas import SweepCreate, SweepOut, SweepProgress
from vllmbench_api.sweep_plan import (
    SweepPlanError,
    expand,
    read_tensor_parallel_size,
    tensor_parallel_variant,
    validate_tensor_parallel,
)
from vllmbench_db.enums import InitiatedBy, RunStatus, SweepStatus
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


async def _to_out(session: AsyncSession, sweep: Sweep) -> SweepOut:
    out = SweepOut.model_validate(sweep)
    out.progress = await _progress(session, sweep.id)
    out.engine_starts = await _engine_starts(session, sweep.id)
    return out


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

    try:
        configs = await _resolve_configs(session, request, host)
        plan = expand(
            config_count=len(configs),
            workload_count=len(workloads),
            replicates=request.replicates,
            order=request.replicate_order,
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
        initiated_by=InitiatedBy.UI,
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
                initiated_by=InitiatedBy.UI,
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
