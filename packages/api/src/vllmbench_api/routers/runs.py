"""Configurations, workloads, and runs.

Creating a run does not execute it. The API records the intent and returns; the
orchestrator picks it up. That split exists because a run takes minutes to hours — an
HTTP request cannot hold it, and more importantly a control-plane restart must not kill
work in flight (ROADMAP 0.4.0).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from vllmbench_api.deps import SessionDep
from vllmbench_api.hashing import config_hash, normalize_yaml, workload_hash
from vllmbench_api.schemas import (
    ConfigCreate,
    ConfigOut,
    EngineSampleOut,
    GpuSampleOut,
    RunCreate,
    RunOut,
    RunTelemetryOut,
    WorkloadCreate,
    WorkloadOut,
)
from vllmbench_db.enums import InitiatedBy, RunStatus
from vllmbench_db.models import EngineSample, GpuHost, GpuSample, Run, ServerConfig, Workload

router = APIRouter(prefix="/api", tags=["runs"])


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


@router.post("/configs", response_model=ConfigOut, status_code=status.HTTP_201_CREATED)
async def create_config(payload: ConfigCreate, session: SessionDep) -> ServerConfig:
    digest = config_hash(payload.yaml)

    # Content addressing means re-submitting the same YAML is idempotent rather than an
    # error. Sweep authoring will do exactly that, repeatedly.
    existing = await session.scalar(select(ServerConfig).where(ServerConfig.config_hash == digest))
    if existing is not None:
        return existing

    config = ServerConfig(
        config_hash=digest,
        name=payload.name,
        yaml=normalize_yaml(payload.yaml),
        notes=payload.notes,
    )
    session.add(config)
    await session.commit()
    await session.refresh(config)
    return config


@router.get("/configs", response_model=list[ConfigOut])
async def list_configs(session: SessionDep) -> list[ServerConfig]:
    result = await session.execute(select(ServerConfig).order_by(ServerConfig.created_at.desc()))
    return list(result.scalars())


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------

_WORKLOAD_IDENTITY_FIELDS = (
    "dataset_name",
    "dataset_path",
    "hf_name",
    "num_prompts",
    "request_rate",
    "max_concurrency",
    "burstiness",
    "input_len",
    "output_len",
)


@router.post("/workloads", response_model=WorkloadOut, status_code=status.HTTP_201_CREATED)
async def create_workload(payload: WorkloadCreate, session: SessionDep) -> Workload:
    # The name is deliberately not part of the identity: two workloads that send the
    # same traffic are the same workload, whatever they are called.
    identity = {field: getattr(payload, field) for field in _WORKLOAD_IDENTITY_FIELDS}
    digest = workload_hash(identity)

    existing = await session.scalar(select(Workload).where(Workload.workload_hash == digest))
    if existing is not None:
        return existing

    workload = Workload(workload_hash=digest, name=payload.name, **identity)
    session.add(workload)
    await session.commit()
    await session.refresh(workload)
    return workload


@router.get("/workloads", response_model=list[WorkloadOut])
async def list_workloads(session: SessionDep) -> list[Workload]:
    result = await session.execute(select(Workload).order_by(Workload.created_at.desc()))
    return list(result.scalars())


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@router.post("/runs", response_model=RunOut, status_code=status.HTTP_202_ACCEPTED)
async def create_run(payload: RunCreate, session: SessionDep) -> Run:
    """Queue a run. 202, not 201: the work has been accepted, not done."""
    host = await session.get(GpuHost, payload.gpu_host_id)
    config = await session.get(ServerConfig, payload.server_config_id)
    workload = await session.get(Workload, payload.workload_id)

    missing = [
        name
        for name, value in (
            ("gpu_host_id", host),
            ("server_config_id", config),
            ("workload_id", workload),
        )
        if value is None
    ]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown {', '.join(missing)}"
        )
    assert host is not None and config is not None and workload is not None

    run = Run(
        gpu_host_id=host.id,
        server_config_id=config.id,
        workload_id=workload.id,
        status=RunStatus.QUEUED,
        # Provenance copied at creation, not joined at read time. What is knowable now
        # is recorded now; the rest is filled in by the orchestrator from what actually
        # ran (invariant 6).
        config_hash=config.config_hash,
        workload_hash=workload.workload_hash,
        gpu_count=max(1, host.gpu_count),
        # Invariant 7's chain of custody: the host's declaration, made by the agent
        # itself, decides whether this run is quarantined. Nothing infers it.
        is_synthetic=host.synthetic_source is not None,
        synthetic_source=host.synthetic_source,
        initiated_by=InitiatedBy.UI,
    )
    session.add(run)
    await session.commit()
    # `summary` is eager-loaded explicitly (it is None for a fresh run) because the
    # relationship is lazy="raise_on_sql". Serializing without this would attempt a lazy
    # load inside the response model and fail — which is precisely what that setting is
    # for: turning a runtime MissingGreenlet into an obvious, immediate error.
    await session.refresh(run, attribute_names=["summary"])
    return run


@router.get("/runs", response_model=list[RunOut])
async def list_runs(session: SessionDep, limit: int = 50) -> list[Run]:
    result = await session.execute(
        select(Run)
        .options(selectinload(Run.summary))
        .order_by(Run.queued_at.desc())
        .limit(min(limit, 200))
    )
    return list(result.scalars())


@router.get("/runs/{run_id}", response_model=RunOut)
async def get_run(run_id: uuid.UUID, session: SessionDep) -> Run:
    run = await session.get(Run, run_id, options=[selectinload(Run.summary)])
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return run


@router.get("/runs/{run_id}/telemetry", response_model=RunTelemetryOut)
async def get_run_telemetry(run_id: uuid.UUID, session: SessionDep) -> RunTelemetryOut:
    """The engine and per-device series for one run.

    A separate endpoint rather than an expansion of the run payload: a long run can carry
    thousands of samples, and the runs list polls every two seconds while anything is in
    flight. Attaching telemetry to that would multiply the cost of the poll by the size
    of the largest run on the page.

    Per device, never aggregated here. One device at 60% while its peer sits at 95% is
    the finding; the mean of the two is not (invariant 8, and the `gpu_sample` keying
    rule in CLAUDE.md).
    """
    if await session.get(Run, run_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")

    engine = list(
        (
            await session.execute(
                select(EngineSample)
                .where(EngineSample.run_id == run_id)
                .order_by(EngineSample.sampled_at)
            )
        ).scalars()
    )
    gpu = list(
        (
            await session.execute(
                select(GpuSample)
                .where(GpuSample.run_id == run_id)
                .order_by(GpuSample.sampled_at, GpuSample.gpu_index)
            )
        ).scalars()
    )

    return RunTelemetryOut(
        run_id=run_id,
        engine=[EngineSampleOut.model_validate(s) for s in engine],
        gpu=[GpuSampleOut.model_validate(s) for s in gpu],
        gpu_indices=sorted({s.gpu_index for s in gpu}),
        sample_count=len(engine) + len(gpu),
    )
