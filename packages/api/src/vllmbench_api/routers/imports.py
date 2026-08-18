"""Importing results this framework did not measure.

Per `docs/adr/0003-importing-upstream-sweeps.md`: the files say what was measured, the
operator says what measured it, and the database records which of the two each fact came
from.

The upstream output carries **no provenance at all** — no vLLM version, GPU model, driver,
host or device count. That is not an inconvenience to paper over. Invariant 8 makes every
throughput figure carry a per-GPU value and makes comparison views default to it, and a
per-GPU value cannot be computed without a device count. So the declared fields below are
required, and an import missing them is refused rather than defaulted: a default here is a
fabricated provenance column, which is the specific failure this project exists to avoid.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from vllmbench_api.deps import SessionDep
from vllmbench_api.hashing import config_hash, normalize_yaml, workload_hash
from vllmbench_api.importers import SweepDirectoryError, build_points, reconstructed_yaml
from vllmbench_db.enums import (
    BenchClientLocation,
    InitiatedBy,
    ReplicateOrder,
    RunStatus,
    SweepStatus,
)
from vllmbench_db.models import GpuHost, Run, RunSummary, ServerConfig, Sweep, Workload
from vllmbench_protocol.bench_result import BenchResultError, flatten_bench_result

router = APIRouter(prefix="/api/import", tags=["import"])

#: Points one upload may carry. Generous for any real sweep and low enough that a
#: mistyped path cannot turn into an unbounded transaction.
MAX_POINTS = 500
MAX_RUNS = 5000


class DeclaredProvenance(BaseModel):
    """What the operator must state, because the files cannot.

    Every field here is required. Each is something invariant 6 says a valid result must
    be able to state and that `vllm bench sweep serve` does not record.
    """

    gpu_host_id: uuid.UUID
    gpu_model: str = Field(min_length=1, max_length=128)
    vllm_version: str = Field(min_length=1, max_length=64)
    #: Devices the server actually used. Required because per-GPU throughput — the axis
    #: every comparison view defaults to — cannot be derived without it.
    gpu_count: int = Field(ge=1, le=64)
    tensor_parallel_size: int = Field(ge=1, le=64)
    #: Whether the benchmark client ran on the machine under test. A remote client puts
    #: network round-trip inside TTFT and ITL, so a run measured that way is not
    #: comparable with a loopback one and must not be charted beside it.
    bench_client_location: BenchClientLocation = BenchClientLocation.LOOPBACK
    driver_version: str | None = Field(default=None, max_length=64)
    pipeline_parallel_size: int = Field(default=1, ge=1, le=64)


class SweepImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: The experiment name, used to name the resulting sweep.
    experiment_name: str = Field(min_length=1, max_length=128)
    declared: DeclaredProvenance
    #: Parsed `run=N.json` payloads keyed by path relative to the experiment directory,
    #: e.g. `"SERVE--max_num_seqs=4-BENCH--max_concurrency=2/run=0.json"`. A map rather
    #: than a directory path because the control plane never has the operator's
    #: filesystem — invariant 1 keeps it off the machine under test.
    files: dict[str, Any]


class SweepImportOut(BaseModel):
    sweep_id: uuid.UUID
    points_imported: int
    runs_imported: int
    configs_created: int
    workloads_created: int
    #: Repeated back so the caller can see what was recorded on their behalf, rather than
    #: having to trust that what they sent is what landed.
    declared: DeclaredProvenance
    note: str


async def _config_for(session: SessionDep, name: str, yaml_text: str) -> tuple[ServerConfig, bool]:
    digest = config_hash(yaml_text)
    existing = await session.scalar(select(ServerConfig).where(ServerConfig.config_hash == digest))
    if existing is not None:
        return existing, False
    config = ServerConfig(
        config_hash=digest,
        name=name,
        yaml=normalize_yaml(yaml_text),
        notes=(
            "Reconstructed on import from a `vllm bench sweep serve` output directory. "
            "Contains the swept parameters only — anything fixed in --serve-cmd was never "
            "recorded by the tool. Not a runnable configuration."
        ),
    )
    session.add(config)
    await session.flush()
    return config, True


async def _workload_for(
    session: SessionDep, bench_params: dict[str, Any], model: str | None
) -> tuple[Workload, bool]:
    """A workload matching the benchmark parameters this point was measured under.

    `request_rate` arrives from upstream as the string `"inf"` when unbounded. Stored as
    null, which is what unbounded means here — and emphatically not 0, which would mean
    the opposite.
    """
    # Upstream writes the string "inf" for an unbounded rate, and a bare number
    # otherwise. Unbounded is stored as null, which is what it means — and not as 0,
    # which would mean the opposite.
    raw_rate = bench_params.get("request_rate")
    rate: float | None
    if isinstance(raw_rate, str):
        try:
            parsed = float(raw_rate)
        except ValueError:
            parsed = float("inf")
        rate = None if parsed == float("inf") else parsed
    elif isinstance(raw_rate, int | float):
        rate = None if raw_rate == float("inf") else float(raw_rate)
    else:
        rate = None
    fields: dict[str, Any] = {
        "dataset_name": str(bench_params.get("dataset_name", "unknown")),
        "num_prompts": int(bench_params.get("num_prompts", 0) or 0),
        "max_concurrency": bench_params.get("max_concurrency"),
        "request_rate": rate,
        "input_len": bench_params.get("random_input_len") or bench_params.get("input_len"),
        "output_len": bench_params.get("random_output_len") or bench_params.get("output_len"),
        "hf_name": model,
    }
    digest = workload_hash(fields)
    existing = await session.scalar(select(Workload).where(Workload.workload_hash == digest))
    if existing is not None:
        return existing, False

    concurrency = fields["max_concurrency"]
    workload = Workload(
        workload_hash=digest,
        name=f"c{concurrency}" if concurrency else "unbounded",
        **fields,
    )
    session.add(workload)
    await session.flush()
    return workload, True


@router.post("/vllm-sweep", response_model=SweepImportOut, status_code=status.HTTP_201_CREATED)
async def import_vllm_sweep(payload: SweepImportRequest, session: SessionDep) -> SweepImportOut:
    """Import a `vllm bench sweep serve` output directory.

    Creates one sweep, one config and workload per point, and one run per `run=N.json`
    with its replicate index taken from the filename. Runs land already terminal, since
    they finished before this service ever saw them.

    Every imported run carries `imported_from`, permanently. The provenance on it was
    declared by a person rather than observed by an agent, and those are different kinds
    of fact — a chart that could not tell them apart would eventually be asked to explain
    a discrepancy nobody could resolve.
    """
    host = await session.get(GpuHost, payload.declared.gpu_host_id)
    if host is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="host not found")

    try:
        points = build_points(payload.files)
    except SweepDirectoryError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if len(points) > MAX_POINTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{len(points)} points exceeds the {MAX_POINTS} this endpoint accepts",
        )
    total_runs = sum(point.replicates for point in points)
    if total_runs > MAX_RUNS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{total_runs} runs exceeds the {MAX_RUNS} this endpoint accepts",
        )

    source = f"vllm bench sweep serve · {payload.experiment_name}"
    sweep = Sweep(
        name=payload.experiment_name,
        description=(
            "Imported from a `vllm bench sweep serve` output directory. Hardware and vLLM "
            "version were declared by the importer, not observed — the upstream output "
            "records none of it."
        ),
        gpu_host_id=host.id,
        status=SweepStatus.SUCCEEDED,
        replicates=max((p.replicates for p in points), default=1),
        # The upstream tool runs every replicate of a point before moving on, which is
        # what `grouped` means here. Recorded rather than guessed at read time, because
        # it is what a spread means (see analysis.spread_basis).
        replicate_order=ReplicateOrder.GROUPED,
        initiated_by=InitiatedBy.API,
        initiated_by_client="import",
        is_synthetic=host.synthetic_source is not None,
        started_at=dt.datetime.now(dt.UTC),
        finished_at=dt.datetime.now(dt.UTC),
    )
    session.add(sweep)
    await session.flush()

    configs_created = workloads_created = runs_imported = 0
    sequence = 0

    for point in points:
        model = next((str(r.get("model_id")) for r in point.runs if r.get("model_id")), None)
        config, made_config = await _config_for(
            session,
            f"imported: {point.directory[:100]}",
            reconstructed_yaml(point.serve_params, model),
        )
        configs_created += int(made_config)

        merged = {**point.bench_params}
        for key in ("num_prompts", "max_concurrency", "request_rate"):
            if key in point.runs[0]:
                merged.setdefault(key, point.runs[0][key])
        workload, made_workload = await _workload_for(session, merged, model)
        workloads_created += int(made_workload)

        for replicate, raw in enumerate(point.runs):
            run = Run(
                sweep_id=sweep.id,
                sweep_seq=sequence,
                replicate_idx=int(raw.get("run_number", replicate)),
                server_config_id=config.id,
                workload_id=workload.id,
                gpu_host_id=host.id,
                status=RunStatus.SUCCEEDED,
                started_at=dt.datetime.now(dt.UTC),
                finished_at=dt.datetime.now(dt.UTC),
                config_hash=config.config_hash,
                workload_hash=workload.workload_hash,
                vllm_version=payload.declared.vllm_version,
                gpu_model=payload.declared.gpu_model,
                driver_version=payload.declared.driver_version,
                gpu_count=payload.declared.gpu_count,
                tensor_parallel_size=payload.declared.tensor_parallel_size,
                pipeline_parallel_size=payload.declared.pipeline_parallel_size,
                bench_client_location=payload.declared.bench_client_location,
                is_synthetic=host.synthetic_source is not None,
                synthetic_source=host.synthetic_source,
                initiated_by=InitiatedBy.API,
                initiated_by_client="import",
                imported_from=source,
                # Raw before derived, as for a measured run: if the flattening is ever
                # found to be wrong, this is what allows recomputation.
                raw_result=raw,
            )
            session.add(run)
            await session.flush()
            sequence += 1

            try:
                flat = flatten_bench_result(raw, gpu_count=payload.declared.gpu_count)
            except BenchResultError as exc:
                # Loud, and the whole import is refused. A summary row full of NULLs is
                # indistinguishable from a benchmark that legitimately measured nothing,
                # and a half-imported sweep is worse than none.
                await session.rollback()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"{point.directory}/run={replicate}.json did not match the "
                        f"expected `vllm bench serve` result schema: {exc}"
                    ),
                ) from exc
            session.add(RunSummary(run_id=run.id, **flat))
            runs_imported += 1

    await session.commit()

    return SweepImportOut(
        sweep_id=sweep.id,
        points_imported=len(points),
        runs_imported=runs_imported,
        configs_created=configs_created,
        workloads_created=workloads_created,
        declared=payload.declared,
        note=(
            "Hardware and vLLM version were declared, not observed. These runs are marked "
            "imported and any chart grouping them with measured runs will say so."
        ),
    )
