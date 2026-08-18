"""What the database is holding, and what a retention pass would remove.

A read surface plus one deliberately awkward write. You cannot manage disk you cannot
see, and until now nothing here reported how much a run costs or where it goes — so the
first question an operator asks ("is this growing, and how fast?") had no answer short of
`psql`.

The pruning endpoint requires `confirm=true` and defaults to a dry run. This is the only
route in the API whose purpose is to destroy data, and it should be harder to invoke than
the ones that create it.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from vllmbench_api.deps import SessionDep
from vllmbench_api.schemas import PruneOut, StorageOut, TableUsageOut
from vllmbench_db.retention import PROTECTED, prune_telemetry, storage_report

router = APIRouter(prefix="/api/storage", tags=["storage"])


@router.get("", response_model=StorageOut)
async def get_storage(session: SessionDep) -> StorageOut:
    """Per-table sizes, telemetry counts, and measured growth per hour of benchmarking.

    The growth figure comes from this database's own history rather than a constant. A
    host with eight GPUs and one with two produce very different numbers from the same
    code, and an estimate that ignores that is worse than none — it is the figure someone
    would size a disk against.
    """
    report = await storage_report(session)
    return StorageOut(
        total_bytes=report.total_bytes,
        tables=[
            TableUsageOut(
                name=t.name,
                rows=t.rows,
                total_bytes=t.total_bytes,
                protected=t.protected,
                protected_because=PROTECTED.get(t.name),
            )
            for t in report.tables
        ],
        engine_samples=report.engine_samples,
        gpu_samples=report.gpu_samples,
        oldest_sample_at=report.oldest_sample_at,
        bytes_per_run_hour=report.bytes_per_run_hour,
        runs_with_telemetry=report.runs_with_telemetry,
        runs_pruned=report.runs_pruned,
    )


@router.post("/prune", response_model=PruneOut)
async def prune(
    session: SessionDep,
    older_than_days: Annotated[int, Query(ge=1, le=3650)],
    confirm: bool = False,
) -> PruneOut:
    """Delete telemetry for runs that finished more than ``older_than_days`` ago.

    Dry run unless ``confirm=true``, so the default answer to "what would this remove" is
    a number rather than a removal. Telemetry is the only thing this can delete —
    measurements, summaries, raw payloads and provenance are protected in
    `retention.PROTECTED`, and the function refuses to run if that ever stops being true.

    Each affected run is recorded as pruned. An empty timeline then reads as "the policy
    removed this" rather than "sampling failed", which are different problems.
    """
    if older_than_days < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="older_than_days must be at least 1",
        )

    result = await prune_telemetry(session, older_than_days=older_than_days, dry_run=not confirm)
    return PruneOut(
        cutoff=result.cutoff,
        runs=result.runs,
        engine_samples=result.engine_samples,
        gpu_samples=result.gpu_samples,
        bytes_reclaimed=result.bytes_reclaimed,
        dry_run=result.dry_run,
    )
