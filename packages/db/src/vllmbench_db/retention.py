"""What may be deleted to reclaim disk, and what may never be.

Lives in ``db`` rather than ``api`` because two services need it and neither should
depend on the other: the API reports and exposes it, the orchestrator applies it on a
schedule. Putting it in the API would have made the orchestrator import FastAPI, which
inverts the reason those are separate services at all.

The second half is the point. This repository produces measurements, and a retention
policy is the one piece of machinery whose entire job is to destroy data — so the list of
what it must never touch is written down first, asserted by a test, and checked at
runtime before anything is deleted.

The numbers below are measured, not estimated. On the reference host — 2x RTX 3090, 1s
sampling — a run costs roughly:

    gpu_sample      225 bytes/row * (duration_seconds * gpu_count)
    engine_sample   218 bytes/row * duration_seconds
    run + summary   ~2 kB, once
    raw_result      ~1.1 kB, once

Telemetry is therefore about 97% of what a run occupies, and it scales with duration and
device count rather than with run count. A one-hour run on eight GPUs is about 7 MB; a
month of continuous benchmarking is a few gigabytes. That is not an emergency, which is
exactly why the default horizon is "keep everything" — but it is unbounded, and the
person who eventually needs to bound it should not have to invent a policy under
pressure.

**Telemetry is the only thing this deletes.** It is diagnostic: it explains *why* a
configuration won, and that question has a shelf life in a way the measurement itself
does not. Everything else is either the measurement or the means of recomputing it.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_db.enums import RunStatus
from vllmbench_db.models import EngineSample, GpuSample, Run, RunTelemetryPruned

log = logging.getLogger(__name__)

#: Tables retention may never delete from, and why. Consulted at runtime, not merely
#: documentation: :func:`prune_telemetry` refuses to run if the tables it is about to
#: touch are not the ones it expects.
#:
#: Two different reasons appear here and they are worth keeping distinct. Some rows *are*
#: the measurement. Others are the only way to recover from having flattened it wrong.
PROTECTED: dict[str, str] = {
    "run": "the run carries the provenance; without it a result cannot say what produced it",
    "run_summary": "the flattened metrics — this is the result",
    "sweep": "a sweep's identity is what makes its runs comparable with each other",
    "server_config": "content-addressed, and referenced by every run that claims it",
    "workload": "same: a run's workload_hash must always resolve",
    "gpu_host": "provenance under invariant 6",
    "mcp_write_audit": "the record of what an agent asked for, including what was refused",
    "config_justification": "the evidence link that makes a configuration defensible later",
    "saved_view": "a stored query, owned by a person",
}

#: The only tables that may be pruned, and the column each is aged by.
#:
#: `raw_result` is deliberately absent even though it is the second-largest per-run
#: payload. CLAUDE.md: "If we later discover we flattened something wrong, the raw record
#: lets us recompute. Never discard the original." Deleting it would trade 1.1 kB for the
#: ability to fix a mistake without re-running weeks of GPU time.
PRUNABLE: tuple[type[EngineSample] | type[GpuSample], ...] = (EngineSample, GpuSample)


@dataclass(frozen=True, slots=True)
class TableUsage:
    name: str
    rows: int
    total_bytes: int
    protected: bool


@dataclass(frozen=True, slots=True)
class StorageReport:
    """What is on disk, and what a further hour of benchmarking would add.

    The growth figure is derived from this database's own rows rather than from the
    constants in this module's docstring — a host with more GPUs or a different sampling
    interval has a different answer, and an estimate that ignores that is worse than
    none.
    """

    tables: list[TableUsage]
    total_bytes: int
    engine_samples: int
    gpu_samples: int
    oldest_sample_at: dt.datetime | None
    bytes_per_run_hour: int | None
    runs_with_telemetry: int
    runs_pruned: int


@dataclass(frozen=True, slots=True)
class PruneResult:
    """What a pass did, or — in a dry run — what it would do."""

    cutoff: dt.datetime
    runs: int
    engine_samples: int
    gpu_samples: int
    bytes_reclaimed: int
    dry_run: bool


async def storage_report(session: AsyncSession) -> StorageReport:
    """Per-table sizes and telemetry counts, read from Postgres itself.

    ``pg_total_relation_size`` rather than a row-count estimate: indexes and TOAST are
    most of what a telemetry table costs, and a figure that omits them would understate
    the thing the report exists to make visible.
    """
    rows = (
        await session.execute(
            text(
                """
                SELECT relname, n_live_tup, pg_total_relation_size(relid)
                FROM pg_stat_user_tables
                WHERE relname <> 'alembic_version'
                ORDER BY pg_total_relation_size(relid) DESC
                """
            )
        )
    ).all()

    tables = [
        TableUsage(name=name, rows=int(live), total_bytes=int(size), protected=name in PROTECTED)
        for name, live, size in rows
    ]

    engine_samples = await session.scalar(select(func.count()).select_from(EngineSample)) or 0
    gpu_samples = await session.scalar(select(func.count()).select_from(GpuSample)) or 0
    oldest = await session.scalar(select(func.min(GpuSample.sampled_at)))
    pruned = await session.scalar(select(func.count()).select_from(RunTelemetryPruned)) or 0

    return StorageReport(
        tables=tables,
        total_bytes=sum(t.total_bytes for t in tables),
        engine_samples=engine_samples,
        gpu_samples=gpu_samples,
        oldest_sample_at=oldest,
        bytes_per_run_hour=await _bytes_per_run_hour(session, engine_samples, gpu_samples),
        runs_with_telemetry=await _runs_with_telemetry(session),
        runs_pruned=pruned,
    )


async def _runs_with_telemetry(session: AsyncSession) -> int:
    return (await session.scalar(select(func.count(func.distinct(GpuSample.run_id))))) or 0


async def _bytes_per_run_hour(
    session: AsyncSession, engine_samples: int, gpu_samples: int
) -> int | None:
    """Telemetry bytes per hour of benchmarking, from this database's own history.

    Returns ``None`` until there is something to measure. A made-up number here would be
    worse than silence: it is the figure an operator would size a disk against.
    """
    seconds = await session.scalar(
        select(
            func.sum(func.extract("epoch", Run.finished_at) - func.extract("epoch", Run.started_at))
        ).where(
            Run.status == RunStatus.SUCCEEDED,
            Run.started_at.is_not(None),
            Run.finished_at.is_not(None),
        )
    )
    if not seconds or seconds <= 0:
        return None

    total = await session.scalar(
        text(
            "SELECT pg_total_relation_size('engine_sample') + pg_total_relation_size('gpu_sample')"
        )
    )
    if not total or not (engine_samples + gpu_samples):
        return None

    return int(float(total) / float(seconds) * 3600)


async def prune_telemetry(
    session: AsyncSession, *, older_than_days: int, dry_run: bool = False
) -> PruneResult:
    """Delete telemetry for runs that finished before the horizon.

    Only terminal runs are considered — a run still in flight has no meaningful age, and
    deleting samples from underneath a benchmark in progress would corrupt the timeline
    of a measurement being taken right now.

    Each affected run gets a ``run_telemetry_pruned`` row. That is not bookkeeping: a run
    detail page with an empty timeline is otherwise indistinguishable from a run whose
    telemetry failed to record, and those call for opposite responses. The record goes in
    its own table rather than on the run, because a terminal run is immutable and enforced
    so by a trigger — this is data *about* a run, not a correction to it.

    Deleting samples does not contradict "time-series tables are append-only". Append-only
    means never rewritten in place, and that is what protects the measurement: a rewritten
    sample makes a chart lie about what was observed. A sample removed under a stated
    horizon, recorded per run, cannot — the chart says "pruned" instead of drawing
    something false.
    """
    if older_than_days < 1:
        raise ValueError("older_than_days must be at least 1; 0 means retention is disabled")

    _assert_only_prunable()

    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=older_than_days)

    # Runs old enough to prune that still have telemetry. The left join is what makes a
    # second pass cheap and idempotent rather than re-deleting nothing over and over.
    candidates = (
        await session.execute(
            select(Run.id)
            .outerjoin(RunTelemetryPruned, RunTelemetryPruned.run_id == Run.id)
            .where(
                Run.status.in_([RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED]),
                Run.finished_at.is_not(None),
                Run.finished_at < cutoff,
                RunTelemetryPruned.run_id.is_(None),
            )
        )
    ).scalars()
    run_ids = list(candidates)

    if not run_ids:
        return PruneResult(cutoff, 0, 0, 0, 0, dry_run)

    engine_count = (
        await session.scalar(
            select(func.count()).select_from(EngineSample).where(EngineSample.run_id.in_(run_ids))
        )
        or 0
    )
    gpu_count = (
        await session.scalar(
            select(func.count()).select_from(GpuSample).where(GpuSample.run_id.in_(run_ids))
        )
        or 0
    )
    # Bytes-per-row from the live tables rather than the constants in the docstring: a
    # different Postgres version or fill factor gives a different answer, and this figure
    # is only useful if it describes this database.
    reclaimed = await _estimated_bytes(session, engine_count, gpu_count)

    if dry_run:
        return PruneResult(cutoff, len(run_ids), engine_count, gpu_count, reclaimed, True)

    await session.execute(delete(EngineSample).where(EngineSample.run_id.in_(run_ids)))
    await session.execute(delete(GpuSample).where(GpuSample.run_id.in_(run_ids)))

    now = dt.datetime.now(dt.UTC)
    session.add_all(
        [
            RunTelemetryPruned(
                run_id=run_id,
                pruned_at=now,
                horizon_days=older_than_days,
            )
            for run_id in run_ids
        ]
    )
    await session.commit()

    log.info(
        "pruned telemetry for %d run(s) finished before %s: %d engine + %d gpu samples, ~%d bytes",
        len(run_ids),
        cutoff.isoformat(),
        engine_count,
        gpu_count,
        reclaimed,
        extra={"retention_cutoff": cutoff.isoformat(), "runs_pruned": len(run_ids)},
    )
    return PruneResult(cutoff, len(run_ids), engine_count, gpu_count, reclaimed, False)


async def _estimated_bytes(session: AsyncSession, engine_rows: int, gpu_rows: int) -> int:
    """Bytes per row from the live tables, times the rows about to go.

    Measured rather than taken from the constants in this module's docstring: a different
    Postgres version, fill factor or index set gives a different answer, and this number
    is only worth reporting if it describes this database.

    The table name reaches SQL as a bind parameter and the row count goes through the
    ORM, so no SQL is built by interpolation here.
    """
    total = 0
    for model, rows in ((EngineSample, engine_rows), (GpuSample, gpu_rows)):
        if not rows:
            continue
        size = await session.scalar(
            text("SELECT pg_total_relation_size(:table)"), {"table": model.__tablename__}
        )
        live = await session.scalar(select(func.count()).select_from(model))
        if size and live:
            total += int(float(size) / float(live) * rows)
    return total


def _assert_only_prunable() -> None:
    """Refuse to run if this module has grown the ability to delete something protected.

    A runtime check rather than only a test, because the cost of being wrong is
    asymmetric: a test catches it in CI, and this catches it in the one deployment where
    someone shipped past the test.
    """
    names = {model.__tablename__ for model in PRUNABLE}
    overlap = names & set(PROTECTED)
    if overlap:
        raise RuntimeError(
            f"retention would reach protected table(s) {sorted(overlap)}; refusing. "
            "See PROTECTED in retention.py for why each one is protected."
        )
