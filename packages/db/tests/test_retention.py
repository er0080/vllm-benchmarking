"""Retention: what it deletes, and — mostly — what it must not.

This is the only machinery in the repository whose job is to destroy data, in a project
whose stated priority is that recorded measurements stay correct. So the balance of these
tests is deliberately lopsided: a few confirm that pruning works, and the rest confirm it
cannot reach anything it should not.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_db.enums import FailureKind, RunStatus
from vllmbench_db.models import EngineSample, GpuSample, Run, RunSummary, RunTelemetryPruned
from vllmbench_db.retention import (
    PROTECTED,
    PRUNABLE,
    prune_telemetry,
    storage_report,
)
from vllmbench_db.session import create_engine, create_session_factory
from vllmbench_db.testing import reset_database, test_database_url

pytestmark = pytest.mark.integration


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Overrides the package fixture, which isolates tests by never committing.

    That works for the invariant tests, which only need the constraint to fire. Retention
    commits by design — it is housekeeping, not part of a caller's transaction — so
    isolation here has to come from emptying the database instead.
    """
    engine = create_engine(test_database_url())
    factory = create_session_factory(engine)
    await reset_database(engine)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _run_with_telemetry(
    session: AsyncSession,
    run_factory,
    *,
    finished_days_ago: float,
    status: RunStatus = RunStatus.SUCCEEDED,
    samples: int = 5,
) -> Run:
    finished = dt.datetime.now(dt.UTC) - dt.timedelta(days=finished_days_ago)
    run = await run_factory(
        status=status,
        started_at=finished - dt.timedelta(seconds=60),
        finished_at=finished,
        **({"failure_kind": FailureKind.INTERNAL} if status is RunStatus.FAILED else {}),
    )
    session.add(RunSummary(run_id=run.id, successful_requests=8, benchmark_duration_sec=60.0))
    for i in range(samples):
        at = finished - dt.timedelta(seconds=samples - i)
        session.add(EngineSample(run_id=run.id, sampled_at=at, num_requests_running=4))
        for gpu in (0, 1):
            session.add(
                GpuSample(run_id=run.id, gpu_index=gpu, sampled_at=at, sm_utilization_pct=90.0)
            )
    await session.commit()
    return run


class TestWhatItRefusesToTouch:
    def test_nothing_prunable_is_also_protected(self) -> None:
        """The invariant the whole module rests on, checked directly.

        `prune_telemetry` asserts this at runtime as well. Belt and braces on purpose:
        the test catches it in CI, and the runtime check catches it in the one deployment
        where someone shipped past the test.
        """
        prunable = {model.__tablename__ for model in PRUNABLE}
        assert prunable.isdisjoint(PROTECTED)

    def test_the_results_themselves_are_protected(self) -> None:
        """Named explicitly rather than derived, so removing one is a visible diff.

        `run` and `run_summary` are the measurement. Nothing that reclaims disk may be one
        refactor away from deleting them.
        """
        assert "run" in PROTECTED
        assert "run_summary" in PROTECTED

    def test_raw_payloads_are_never_pruned(self, session: AsyncSession, run_factory) -> None:
        """CLAUDE.md: "Never discard the original."

        `raw_result` is the second-largest per-run payload and the obvious next thing to
        reclaim. It is also the only way to fix a flattening mistake without re-running
        weeks of GPU time — about a kilobyte against that. Asserted as an absence, since
        the failure mode is somebody adding it.
        """
        prunable_columns = {
            (model.__tablename__, column.name)
            for model in PRUNABLE
            for column in model.__table__.columns
        }
        assert ("run", "raw_result") not in prunable_columns

    async def test_a_pass_leaves_every_measurement_intact(
        self, session: AsyncSession, run_factory
    ) -> None:
        old = await _run_with_telemetry(session, run_factory, finished_days_ago=400)
        old_id = old.id
        runs_before = await session.scalar(select(func.count()).select_from(Run))
        summaries_before = await session.scalar(select(func.count()).select_from(RunSummary))

        await prune_telemetry(session, older_than_days=30)

        assert await session.scalar(select(func.count()).select_from(Run)) == runs_before
        assert (
            await session.scalar(select(func.count()).select_from(RunSummary)) == summaries_before
        )
        # And the run itself still says what produced it.
        session.expire_all()
        kept = await session.get(Run, old_id)
        assert kept is not None
        assert kept.config_hash and kept.workload_hash


class TestWhatItPrunes:
    async def test_telemetry_older_than_the_horizon_goes(
        self, session: AsyncSession, run_factory
    ) -> None:
        run = await _run_with_telemetry(session, run_factory, finished_days_ago=90)

        result = await prune_telemetry(session, older_than_days=30)

        assert result.runs == 1
        assert result.engine_samples == 5
        assert result.gpu_samples == 10  # two devices
        assert not await session.scalar(
            select(func.count()).select_from(GpuSample).where(GpuSample.run_id == run.id)
        )

    async def test_telemetry_inside_the_horizon_stays(
        self, session: AsyncSession, run_factory
    ) -> None:
        run = await _run_with_telemetry(session, run_factory, finished_days_ago=5)

        result = await prune_telemetry(session, older_than_days=30)

        assert result.runs == 0
        assert (
            await session.scalar(
                select(func.count()).select_from(GpuSample).where(GpuSample.run_id == run.id)
            )
            == 10
        )

    async def test_a_run_still_in_flight_is_never_pruned(
        self, session: AsyncSession, run_factory
    ) -> None:
        """A run with no `finished_at` has no meaningful age.

        Deleting samples out from under a benchmark in progress would corrupt the
        timeline of a measurement being taken right now — and an unfinished run whose
        `queued_at` is old is exactly the shape a stuck sweep leaves behind.
        """
        run = await run_factory(
            status=RunStatus.BENCHMARKING,
            started_at=dt.datetime.now(dt.UTC) - dt.timedelta(days=400),
            finished_at=None,
        )
        session.add(
            GpuSample(
                run_id=run.id,
                gpu_index=0,
                sampled_at=dt.datetime.now(dt.UTC),
                sm_utilization_pct=50.0,
            )
        )
        await session.commit()

        result = await prune_telemetry(session, older_than_days=1)

        assert result.runs == 0
        assert await session.scalar(
            select(func.count()).select_from(GpuSample).where(GpuSample.run_id == run.id)
        )

    async def test_failed_and_cancelled_runs_are_pruned_too(
        self, session: AsyncSession, run_factory
    ) -> None:
        """Their telemetry ages the same way. The run and its reason are kept."""
        for status in (RunStatus.FAILED, RunStatus.CANCELLED):
            await _run_with_telemetry(session, run_factory, finished_days_ago=90, status=status)

        result = await prune_telemetry(session, older_than_days=30)

        assert result.runs == 2

    async def test_a_second_pass_finds_nothing(self, session: AsyncSession, run_factory) -> None:
        """Idempotence, which is what makes a daily schedule cheap.

        Without the pruned-marker join, every pass would re-scan and re-"delete" the same
        already-empty runs forever, and the cost would grow with history.
        """
        await _run_with_telemetry(session, run_factory, finished_days_ago=90)

        first = await prune_telemetry(session, older_than_days=30)
        second = await prune_telemetry(session, older_than_days=30)

        assert first.runs == 1
        assert second.runs == 0

    async def test_a_dry_run_deletes_nothing(self, session: AsyncSession, run_factory) -> None:
        run = await _run_with_telemetry(session, run_factory, finished_days_ago=90)

        result = await prune_telemetry(session, older_than_days=30, dry_run=True)

        assert result.dry_run and result.runs == 1 and result.gpu_samples == 10
        assert (
            await session.scalar(
                select(func.count()).select_from(GpuSample).where(GpuSample.run_id == run.id)
            )
            == 10
        )
        assert not await session.scalar(select(func.count()).select_from(RunTelemetryPruned))

    async def test_a_horizon_of_zero_is_refused(self, session: AsyncSession) -> None:
        """0 means "retention is off", never "delete everything".

        The two readings differ by the entire telemetry history, so the ambiguous value
        raises instead of picking one.
        """
        with pytest.raises(ValueError, match="retention is disabled"):
            await prune_telemetry(session, older_than_days=0)


class TestPrunedIsNotMissing:
    async def test_pruning_is_recorded_against_the_run(
        self, session: AsyncSession, run_factory
    ) -> None:
        """An empty timeline needs an explanation, or it reads as a sampling bug.

        A run whose telemetry was deleted by policy and a run whose telemetry never
        recorded look identical on a detail page, and they call for opposite responses:
        one is the policy working, the other is diagnostic data being lost silently.
        """
        run = await _run_with_telemetry(session, run_factory, finished_days_ago=90)

        await prune_telemetry(session, older_than_days=30)

        record = await session.get(RunTelemetryPruned, run.id)
        assert record is not None
        assert record.horizon_days == 30
        assert record.pruned_at is not None

    async def test_a_run_that_never_had_telemetry_is_not_marked(
        self, session: AsyncSession, run_factory
    ) -> None:
        """The distinction only means something if it is not applied to everything.

        A run old enough to prune but with no samples still gets a marker — it went
        through the pass — while one inside the horizon does not. What must never happen
        is a marker on a run nothing was done to.
        """
        recent = await _run_with_telemetry(session, run_factory, finished_days_ago=1)

        await prune_telemetry(session, older_than_days=30)

        assert await session.get(RunTelemetryPruned, recent.id) is None

    async def test_the_marker_goes_with_the_run_if_the_run_is_ever_deleted(
        self, session: AsyncSession, run_factory
    ) -> None:
        """ON DELETE CASCADE, so the marker cannot outlive what it describes."""
        run = await _run_with_telemetry(session, run_factory, finished_days_ago=90)
        await prune_telemetry(session, older_than_days=30)

        await session.delete(await session.get(Run, run.id))
        await session.commit()

        assert await session.get(RunTelemetryPruned, run.id) is None


class TestStorageReport:
    async def test_it_names_which_tables_are_reclaimable(
        self, session: AsyncSession, run_factory
    ) -> None:
        """The question an operator actually has about a large table.

        "gpu_sample is 40 GB" is only actionable alongside "and it may be deleted";
        "run_summary is 40 GB" alongside "and it may not" is a different conversation.
        """
        await _run_with_telemetry(session, run_factory, finished_days_ago=1)

        report = await storage_report(session)
        by_name = {t.name: t for t in report.tables}

        assert by_name["gpu_sample"].protected is False
        assert by_name["run_summary"].protected is True
        assert report.gpu_samples == 10
        assert report.engine_samples == 5

    async def test_growth_is_measured_rather_than_assumed(
        self, session: AsyncSession, run_factory
    ) -> None:
        """From this database's own history, because the constant would be wrong.

        A two-GPU host and an eight-GPU host produce very different numbers from the same
        code, and this is the figure someone would size a disk against.
        """
        await _run_with_telemetry(session, run_factory, finished_days_ago=1, samples=50)

        report = await storage_report(session)

        assert report.bytes_per_run_hour is not None
        assert report.bytes_per_run_hour > 0

    async def test_it_says_nothing_rather_than_guessing_on_an_empty_database(
        self, session: AsyncSession
    ) -> None:
        report = await storage_report(session)
        assert report.bytes_per_run_hour is None
