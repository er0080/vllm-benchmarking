"""The schema's guarantees, tested against a real PostgreSQL.

Each test here corresponds to a rule in CLAUDE.md that would otherwise be enforced only
by everyone remembering it. These are the tests worth keeping green at any cost: they are
what stands between a bug and a permanently corrupted measurement.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_db.enums import FailureKind, RunStatus, SyntheticSource
from vllmbench_db.models import EngineSample, GpuSample

pytestmark = pytest.mark.integration


class TestRunImmutability:
    """CLAUDE.md: 'Runs are immutable once terminal.'"""

    @pytest.mark.parametrize("status", [RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED])
    async def test_terminal_run_cannot_be_updated(
        self, session: AsyncSession, run_factory, status: RunStatus
    ) -> None:
        run = await run_factory(status=status)
        with pytest.raises(DBAPIError, match="terminal"):
            await session.execute(
                text("UPDATE run SET error = 'tampered' WHERE id = :id"), {"id": run.id}
            )
        await session.rollback()

    @pytest.mark.parametrize(
        "status", [RunStatus.QUEUED, RunStatus.STARTING, RunStatus.BENCHMARKING]
    )
    async def test_in_flight_run_can_still_be_updated(
        self, session: AsyncSession, run_factory, status: RunStatus
    ) -> None:
        # The trigger must not be so blunt that it blocks a run from reaching a terminal
        # state in the first place.
        run = await run_factory(status=status)
        await session.execute(
            text("UPDATE run SET status = 'succeeded' WHERE id = :id"), {"id": run.id}
        )

    async def test_a_run_can_be_moved_into_a_terminal_state_exactly_once(
        self, session: AsyncSession, run_factory
    ) -> None:
        run = await run_factory(status=RunStatus.BENCHMARKING)
        await session.execute(
            text("UPDATE run SET status = 'succeeded' WHERE id = :id"), {"id": run.id}
        )
        with pytest.raises(DBAPIError, match="terminal"):
            await session.execute(
                text("UPDATE run SET status = 'failed' WHERE id = :id"), {"id": run.id}
            )
        await session.rollback()


class TestTelemetryIsAppendOnly:
    """CLAUDE.md: time-series tables 'are never updated in place'."""

    async def test_engine_sample_cannot_be_updated(
        self, session: AsyncSession, run_factory
    ) -> None:
        run = await run_factory()
        session.add(
            EngineSample(
                run_id=run.id, sampled_at=dt.datetime.now(dt.UTC), kv_cache_usage_fraction=0.5
            )
        )
        await session.flush()

        with pytest.raises(DBAPIError, match="append-only"):
            await session.execute(
                text("UPDATE engine_sample SET kv_cache_usage_fraction = 0.9 WHERE run_id = :id"),
                {"id": run.id},
            )
        await session.rollback()

    async def test_gpu_sample_cannot_be_updated(self, session: AsyncSession, run_factory) -> None:
        run = await run_factory()
        session.add(
            GpuSample(
                run_id=run.id,
                gpu_index=0,
                sampled_at=dt.datetime.now(dt.UTC),
                sm_utilization_pct=80.0,
            )
        )
        await session.flush()

        with pytest.raises(DBAPIError, match="append-only"):
            await session.execute(
                text("UPDATE gpu_sample SET sm_utilization_pct = 10 WHERE run_id = :id"),
                {"id": run.id},
            )
        await session.rollback()


class TestPerDeviceTelemetry:
    """Invariant 8: per-device rows, never aggregated at write time."""

    async def test_one_row_per_device_per_timestamp(
        self, session: AsyncSession, run_factory
    ) -> None:
        run = await run_factory(gpu_count=4, tensor_parallel_size=4)
        now = dt.datetime.now(dt.UTC)

        # The imbalance a host-level average would erase: one device loafing while three
        # saturate. This shape must be representable.
        for index, utilization in enumerate([60.0, 95.0, 95.0, 95.0]):
            session.add(
                GpuSample(
                    run_id=run.id,
                    gpu_index=index,
                    sampled_at=now,
                    sm_utilization_pct=utilization,
                )
            )
        await session.flush()

        result = await session.execute(
            text(
                "SELECT gpu_index, sm_utilization_pct FROM gpu_sample"
                " WHERE run_id = :id ORDER BY gpu_index"
            ),
            {"id": run.id},
        )
        rows = result.all()
        assert [r[0] for r in rows] == [0, 1, 2, 3]
        assert rows[0][1] == pytest.approx(60.0)


class TestSyntheticQuarantine:
    """Invariant 7: a synthetic run must name its source, and a real run must not have one."""

    async def test_synthetic_run_without_a_source_is_rejected(self, run_factory) -> None:
        with pytest.raises(IntegrityError, match="synthetic_source_matches_flag"):
            await run_factory(is_synthetic=True, synthetic_source=None)

    async def test_real_run_with_a_source_is_rejected(self, run_factory) -> None:
        # The dangerous direction: a run that looks real but was produced by a fake.
        with pytest.raises(IntegrityError, match="synthetic_source_matches_flag"):
            await run_factory(is_synthetic=False, synthetic_source=SyntheticSource.MOCK_AGENT)

    async def test_consistent_synthetic_run_is_accepted(self, run_factory) -> None:
        run = await run_factory(is_synthetic=True, synthetic_source=SyntheticSource.MOCK_AGENT)
        assert run.is_synthetic is True

    async def test_consistent_real_run_is_accepted(self, run_factory) -> None:
        run = await run_factory(is_synthetic=False, synthetic_source=None)
        assert run.is_synthetic is False


class TestFailuresNameThemselves:
    """A failed run says what kind of failure it was.

    Not a style rule: the column exists so that a sweep with eleven failed points can be
    asked whether it hit one cause or eleven, and a NULL is a point that cannot join that
    answer. The orchestrator's floor is `internal`, so there is no path that fails
    without a kind — this is what keeps that true if someone adds one.
    """

    async def test_a_failed_run_without_a_kind_is_rejected(self, run_factory) -> None:
        with pytest.raises(IntegrityError, match="failed_run_names_its_failure"):
            await run_factory(status=RunStatus.FAILED, failure_kind=None)

    async def test_a_failed_run_with_a_kind_is_accepted(self, run_factory) -> None:
        run = await run_factory(
            status=RunStatus.FAILED, failure_kind=FailureKind.ENGINE_OUT_OF_MEMORY
        )
        assert run.failure_kind == FailureKind.ENGINE_OUT_OF_MEMORY

    async def test_an_unfamiliar_kind_is_recorded_rather_than_refused(self, run_factory) -> None:
        """Free text, not a native enum, and this is the reason.

        A newer agent naming a failure this build has never heard of must still be
        *recorded*. With a native enum the insert would fail instead — turning "I do not
        recognise this failure" into "the failure is lost", which is the one outcome
        worse than filing it under the wrong heading.
        """
        run = await run_factory(status=RunStatus.FAILED, failure_kind="engine_ate_the_cache")
        assert run.failure_kind == "engine_ate_the_cache"

    async def test_runs_that_did_not_fail_need_no_kind(self, run_factory) -> None:
        for status in (RunStatus.SUCCEEDED, RunStatus.CANCELLED, RunStatus.QUEUED):
            run = await run_factory(status=status, failure_kind=None)
            assert run.failure_kind is None


class TestContentAddressing:
    """CLAUDE.md: 'Configs are content-addressed.'"""

    async def test_config_hash_is_unique(self, session: AsyncSession, config) -> None:
        from vllmbench_db.models import ServerConfig

        session.add(
            ServerConfig(
                config_hash=config.config_hash, name="a different name", yaml="model: other\n"
            )
        )
        with pytest.raises(IntegrityError):
            await session.flush()
        await session.rollback()


class TestTopologyGuards:
    async def test_gpu_count_must_be_positive(self, run_factory) -> None:
        with pytest.raises(IntegrityError, match="gpu_count_positive"):
            await run_factory(gpu_count=0)

    async def test_tensor_parallel_size_must_be_positive(self, run_factory) -> None:
        with pytest.raises(IntegrityError, match="tp_positive"):
            await run_factory(tensor_parallel_size=0)


class TestResetHelper:
    """The reset every package's integration tests depend on.

    Tested because its failure mode is not its own: a table it misses leaves rows behind
    that break a different package's setup, and the error names a file that had nothing
    to do with the omission.
    """

    def test_delete_order_covers_every_mapped_table(self) -> None:
        from vllmbench_db.base import Base
        from vllmbench_db.testing import delete_order

        assert set(delete_order()) == set(Base.metadata.tables)

    def test_children_are_deleted_before_their_parents(self) -> None:
        from vllmbench_db.testing import delete_order

        order = delete_order()
        for child, parent in (
            ("run_summary", "run"),
            ("engine_sample", "run"),
            ("gpu_sample", "run"),
            ("run", "sweep"),
            ("run", "server_config"),
            ("sweep", "gpu_host"),
            ("gpu_device", "gpu_host"),
        ):
            assert order.index(child) < order.index(parent), f"{child} must precede {parent}"

    async def test_reset_empties_a_populated_database(self) -> None:
        """End to end against a real database, on its own engine.

        Its own engine rather than the shared fixture's: the reset commits, and the
        fixture's session deliberately rolls back so that tests do not see each other's
        rows. Reaching into that session's bind would mix the two isolation models.
        """
        import uuid

        from sqlalchemy import func, select

        from vllmbench_db.models import GpuHost
        from vllmbench_db.session import create_engine, create_session_factory
        from vllmbench_db.testing import reset_database, test_database_url

        engine = create_engine(test_database_url())
        try:
            factory = create_session_factory(engine)
            async with factory() as s:
                s.add(
                    GpuHost(
                        name=f"reset-{uuid.uuid4().hex[:8]}",
                        agent_url="http://10.0.0.9:9110",
                    )
                )
                await s.commit()

            await reset_database(engine)

            async with factory() as s:
                assert await s.scalar(select(func.count()).select_from(GpuHost)) == 0
        finally:
            await engine.dispose()
