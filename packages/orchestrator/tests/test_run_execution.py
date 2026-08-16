"""A run, end to end: queued → server started → benchmarked → persisted.

This is milestone 0.2.0's promise, exercised against a real database and the mock agent.
The failure paths get equal weight, because a run that fails must still end up in a
terminal state with a reason — a run stuck in `benchmarking` forever shows in the UI as
perpetually working and blocks whatever comes next.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import httpx
import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from vllmbench_db.enums import InitiatedBy, RunStatus
from vllmbench_db.models import GpuHost, Run, ServerConfig, Workload
from vllmbench_db.session import create_engine, create_session_factory
from vllmbench_mockagent.main import create_app as create_mock_app
from vllmbench_orchestrator.runner import claim_next_run, execute_run

pytestmark = pytest.mark.integration

TOKEN = "test-token-not-a-real-secret"
MOCK_URL = "http://mock-agent"

CONFIG_YAML = "model: facebook/opt-125m\ntensor_parallel_size: 2\n"


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://vllmbench:change-me@localhost:5432/vllmbench",
    )


@pytest.fixture(autouse=True)
def fast_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLMBENCH_MOCK_LOAD_SECONDS", "0.01")
    monkeypatch.setenv("VLLMBENCH_MOCK_BENCH_SECONDS", "0.01")


@pytest.fixture
def route_to_mock(monkeypatch: pytest.MonkeyPatch):
    """Point the agent client at an in-process mock, keeping the real client path."""

    def _install(app) -> None:
        transport = httpx.ASGITransport(app=app)
        real = httpx.AsyncClient

        def patched(*args: object, **kwargs: object) -> httpx.AsyncClient:
            if str(kwargs.get("base_url", "")).startswith(MOCK_URL):
                kwargs["transport"] = transport
            return real(*args, **kwargs)

        monkeypatch.setattr("vllmbench_protocol.client.httpx.AsyncClient", patched)

    return _install


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_engine(_database_url())
    factory = create_session_factory(engine)
    async with factory() as s:
        async with engine.begin() as connection:
            await connection.execute(text("DELETE FROM run_summary"))
            await connection.execute(text("DELETE FROM run"))
            await connection.execute(text("DELETE FROM gpu_device"))
            await connection.execute(text("DELETE FROM gpu_host"))
        yield s
    await engine.dispose()


async def _seed(session: AsyncSession, *, synthetic: str | None = "mock_agent") -> Run:
    host = GpuHost(
        name=f"host-{os.urandom(4).hex()}",
        agent_url=MOCK_URL,
        gpu_count=2,
        synthetic_source=synthetic,
    )
    config = ServerConfig(config_hash=os.urandom(32).hex(), name="cfg", yaml=CONFIG_YAML)
    workload = Workload(
        workload_hash=os.urandom(32).hex(),
        name="wl",
        dataset_name="random",
        num_prompts=64,
        max_concurrency=16,
        input_len=128,
        output_len=64,
    )
    session.add_all([host, config, workload])
    await session.flush()

    run = Run(
        gpu_host_id=host.id,
        server_config_id=config.id,
        workload_id=workload.id,
        status=RunStatus.QUEUED,
        config_hash=config.config_hash,
        workload_hash=workload.workload_hash,
        gpu_count=2,
        is_synthetic=synthetic is not None,
        synthetic_source=synthetic,
        initiated_by=InitiatedBy.API,
    )
    session.add(run)
    await session.commit()
    return run


class TestHappyPath:
    async def test_run_completes_and_records_metrics(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        route_to_mock(create_mock_app(token=TOKEN))
        await _seed(session)

        claimed = await claim_next_run(session)
        assert claimed is not None
        await execute_run(session, claimed, TOKEN)

        # Capture the id before expiring: expire_all() also expires `claimed`, and
        # reading an attribute off an expired object triggers a synchronous refresh.
        # Expiring is necessary because session.get returns the identity-mapped object
        # without re-applying loader options, so the eager load would silently not run.
        run_id = claimed.id
        session.expire_all()
        run = await session.scalar(
            select(Run).options(selectinload(Run.summary)).where(Run.id == run_id)
        )
        assert run is not None
        assert run.status is RunStatus.SUCCEEDED, run.error
        assert run.finished_at is not None
        assert run.summary is not None
        assert run.summary.successful_requests == 64
        assert run.summary.ttft_ms_p99 is not None

    async def test_raw_payload_is_kept_alongside_the_flattened_columns(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        # "Raw before derived": if the flattening is ever wrong, this is what allows
        # recomputation instead of re-running the GPU time.
        route_to_mock(create_mock_app(token=TOKEN))
        await _seed(session)
        claimed = await claim_next_run(session)
        assert claimed is not None
        await execute_run(session, claimed, TOKEN)

        run = await session.get(Run, claimed.id)
        assert run is not None and run.raw_result is not None
        assert run.raw_result["completed"] == 64

    async def test_provenance_is_recorded_from_the_host(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        route_to_mock(create_mock_app(token=TOKEN))
        await _seed(session)
        claimed = await claim_next_run(session)
        assert claimed is not None
        await execute_run(session, claimed, TOKEN)

        run = await session.get(Run, claimed.id)
        assert run is not None
        # Invariant 6: a run that cannot say what produced it is not a valid result.
        assert run.vllm_version == "0.25.1"
        assert run.agent_version
        assert run.gpu_model
        assert run.config_hash and run.workload_hash

    async def test_topology_comes_from_what_ran_not_the_config_text(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        """Invariant 8.

        The config asks for TP=2 and the agent confirms it. The value recorded is the
        agent's, so that a config whose request was silently ignored — a TP the host
        cannot satisfy — does not produce a run claiming a topology that never existed.
        """
        route_to_mock(create_mock_app(token=TOKEN))
        await _seed(session)
        claimed = await claim_next_run(session)
        assert claimed is not None
        await execute_run(session, claimed, TOKEN)

        run = await session.get(Run, claimed.id)
        assert run is not None
        assert run.tensor_parallel_size == 2
        assert run.device_indices == [0, 1]
        assert run.gpu_count == 2

    async def test_throughput_is_normalized_per_device(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        route_to_mock(create_mock_app(token=TOKEN))
        await _seed(session)
        claimed = await claim_next_run(session)
        assert claimed is not None
        await execute_run(session, claimed, TOKEN)

        run_id = claimed.id
        session.expire_all()
        run = await session.scalar(
            select(Run).options(selectinload(Run.summary)).where(Run.id == run_id)
        )
        assert run is not None and run.summary is not None
        aggregate = run.summary.output_token_throughput_tok_sec
        per_gpu = run.summary.output_token_throughput_per_gpu
        assert aggregate is not None and per_gpu is not None
        # Divided by the two devices that actually ran, not by a default of one.
        assert per_gpu == pytest.approx(aggregate / 2)


class TestQuarantine:
    async def test_a_run_from_a_mock_host_is_flagged_synthetic(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        route_to_mock(create_mock_app(token=TOKEN))
        await _seed(session)
        claimed = await claim_next_run(session)
        assert claimed is not None
        await execute_run(session, claimed, TOKEN)

        run = await session.get(Run, claimed.id)
        assert run is not None
        assert run.is_synthetic is True
        assert run.synthetic_source == "mock_agent"

    async def test_the_flag_is_re_derived_at_execution_time(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        """A host swapped between queueing and running must not launder its results.

        The run is seeded as non-synthetic; the agent it actually reaches declares
        itself a mock. What the agent says at execution time wins, because that is what
        actually produced the numbers.
        """
        route_to_mock(create_mock_app(token=TOKEN))
        await _seed(session, synthetic=None)
        claimed = await claim_next_run(session)
        assert claimed is not None
        await execute_run(session, claimed, TOKEN)

        run = await session.get(Run, claimed.id)
        assert run is not None
        assert run.is_synthetic is True
        assert run.synthetic_source == "mock_agent"


class TestFailurePaths:
    async def test_unreachable_agent_fails_the_run_with_a_reason(
        self, session: AsyncSession
    ) -> None:
        run = await _seed(session)
        run.gpu_host_id = run.gpu_host_id
        host = await session.get(GpuHost, run.gpu_host_id)
        assert host is not None
        host.agent_url = "http://127.0.0.1:9/nope"
        await session.commit()

        claimed = await claim_next_run(session)
        assert claimed is not None
        await execute_run(session, claimed, TOKEN)

        finished = await session.get(Run, claimed.id)
        assert finished is not None
        # Terminal, with an explanation. Anything else leaves the UI showing a run that
        # never finishes and a queue that never advances.
        assert finished.status is RunStatus.FAILED
        assert finished.error and "unreachable" in finished.error.lower()
        assert finished.finished_at is not None

    async def test_a_config_without_a_model_fails_before_benchmarking(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        route_to_mock(create_mock_app(token=TOKEN))
        run = await _seed(session)
        config = await session.get(ServerConfig, run.server_config_id)
        assert config is not None
        config.yaml = "max_num_seqs: 32\n"
        await session.commit()

        claimed = await claim_next_run(session)
        assert claimed is not None
        await execute_run(session, claimed, TOKEN)

        finished = await session.get(Run, claimed.id)
        assert finished is not None
        assert finished.status is RunStatus.FAILED
        assert "model" in (finished.error or "")


class TestClaiming:
    async def test_claiming_moves_the_run_out_of_queued(self, session: AsyncSession) -> None:
        await _seed(session)
        claimed = await claim_next_run(session)
        assert claimed is not None
        assert claimed.status is RunStatus.STARTING
        assert claimed.started_at is not None

    async def test_nothing_queued_returns_none(self, session: AsyncSession) -> None:
        assert await claim_next_run(session) is None

    async def test_a_claimed_run_is_not_claimed_twice(self, session: AsyncSession) -> None:
        # The property the sweep engine will depend on: two workers must not both pick
        # up the same point and run it twice on one GPU.
        await _seed(session)
        first = await claim_next_run(session)
        second = await claim_next_run(session)
        assert first is not None
        assert second is None
