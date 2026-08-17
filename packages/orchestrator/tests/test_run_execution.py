"""A run, end to end: queued → server started → benchmarked → persisted.

This is milestone 0.2.0's promise, exercised against a real database and the mock agent.
The failure paths get equal weight, because a run that fails must still end up in a
terminal state with a reason — a run stuck in `benchmarking` forever shows in the UI as
perpetually working and blocks whatever comes next.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
from collections.abc import AsyncIterator

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from vllmbench_db.enums import InitiatedBy, RunStatus
from vllmbench_db.models import (
    EngineSample,
    GpuHost,
    GpuSample,
    Run,
    ServerConfig,
    Workload,
)
from vllmbench_db.session import create_engine, create_session_factory
from vllmbench_db.testing import reset_database, test_database_url
from vllmbench_mockagent.main import create_app as create_mock_app
from vllmbench_orchestrator.runner import claim_next_run, execute_run

pytestmark = pytest.mark.integration

TOKEN = "test-token-not-a-real-secret"
MOCK_URL = "http://mock-agent"

CONFIG_YAML = "model: facebook/opt-125m\ntensor_parallel_size: 2\n"


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
    engine = create_engine(test_database_url())
    factory = create_session_factory(engine)
    # Sweeps and telemetry are emptied too. This fixture used to delete only runs, hosts
    # and summaries, so a sweep left behind by another package's tests made setup here
    # fail on a foreign key — in a file with nothing to do with the change.
    await reset_database(engine)
    async with factory() as s:
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


class TestServedModelName:
    """A config's `model:` is not necessarily the name the API answers to.

    Taken from a real config on the first GPU host: `model: Qwen/Qwen3.8-27B-FP8` with
    `served-model-name: Qwen3.8-27B`. The server's /v1/models reports only the alias, so
    requesting the HF id 404s every run. The mock never produced this because its configs
    had no alias.

    Both names have to travel, to different flags. `--model` is the weights, and vLLM
    loads the tokenizer from it; `--served-model-name` is what goes in the request body.
    Sending one name for both is wrong in whichever direction you pick.
    """

    async def test_both_names_travel_to_their_own_flags(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        route_to_mock(create_mock_app(token=TOKEN))
        run = await _seed(session)
        config = await session.get(ServerConfig, run.server_config_id)
        assert config is not None
        config.yaml = (
            "model: Qwen/Qwen3.8-27B-FP8\nserved-model-name: Qwen3.8-27B\ntensor-parallel-size: 2\n"
        )
        await session.commit()

        claimed = await claim_next_run(session)
        assert claimed is not None
        await execute_run(session, claimed, TOKEN)

        finished = await session.get(Run, claimed.id)
        assert finished is not None
        assert finished.status is RunStatus.SUCCEEDED, finished.error
        assert finished.raw_result is not None

        # The tokenizer came from the weights, not the alias. Getting this backwards is
        # how a run reports plausible numbers computed against the wrong tokenizer.
        assert finished.raw_result["model_id"] == "Qwen/Qwen3.8-27B-FP8"
        assert finished.raw_result["tokenizer_id"] == "Qwen/Qwen3.8-27B-FP8"
        # ...and the requests went to the alias the engine actually answers to.
        assert finished.raw_result["served_model_name"] == "Qwen3.8-27B"

    async def test_hyphenated_keys_are_understood(self) -> None:
        # Real vLLM configs use hyphens; ours used underscores. Both are accepted by
        # vLLM, so both must be accepted here.
        from vllmbench_orchestrator.runner import _model_names_from_config

        config = ServerConfig(
            config_hash="a" * 64, name="c", yaml="served-model-name: aliased\nmodel: w/p\n"
        )
        assert _model_names_from_config(config) == ("w/p", "aliased")

    async def test_alias_never_replaces_the_weights_id(self) -> None:
        """The alias must not end up as ``--model``.

        This is the regression that matters. vLLM loads the *tokenizer* from ``--model``,
        so passing the alias there either kills the benchmark (alias is not a repo id) or,
        worse, silently tokenizes against an unrelated tokenizer and records input-token
        counts that are confidently wrong.
        """
        from vllmbench_orchestrator.runner import _model_names_from_config

        config = ServerConfig(
            config_hash="c" * 64,
            name="c",
            yaml="model: Qwen/Qwen3.8-27B-FP8\nserved-model-name: Qwen3.8-27B\n",
        )
        weights, alias = _model_names_from_config(config)
        assert weights == "Qwen/Qwen3.8-27B-FP8"
        assert alias == "Qwen3.8-27B"

    async def test_plain_model_still_works(self) -> None:
        from vllmbench_orchestrator.runner import _model_names_from_config

        config = ServerConfig(config_hash="b" * 64, name="c", yaml="model: facebook/opt-125m\n")
        assert _model_names_from_config(config) == ("facebook/opt-125m", None)

    async def test_config_without_model_is_rejected(self) -> None:
        # A config with only an alias cannot be benchmarked: there is nothing to load a
        # tokenizer from, and guessing would mean inventing provenance.
        from vllmbench_orchestrator.runner import RunFailed, _model_names_from_config

        config = ServerConfig(config_hash="d" * 64, name="c", yaml="served-model-name: alias\n")
        with pytest.raises(RunFailed, match="model:"):
            _model_names_from_config(config)


class TestTelemetryPersistence:
    """Telemetry lands as rows, per device, aligned to the run.

    The interesting property is the alignment. The agent reports offsets from when
    sampling started, not wall-clock stamps, because the two hosts' clocks are not
    synchronised — the GPU host has no reason to run NTP against the control plane. A
    few seconds of skew would slide an entire timeline out of the window it describes,
    and nothing downstream could detect it.
    """

    async def test_samples_are_written_per_device(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        route_to_mock(create_mock_app(token=TOKEN))
        run = await _seed(session)
        config = await session.get(ServerConfig, run.server_config_id)
        assert config is not None
        config.yaml = "model: facebook/opt-125m\ntensor-parallel-size: 2\n"
        await session.commit()

        claimed = await claim_next_run(session)
        assert claimed is not None
        await execute_run(session, claimed, TOKEN)
        run_id = claimed.id
        session.expire_all()

        engine = list(
            (
                await session.execute(select(EngineSample).where(EngineSample.run_id == run_id))
            ).scalars()
        )
        gpu = list(
            (await session.execute(select(GpuSample).where(GpuSample.run_id == run_id))).scalars()
        )

        assert engine, "no engine telemetry was recorded"
        # Both devices present, and neither collapsed into a host-level average — the
        # per-device keying is what makes an imbalanced TP run diagnosable.
        assert {s.gpu_index for s in gpu} == {0, 1}
        per_device = [sum(1 for s in gpu if s.gpu_index == i) for i in (0, 1)]
        assert per_device[0] == per_device[1] > 0

    async def test_kv_cache_is_stored_as_a_fraction(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        # vLLM's metric is named `_perc` but emits 0..1. Storing a percent here would
        # make every chart and export wrong by 100x in one direction or the other.
        route_to_mock(create_mock_app(token=TOKEN))
        await _seed(session)
        claimed = await claim_next_run(session)
        assert claimed is not None
        await execute_run(session, claimed, TOKEN)
        # Before expire_all: expiring `claimed` makes any later attribute read attempt a
        # synchronous refresh, which is IO on the wrong side of the greenlet boundary.
        run_id = claimed.id
        session.expire_all()

        values = [
            s.kv_cache_usage_fraction
            for s in (
                await session.execute(select(EngineSample).where(EngineSample.run_id == run_id))
            ).scalars()
            if s.kv_cache_usage_fraction is not None
        ]
        assert values
        assert all(0.0 <= v <= 1.0 for v in values), f"not a fraction: {max(values)}"

    async def test_counters_never_go_backwards(self, session: AsyncSession, route_to_mock) -> None:
        """Counters are stored raw so any window can be differenced.

        A counter that decreased would produce a negative rate downstream, which is
        worse than no rate at all — it looks like a measurement rather than a bug.
        """
        route_to_mock(create_mock_app(token=TOKEN))
        await _seed(session)
        claimed = await claim_next_run(session)
        assert claimed is not None
        await execute_run(session, claimed, TOKEN)
        run_id = claimed.id
        session.expire_all()

        samples = list(
            (
                await session.execute(
                    select(EngineSample)
                    .where(EngineSample.run_id == run_id)
                    .order_by(EngineSample.sampled_at)
                )
            ).scalars()
        )
        queries = [s.prefix_cache_queries_total for s in samples if s.prefix_cache_queries_total]
        assert queries == sorted(queries)

    async def test_the_series_lands_inside_the_run_window(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        # Anchored to the benchmark, not to either host's wall clock. If this drifts,
        # the timeline describes a window the run never occupied.
        route_to_mock(create_mock_app(token=TOKEN))
        await _seed(session)
        claimed = await claim_next_run(session)
        assert claimed is not None
        await execute_run(session, claimed, TOKEN)
        run_id = claimed.id
        session.expire_all()

        run = await session.get(Run, run_id)
        assert run is not None and run.started_at is not None and run.finished_at is not None
        samples = list(
            (
                await session.execute(
                    select(EngineSample)
                    .where(EngineSample.run_id == run_id)
                    .order_by(EngineSample.sampled_at)
                )
            ).scalars()
        )
        assert samples
        # The mock stretches its synthetic window past its own (very short) benchmark, so
        # the assertion is that sampling begins at or after the run did, not that it fits
        # entirely inside — that is what a real, minutes-long benchmark gives.
        assert samples[0].sampled_at >= run.started_at - dt.timedelta(seconds=5)
        assert samples[0].sampled_at < samples[-1].sampled_at


class TestServerReuse:
    """Restart the engine only when the server config actually changes.

    A sweep varies workloads and replicates underneath one config far more often than it
    changes the config, and a restart is minutes for a large model. This is most of the
    difference between a sweep that takes an afternoon and one that takes a day — but it
    is also the change most able to leak a warm engine into the next measurement, so the
    conditions are pinned here.
    """

    async def test_a_second_run_on_the_same_config_reuses_the_engine(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        mock = create_mock_app(token=TOKEN)
        route_to_mock(mock)

        first = await _seed(session)
        config_id, workload_id, host_id = (
            first.server_config_id,
            first.workload_id,
            first.gpu_host_id,
        )
        # A second run with identical config and workload, as a sweep would produce.
        session.add(
            Run(
                gpu_host_id=host_id,
                server_config_id=config_id,
                workload_id=workload_id,
                status=RunStatus.QUEUED,
                config_hash=first.config_hash,
                workload_hash=first.workload_hash,
                gpu_count=2,
                is_synthetic=True,
                synthetic_source="mock_agent",
                initiated_by=InitiatedBy.API,
                sweep_seq=1,
            )
        )
        await session.commit()

        one = await claim_next_run(session)
        assert one is not None
        await execute_run(session, one, TOKEN)

        # The engine is deliberately left loaded, because the next queued run wants it.
        status_after = await _mock_server_state(mock)
        assert status_after == "ready", "engine was torn down between identical configs"

        two = await claim_next_run(session)
        assert two is not None
        await execute_run(session, two, TOKEN)

        session.expire_all()
        both = list(
            (await session.execute(select(Run).order_by(Run.sweep_seq.nulls_first()))).scalars()
        )
        assert [r.status for r in both] == [RunStatus.SUCCEEDED, RunStatus.SUCCEEDED]

    async def test_the_last_run_tears_the_engine_down(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        # Holding VRAM is only justified by something about to use it. With the queue
        # empty, nothing is.
        mock = create_mock_app(token=TOKEN)
        route_to_mock(mock)
        await _seed(session)

        claimed = await claim_next_run(session)
        assert claimed is not None
        await execute_run(session, claimed, TOKEN)

        assert await _mock_server_state(mock) == "stopped"

    async def test_a_changed_config_restarts_the_engine(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        mock = create_mock_app(token=TOKEN)
        route_to_mock(mock)

        first = await _seed(session)
        other = ServerConfig(
            config_hash=os.urandom(32).hex(),
            name="other",
            yaml="model: facebook/opt-125m\ntensor-parallel-size: 1\n",
        )
        session.add(other)
        await session.flush()
        session.add(
            Run(
                gpu_host_id=first.gpu_host_id,
                server_config_id=other.id,
                workload_id=first.workload_id,
                status=RunStatus.QUEUED,
                config_hash=other.config_hash,
                workload_hash=first.workload_hash,
                gpu_count=2,
                is_synthetic=True,
                synthetic_source="mock_agent",
                initiated_by=InitiatedBy.API,
                sweep_seq=1,
            )
        )
        await session.commit()

        one = await claim_next_run(session)
        assert one is not None
        await execute_run(session, one, TOKEN)
        # Next run wants a different config, so nothing is worth keeping loaded.
        assert await _mock_server_state(mock) == "stopped"

        two = await claim_next_run(session)
        assert two is not None
        await execute_run(session, two, TOKEN)

        # Before expire_all: reading an attribute off an expired object triggers a
        # synchronous refresh, which is IO on the wrong side of the greenlet boundary.
        two_id = two.id
        session.expire_all()
        finished = await session.get(Run, two_id)
        assert finished is not None and finished.status is RunStatus.SUCCEEDED

    async def test_a_failed_run_never_leaves_the_engine_loaded(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        """Reuse is an optimization; holding VRAM after a failure is a leak.

        The next run may well want the same config, but a host that just failed is not
        one to hand a warm engine to without knowing why.
        """
        mock = create_mock_app(token=TOKEN)
        route_to_mock(mock)
        run = await _seed(session)
        config = await session.get(ServerConfig, run.server_config_id)
        assert config is not None
        # No `model:` line, so the run fails before benchmarking.
        config.yaml = "served-model-name: nothing\n"
        await session.commit()

        claimed = await claim_next_run(session)
        assert claimed is not None
        await execute_run(session, claimed, TOKEN)

        run_id = claimed.id
        session.expire_all()
        finished = await session.get(Run, run_id)
        assert finished is not None and finished.status is RunStatus.FAILED
        assert await _mock_server_state(mock) == "stopped"


class TestInFlightCancellation:
    async def test_cancelling_mid_benchmark_stops_the_run(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        """The point of cancelling: it takes effect now, not after this point finishes.

        A benchmark runs for minutes to hours. Waiting for it would make cancelling a
        sweep nearly meaningless, which is why the agent can stop a benchmark in flight.
        """
        mock = create_mock_app(token=TOKEN)
        route_to_mock(mock)
        await _seed(session)

        claimed = await claim_next_run(session)
        assert claimed is not None

        cancel = asyncio.Event()
        cancel.set()  # already cancelled by the time the run reaches the benchmark
        run_id = claimed.id
        await execute_run(session, claimed, TOKEN, cancel=cancel)

        session.expire_all()
        finished = await session.get(Run, run_id)
        assert finished is not None
        # Cancelled, not failed: the system did what it was told.
        assert finished.status is RunStatus.CANCELLED
        assert "cancel" in (finished.error or "").lower()
        # And nothing was left holding VRAM.
        assert await _mock_server_state(mock) == "stopped"

    async def test_cancelling_records_no_summary(
        self, session: AsyncSession, route_to_mock
    ) -> None:
        # A cancelled run measured nothing. A summary row would put a half-finished
        # benchmark into charts alongside complete ones.
        route_to_mock(create_mock_app(token=TOKEN))
        await _seed(session)
        claimed = await claim_next_run(session)
        assert claimed is not None

        cancel = asyncio.Event()
        cancel.set()
        run_id = claimed.id
        await execute_run(session, claimed, TOKEN, cancel=cancel)

        session.expire_all()
        run = await session.scalar(
            select(Run).options(selectinload(Run.summary)).where(Run.id == run_id)
        )
        assert run is not None
        assert run.summary is None


async def _mock_server_state(mock_app) -> str:
    """Ask the mock agent what its engine is doing, through its real HTTP contract."""
    transport = httpx.ASGITransport(app=mock_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock-agent") as c:
        response = await c.get("/server", headers={"Authorization": f"Bearer {TOKEN}"})
        return str(response.json()["state"])
