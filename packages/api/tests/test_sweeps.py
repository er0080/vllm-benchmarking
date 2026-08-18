"""Sweep authoring, end to end against a real database.

The property under test throughout is that **the plan is a fact in the database**, not a
belief held by a process. Every run exists before anything executes, in the order it will
execute, with the provenance it will carry. That is what makes progress countable, resume
free, and a multi-hour job independent of any one service staying up.
"""

from __future__ import annotations

import datetime as dt
import itertools
import os
import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_api.main import app as api_app
from vllmbench_api.settings import ApiSettings
from vllmbench_db.enums import ReplicateOrder, RunStatus, SweepStatus
from vllmbench_db.models import GpuHost, Run, ServerConfig, Sweep, Workload
from vllmbench_db.session import create_engine, create_session_factory
from vllmbench_db.testing import reset_database, test_database_url

pytestmark = pytest.mark.integration

BASE_CONFIG = """\
model: Qwen/Qwen3.8-27B-FP8
served-model-name: Qwen3.8-27B
tensor-parallel-size: 2
max-num-seqs: 2                # was 1 — this is the fix
"""


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """A clean database, with the API wired to the same engine.

    The app's state is populated here rather than by running its lifespan: the lifespan
    also runs the schema-version check and would need a live migration state, and what
    these tests exercise is the routers, not startup.
    """
    engine = create_engine(test_database_url())
    factory = create_session_factory(engine)
    api_app.state.engine = engine
    api_app.state.sessions = factory
    api_app.state.settings = ApiSettings(token="test-token-not-a-real-secret")

    await reset_database(engine)
    async with factory() as s:
        yield s
    await engine.dispose()


@pytest.fixture
async def client(session: AsyncSession) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as c:
        yield c


async def _fixtures(
    session: AsyncSession, *, gpu_count: int = 2, synthetic: str | None = None
) -> tuple[GpuHost, ServerConfig, Workload]:
    host = GpuHost(
        name=f"host-{os.urandom(4).hex()}",
        agent_url="http://agent",
        gpu_count=gpu_count,
        synthetic_source=synthetic,
    )
    config = ServerConfig(config_hash=os.urandom(32).hex(), name="base", yaml=BASE_CONFIG)
    workload = Workload(
        workload_hash=os.urandom(32).hex(),
        name="wl",
        dataset_name="random",
        num_prompts=64,
        max_concurrency=16,
    )
    session.add_all([host, config, workload])
    await session.commit()
    return host, config, workload


class TestMaterialization:
    async def test_every_run_exists_before_anything_runs(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        host, config, workload = await _fixtures(session)

        response = await client.post(
            "/api/sweeps",
            json={
                "name": "matrix",
                "gpu_host_id": str(host.id),
                "server_config_ids": [str(config.id)],
                "workload_ids": [str(workload.id)],
                "tensor_parallel_sizes": [1, 2],
                "replicates": 3,
            },
        )
        assert response.status_code == 201, response.text
        body = response.json()

        # 2 configs (TP1, TP2) x 1 workload x 3 replicates
        assert body["progress"]["total"] == 6
        assert body["progress"]["queued"] == 6
        assert body["status"] == SweepStatus.QUEUED

        runs = list(
            (
                await session.execute(
                    select(Run).where(Run.sweep_id == body["id"]).order_by(Run.sweep_seq)
                )
            ).scalars()
        )
        assert [r.sweep_seq for r in runs] == [0, 1, 2, 3, 4, 5]
        assert all(r.status is RunStatus.QUEUED for r in runs)

    async def test_plan_order_keeps_a_config_contiguous(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        # The whole reason sweep_seq exists: a config change costs an engine restart.
        host, config, workload = await _fixtures(session)
        response = await client.post(
            "/api/sweeps",
            json={
                "name": "ordered",
                "gpu_host_id": str(host.id),
                "server_config_ids": [str(config.id)],
                "workload_ids": [str(workload.id)],
                "tensor_parallel_sizes": [1, 2],
                "replicates": 2,
            },
        )
        assert response.status_code == 201
        hashes = list(
            (
                await session.execute(
                    select(Run.config_hash)
                    .where(Run.sweep_id == response.json()["id"])
                    .order_by(Run.sweep_seq)
                )
            ).scalars()
        )
        # Two configs, each contiguous: exactly one change along the sequence.
        assert len(set(hashes)) == 2
        assert sum(1 for a, b in itertools.pairwise(hashes) if a != b) == 1
        assert response.json()["engine_starts"] == 2

    async def test_interleaved_costs_more_engine_starts(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        host, config, workload = await _fixtures(session)
        response = await client.post(
            "/api/sweeps",
            json={
                "name": "interleaved",
                "gpu_host_id": str(host.id),
                "server_config_ids": [str(config.id)],
                "workload_ids": [str(workload.id)],
                "tensor_parallel_sizes": [1, 2],
                "replicates": 3,
                "replicate_order": ReplicateOrder.INTERLEAVED,
            },
        )
        assert response.status_code == 201
        # 2 configs x 3 replicate passes. The cost of honest error bars, made visible
        # before the sweep is started rather than discovered while it runs.
        assert response.json()["engine_starts"] == 6


class TestTensorParallelAxis:
    async def test_variants_become_real_content_addressed_configs(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        """A generated variant is an ordinary config, not a special case.

        This is what keeps invariant 5 intact: what is stored is what runs, and the
        variant is stored exactly once, hashed by its own text.
        """
        host, config, workload = await _fixtures(session)
        response = await client.post(
            "/api/sweeps",
            json={
                "name": "tp",
                "gpu_host_id": str(host.id),
                "server_config_ids": [str(config.id)],
                "workload_ids": [str(workload.id)],
                "tensor_parallel_sizes": [1, 2],
                "replicates": 1,
            },
        )
        assert response.status_code == 201

        runs = list(
            (
                await session.execute(
                    select(Run).where(Run.sweep_id == response.json()["id"]).order_by(Run.sweep_seq)
                )
            ).scalars()
        )
        configs = [await session.get(ServerConfig, r.server_config_id) for r in runs]
        assert all(c is not None for c in configs)

        texts = [c.yaml for c in configs if c is not None]
        assert any("tensor-parallel-size: 1" in t for t in texts)
        assert any("tensor-parallel-size: 2" in t for t in texts)
        # The author's comment survived into the executed config.
        assert all("# was 1 — this is the fix" in t for t in texts)
        # Each run's recorded hash matches the config it points at.
        for run, cfg in zip(runs, configs, strict=True):
            assert cfg is not None
            assert run.config_hash == cfg.config_hash

    async def test_declared_tp_is_recorded_on_the_run(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        host, config, workload = await _fixtures(session)
        response = await client.post(
            "/api/sweeps",
            json={
                "name": "tp",
                "gpu_host_id": str(host.id),
                "server_config_ids": [str(config.id)],
                "workload_ids": [str(workload.id)],
                "tensor_parallel_sizes": [1, 2],
                "replicates": 1,
            },
        )
        runs = list(
            (
                await session.execute(
                    select(Run).where(Run.sweep_id == response.json()["id"]).order_by(Run.sweep_seq)
                )
            ).scalars()
        )
        assert sorted(r.tensor_parallel_size for r in runs) == [1, 2]

    async def test_reruns_reuse_the_same_variant_rows(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        # Content addressing: authoring the same sweep twice must not duplicate configs.
        host, config, workload = await _fixtures(session)
        payload = {
            "name": "again",
            "gpu_host_id": str(host.id),
            "server_config_ids": [str(config.id)],
            "workload_ids": [str(workload.id)],
            "tensor_parallel_sizes": [1, 2],
            "replicates": 1,
        }
        created = await client.post("/api/sweeps", json=payload)
        assert created.status_code == 201
        first = len(list((await session.execute(select(ServerConfig))).scalars()))

        # Cancelled between the two, because a host may only hold one active sweep — the
        # rerun this test is about is a rerun, not a second concurrent sweep.
        cancelled = await client.post(f"/api/sweeps/{created.json()['id']}/cancel")
        assert cancelled.status_code == 200

        assert (await client.post("/api/sweeps", json=payload)).status_code == 201
        second = len(list((await session.execute(select(ServerConfig))).scalars()))

        assert first == second, "a repeated sweep created duplicate configs"

    async def test_tp_beyond_the_host_is_refused(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        """Refused at authoring, not left to fail at run time.

        vLLM can come up on fewer devices than requested, and the run would then be
        normalized per-GPU against a device count that never existed.
        """
        host, config, workload = await _fixtures(session, gpu_count=2)
        response = await client.post(
            "/api/sweeps",
            json={
                "name": "too wide",
                "gpu_host_id": str(host.id),
                "server_config_ids": [str(config.id)],
                "workload_ids": [str(workload.id)],
                "tensor_parallel_sizes": [4],
                "replicates": 1,
            },
        )
        assert response.status_code == 422
        assert "exceeds the 2 GPU" in response.json()["detail"]
        # And nothing was left behind.
        assert list((await session.execute(select(Sweep))).scalars()) == []

    async def test_config_own_tp_is_validated_when_there_is_no_axis(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        host, _, workload = await _fixtures(session, gpu_count=1)
        config = ServerConfig(
            config_hash=os.urandom(32).hex(),
            name="wide",
            yaml="model: m\ntensor-parallel-size: 4\n",
        )
        session.add(config)
        await session.commit()

        response = await client.post(
            "/api/sweeps",
            json={
                "name": "no axis",
                "gpu_host_id": str(host.id),
                "server_config_ids": [str(config.id)],
                "workload_ids": [str(workload.id)],
                "replicates": 1,
            },
        )
        assert response.status_code == 422


class TestProvenanceAndQuarantine:
    async def test_synthetic_host_quarantines_every_run(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        # Invariant 7's chain of custody has to survive the sweep path too — this is the
        # path that will produce most runs, so a gap here is a gap everywhere.
        host, config, workload = await _fixtures(session, synthetic="mock_agent")
        response = await client.post(
            "/api/sweeps",
            json={
                "name": "fake",
                "gpu_host_id": str(host.id),
                "server_config_ids": [str(config.id)],
                "workload_ids": [str(workload.id)],
                "replicates": 2,
            },
        )
        assert response.status_code == 201
        assert response.json()["is_synthetic"] is True

        runs = list(
            (
                await session.execute(select(Run).where(Run.sweep_id == response.json()["id"]))
            ).scalars()
        )
        assert runs
        assert all(r.is_synthetic and r.synthetic_source == "mock_agent" for r in runs)


class TestCancellation:
    async def test_cancel_stops_queued_runs(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        host, config, workload = await _fixtures(session)
        created = await client.post(
            "/api/sweeps",
            json={
                "name": "cancel me",
                "gpu_host_id": str(host.id),
                "server_config_ids": [str(config.id)],
                "workload_ids": [str(workload.id)],
                "replicates": 4,
            },
        )
        sweep_id = created.json()["id"]

        response = await client.post(f"/api/sweeps/{sweep_id}/cancel")
        assert response.status_code == 200
        assert response.json()["status"] == SweepStatus.CANCELLED
        assert response.json()["progress"]["cancelled"] == 4
        assert response.json()["progress"]["queued"] == 0

    async def test_cancel_is_idempotent(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        host, config, workload = await _fixtures(session)
        created = await client.post(
            "/api/sweeps",
            json={
                "name": "twice",
                "gpu_host_id": str(host.id),
                "server_config_ids": [str(config.id)],
                "workload_ids": [str(workload.id)],
                "replicates": 1,
            },
        )
        sweep_id = created.json()["id"]
        assert (await client.post(f"/api/sweeps/{sweep_id}/cancel")).status_code == 200
        assert (await client.post(f"/api/sweeps/{sweep_id}/cancel")).status_code == 200

    async def test_cancel_leaves_an_in_flight_run_alone(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        """The orchestrator owns the run that is executing.

        Marking it cancelled from here would leave the orchestrator writing results for a
        run the database says was cancelled — and a terminal run is immutable, so that
        write fails at the database and the measurement is lost.
        """
        host, config, workload = await _fixtures(session)
        created = await client.post(
            "/api/sweeps",
            json={
                "name": "in flight",
                "gpu_host_id": str(host.id),
                "server_config_ids": [str(config.id)],
                "workload_ids": [str(workload.id)],
                "replicates": 3,
            },
        )
        sweep_id = created.json()["id"]

        first = await session.scalar(
            select(Run).where(Run.sweep_id == sweep_id).order_by(Run.sweep_seq).limit(1)
        )
        assert first is not None
        first.status = RunStatus.BENCHMARKING
        await session.commit()

        response = await client.post(f"/api/sweeps/{sweep_id}/cancel")
        assert response.status_code == 200
        assert response.json()["progress"]["benchmarking"] == 1
        assert response.json()["progress"]["cancelled"] == 2


class TestValidation:
    async def test_unknown_host(self, session: AsyncSession, client: httpx.AsyncClient) -> None:
        _, config, workload = await _fixtures(session)
        response = await client.post(
            "/api/sweeps",
            json={
                "name": "x",
                "gpu_host_id": "00000000-0000-0000-0000-000000000000",
                "server_config_ids": [str(config.id)],
                "workload_ids": [str(workload.id)],
            },
        )
        assert response.status_code == 404

    async def test_empty_axes_rejected_by_the_schema(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        host, _, workload = await _fixtures(session)
        response = await client.post(
            "/api/sweeps",
            json={
                "name": "x",
                "gpu_host_id": str(host.id),
                "server_config_ids": [],
                "workload_ids": [str(workload.id)],
            },
        )
        assert response.status_code == 422


class TestGuardrails:
    """Limits that exist because two callers can now author sweeps without seeing each
    other — a person in the UI and an agent over MCP."""

    async def test_a_host_holds_one_active_sweep(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        """Physical, not procedural: two sweeps on one host interleave engine restarts.

        Each then pays for the other's config changes and neither measures the sequence
        it planned.
        """
        host, config, workload = await _fixtures(session)
        payload = {
            "name": "first",
            "gpu_host_id": str(host.id),
            "server_config_ids": [str(config.id)],
            "workload_ids": [str(workload.id)],
            "replicates": 1,
        }
        assert (await client.post("/api/sweeps", json=payload)).status_code == 201

        second = await client.post("/api/sweeps", json={**payload, "name": "second"})
        assert second.status_code == 409
        # Names the sweep in the way, so the caller can act rather than guess.
        assert "first" in second.json()["detail"]

    async def test_a_finished_sweep_releases_the_host(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        host, config, workload = await _fixtures(session)
        payload = {
            "name": "done",
            "gpu_host_id": str(host.id),
            "server_config_ids": [str(config.id)],
            "workload_ids": [str(workload.id)],
            "replicates": 1,
        }
        created = await client.post("/api/sweeps", json=payload)
        await client.post(f"/api/sweeps/{created.json()['id']}/cancel")

        assert (
            await client.post("/api/sweeps", json={**payload, "name": "next"})
        ).status_code == 201

    async def test_a_runaway_matrix_is_refused_at_authoring(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        """A caller that got its product wrong finds out now, not after days of GPU time.

        The failure this prevents is not an error — it is a queue quietly filled with work
        nobody meant to ask for. Replicates are already bounded on their own; this is the
        *product* of the axes, which no single field can catch.
        """
        host, config, _ = await _fixtures(session)
        workload_ids = []
        for concurrency in range(1, 12):
            created = await client.post(
                "/api/workloads",
                json={
                    "name": f"c{concurrency}",
                    "dataset_name": "random",
                    "num_prompts": 32,
                    "max_concurrency": concurrency,
                },
            )
            assert created.status_code in (200, 201), created.text
            workload_ids.append(created.json()["id"])

        # 11 workloads x 2 tensor-parallel sizes x 25 replicates = 550.
        response = await client.post(
            "/api/sweeps",
            json={
                "name": "runaway",
                "gpu_host_id": str(host.id),
                "server_config_ids": [str(config.id)],
                "workload_ids": workload_ids,
                "tensor_parallel_sizes": [1, 2],
                "replicates": 25,
            },
        )
        assert response.status_code == 422, response.text
        assert "550 runs" in response.json()["detail"]

    async def test_who_asked_is_recorded(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        """Invariant 6. Every run and its sweep must be able to say what produced it.

        This only became answerable-and-wrong once a second interface could create work:
        before MCP, everything claimed the UI, and the claim happened to be true.
        """
        host, config, workload = await _fixtures(session)
        created = await client.post(
            "/api/sweeps",
            json={
                "name": "by an agent",
                "gpu_host_id": str(host.id),
                "server_config_ids": [str(config.id)],
                "workload_ids": [str(workload.id)],
                "replicates": 1,
                "initiated_by": "mcp",
                "initiated_by_client": "claude-code",
            },
        )
        assert created.status_code == 201
        assert created.json()["initiated_by"] == "mcp"

        sweep_id = created.json()["id"]
        runs = list(
            (
                await session.execute(select(Run).where(Run.sweep_id == uuid.UUID(sweep_id)))
            ).scalars()
        )
        # On every run, not only on the sweep: a run is the thing that gets charted, and
        # it has to answer for itself.
        assert runs and all(str(run.initiated_by) == "mcp" for run in runs)
        assert all(run.initiated_by_client == "claude-code" for run in runs)

    async def test_an_unidentified_caller_is_recorded_as_api_not_ui(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        # HTTP cannot tell a browser from curl, so the honest default is "some API
        # caller". Claiming the UI would be a confident wrong answer in a provenance
        # column.
        host, config, workload = await _fixtures(session)
        created = await client.post(
            "/api/sweeps",
            json={
                "name": "anonymous",
                "gpu_host_id": str(host.id),
                "server_config_ids": [str(config.id)],
                "workload_ids": [str(workload.id)],
                "replicates": 1,
            },
        )
        assert created.json()["initiated_by"] == "api"


class TestRemainingTimeEstimate:
    """What `get_sweep` says about how much longer, and when it declines to say.

    The estimate's arithmetic is covered in :mod:`test_duration`; this is the wiring —
    that the plan is read in execution order, that a run's engine load is attributed to
    the right run, and that a finished sweep does not carry a countdown.
    """

    async def _sweep_of(
        self, session: AsyncSession, client: httpx.AsyncClient, *, replicates: int
    ) -> str:
        host, config, workload = await _fixtures(session)
        response = await client.post(
            "/api/sweeps",
            json={
                "name": "estimated",
                "gpu_host_id": str(host.id),
                "server_config_ids": [str(config.id)],
                "workload_ids": [str(workload.id)],
                "replicates": replicates,
            },
        )
        assert response.status_code == 201, response.text
        return str(response.json()["id"])

    async def _finish(
        self, session: AsyncSession, sweep_id: str, *, count: int, seconds: list[float]
    ) -> None:
        """Mark the first `count` runs succeeded, with the durations given."""
        runs = list(
            (
                await session.execute(
                    select(Run).where(Run.sweep_id == uuid.UUID(sweep_id)).order_by(Run.sweep_seq)
                )
            ).scalars()
        )
        base = dt.datetime.now(dt.UTC)
        for run, duration in zip(runs[:count], seconds, strict=True):
            run.started_at = base
            run.finished_at = base + dt.timedelta(seconds=duration)
            run.status = RunStatus.SUCCEEDED
        await session.commit()

    async def test_nothing_finished_means_no_number(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        """A countdown with nothing behind it looks measured. It must not exist."""
        sweep_id = await self._sweep_of(session, client, replicates=4)

        estimate = (await client.get(f"/api/sweeps/{sweep_id}")).json()["estimated_remaining"]

        assert estimate["seconds_remaining"] is None
        assert estimate["runs_remaining"] == 4
        # One config, so only the first run pays for a load.
        assert estimate["engine_loads_remaining"] == 1
        assert estimate["caveats"]

    async def test_the_engine_load_is_charged_to_the_run_that_paid_for_it(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        """The decomposition, end to end.

        Four runs of one config: the first loads the engine, the rest do not. Once two
        have finished, the two remaining are priced at benchmark time alone — and the
        recovered load overhead is not charged again, because nothing left restarts.
        """
        sweep_id = await self._sweep_of(session, client, replicates=4)
        await self._finish(session, sweep_id, count=2, seconds=[240.0, 60.0])

        estimate = (await client.get(f"/api/sweeps/{sweep_id}")).json()["estimated_remaining"]

        assert estimate["median_run_seconds"] == 60.0
        assert estimate["median_engine_load_seconds"] == 180.0
        assert estimate["engine_loads_remaining"] == 0
        # Two runs at 60s, with no load to pay for. An estimate that pooled the two
        # observed durations would have said 300s — five minutes for two minutes of work.
        assert estimate["seconds_remaining"] == 120.0
        assert estimate["sample_size"] == 2

    async def test_a_failed_run_is_neither_counted_nor_extrapolated_from(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        """A run that died after ten seconds is not evidence a run takes ten seconds."""
        sweep_id = await self._sweep_of(session, client, replicates=4)
        await self._finish(session, sweep_id, count=2, seconds=[240.0, 60.0])
        runs = list(
            (
                await session.execute(
                    select(Run).where(Run.sweep_id == uuid.UUID(sweep_id)).order_by(Run.sweep_seq)
                )
            ).scalars()
        )
        died_at = dt.datetime.now(dt.UTC)
        runs[2].status = RunStatus.FAILED
        runs[2].started_at = died_at
        runs[2].finished_at = died_at + dt.timedelta(seconds=10)
        await session.commit()

        estimate = (await client.get(f"/api/sweeps/{sweep_id}")).json()["estimated_remaining"]

        assert estimate["sample_size"] == 2
        assert estimate["runs_remaining"] == 1
        assert estimate["seconds_remaining"] == 60.0

    async def test_a_finished_sweep_has_no_countdown(
        self, session: AsyncSession, client: httpx.AsyncClient
    ) -> None:
        """Not a zero, which would be a claim about work that is not happening."""
        sweep_id = await self._sweep_of(session, client, replicates=2)
        await client.post(f"/api/sweeps/{sweep_id}/cancel")

        body = (await client.get(f"/api/sweeps/{sweep_id}")).json()

        assert body["estimated_remaining"] is None
