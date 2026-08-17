"""Sweep authoring, end to end against a real database.

The property under test throughout is that **the plan is a fact in the database**, not a
belief held by a process. Every run exists before anything executes, in the order it will
execute, with the provenance it will carry. That is what makes progress countable, resume
free, and a multi-hour job independent of any one service staying up.
"""

from __future__ import annotations

import itertools
import os
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
        assert (await client.post("/api/sweeps", json=payload)).status_code == 201
        first = len(list((await session.execute(select(ServerConfig))).scalars()))

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
