"""Telemetry retrieval, and what thinning it must never do.

The endpoint under test has one job beyond fetching rows: return fewer of them when
asked, without changing what they say. That is easy to get wrong in a way nothing
notices — `gpu_sample` is keyed per device and stored interleaved, so the obvious
implementation (stride the flat list) returns a series that is complete, plausible,
correctly labelled, and missing entire GPUs.

These tests exist for that failure specifically. It is the `gpu_sample` keying rule in
CLAUDE.md restated as an assertion: a summary that destroys the imbalance signal is worse
than no summary, and one that destroys it silently is worse again.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import AsyncIterator

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_api.main import app as api_app
from vllmbench_api.settings import ApiSettings
from vllmbench_db.enums import InitiatedBy, RunStatus
from vllmbench_db.models import EngineSample, GpuHost, GpuSample, Run, ServerConfig, Workload
from vllmbench_db.session import create_engine, create_session_factory
from vllmbench_db.testing import reset_database, test_database_url

pytestmark = pytest.mark.integration


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
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


async def a_run(session: AsyncSession, *, gpus: int = 2) -> Run:
    host = GpuHost(name="ubuntu-llm", agent_url="http://agent", gpu_count=gpus)
    config = ServerConfig(config_hash=os.urandom(32).hex(), name="c", yaml="model: m\n")
    workload = Workload(
        workload_hash=os.urandom(32).hex(),
        name="w",
        dataset_name="random",
        num_prompts=64,
        max_concurrency=16,
    )
    session.add_all([host, config, workload])
    await session.flush()

    run = Run(
        server_config_id=config.id,
        workload_id=workload.id,
        gpu_host_id=host.id,
        status=RunStatus.SUCCEEDED,
        finished_at=dt.datetime.now(dt.UTC),
        config_hash=config.config_hash,
        workload_hash=workload.workload_hash,
        vllm_version="0.25.1",
        gpu_model="NVIDIA GeForce RTX 3090",
        gpu_count=gpus,
        tensor_parallel_size=gpus,
        initiated_by=InitiatedBy.UI,
    )
    session.add(run)
    await session.flush()
    return run


async def sample(
    session: AsyncSession,
    run: Run,
    *,
    ticks: int,
    devices: list[int],
    utilization: dict[int, float],
) -> None:
    """A per-device series, written interleaved as the agent writes it."""
    base = dt.datetime.now(dt.UTC)
    for tick in range(ticks):
        at = base + dt.timedelta(seconds=tick)
        session.add(EngineSample(run_id=run.id, sampled_at=at, num_requests_running=tick))
        for index in devices:
            session.add(
                GpuSample(
                    run_id=run.id,
                    gpu_index=index,
                    sampled_at=at,
                    sm_utilization_pct=utilization[index],
                    memory_used_bytes=20_000_000_000,
                )
            )
    await session.commit()


async def test_every_device_survives_thinning(
    client: httpx.AsyncClient, session: AsyncSession
) -> None:
    """The bug this endpoint's thinning exists to not have.

    Two devices, 400 samples each, thinned to 200. Striding the interleaved flat series
    by 2 returns 400 rows that are entirely device 0 — a full-looking, correctly-shaped,
    completely wrong picture in which the idle GPU does not exist.
    """
    run = await a_run(session)
    await sample(session, run, ticks=400, devices=[0, 1], utilization={0: 95.0, 1: 30.0})

    body = (await client.get(f"/api/runs/{run.id}/telemetry?max_samples=200")).json()

    assert sorted({s["gpu_index"] for s in body["gpu"]}) == [0, 1]
    # And both are still there in strength, not one sample of device 1 by luck.
    per_device = {0: 0, 1: 0}
    for s in body["gpu"]:
        per_device[s["gpu_index"]] += 1
    assert per_device[0] == per_device[1]

    # The exact wipeout: a budget of twice the per-device count makes the naive flat
    # stride 2, and the flat series is ordered (sampled_at, gpu_index) — so every kept
    # row is device 0 and device 1 disappears completely. Nothing about the response
    # would say so.
    body = (await client.get(f"/api/runs/{run.id}/telemetry?max_samples=400")).json()
    assert sorted({s["gpu_index"] for s in body["gpu"]}) == [0, 1]


async def test_the_imbalance_is_still_readable_after_thinning(
    client: httpx.AsyncClient, session: AsyncSession
) -> None:
    """Thinning drops samples; it must not move what the survivors say."""
    run = await a_run(session)
    await sample(session, run, ticks=400, devices=[0, 1], utilization={0: 95.0, 1: 30.0})

    body = (await client.get(f"/api/runs/{run.id}/telemetry?max_samples=50")).json()

    busy = [s["sm_utilization_pct"] for s in body["gpu"] if s["gpu_index"] == 0]
    idle = [s["sm_utilization_pct"] for s in body["gpu"] if s["gpu_index"] == 1]
    assert set(busy) == {95.0}
    assert set(idle) == {30.0}


async def test_the_budget_bounds_the_gpu_series_as_a_whole(
    client: httpx.AsyncClient, session: AsyncSession
) -> None:
    """Not per device.

    Otherwise `max_samples=200` on an eight-GPU host is 1600 rows, and the parameter
    that exists to bound a context window instead scales with the hardware.
    """
    run = await a_run(session, gpus=4)
    await sample(
        session, run, ticks=400, devices=[0, 1, 2, 3], utilization={i: 50.0 for i in range(4)}
    )

    body = (await client.get(f"/api/runs/{run.id}/telemetry?max_samples=200")).json()

    # Four devices sharing a 200 budget: 50 each, plus the kept final sample where the
    # series length does not divide evenly.
    assert len(body["gpu"]) <= 4 * (200 // 4 + 1)
    assert len(body["engine"]) <= 201
    assert sorted({s["gpu_index"] for s in body["gpu"]}) == [0, 1, 2, 3]


async def test_a_device_outlives_an_absurd_budget(
    client: httpx.AsyncClient, session: AsyncSession
) -> None:
    """The device floor beats the budget.

    Overshooting a context limit by three rows is recoverable. A device that is silently
    not in the response is the failure this whole module is about, and it must not be
    reachable by asking for a small number.
    """
    run = await a_run(session, gpus=4)
    await sample(
        session, run, ticks=400, devices=[0, 1, 2, 3], utilization={i: 50.0 for i in range(4)}
    )

    body = (await client.get(f"/api/runs/{run.id}/telemetry?max_samples=1")).json()

    assert sorted({s["gpu_index"] for s in body["gpu"]}) == [0, 1, 2, 3]


async def test_the_last_sample_is_always_kept(
    client: httpx.AsyncClient, session: AsyncSession
) -> None:
    """Where the run ended is a fact about it, not a casualty of arithmetic."""
    run = await a_run(session)
    await sample(session, run, ticks=333, devices=[0, 1], utilization={0: 95.0, 1: 30.0})

    full = (await client.get(f"/api/runs/{run.id}/telemetry")).json()
    thin = (await client.get(f"/api/runs/{run.id}/telemetry?max_samples=40")).json()

    assert thin["engine"][-1]["sampled_at"] == full["engine"][-1]["sampled_at"]
    for index in (0, 1):
        last_full = [s for s in full["gpu"] if s["gpu_index"] == index][-1]
        last_thin = [s for s in thin["gpu"] if s["gpu_index"] == index][-1]
        assert last_thin["sampled_at"] == last_full["sampled_at"]


async def test_the_response_says_it_was_thinned(
    client: httpx.AsyncClient, session: AsyncSession
) -> None:
    """A thinned series and a slowly-sampled one are indistinguishable otherwise.

    `sample_count` is what was recorded, so it stays put while the arrays shrink; that
    difference is the reader's only signal that they are not looking at all of it.
    """
    run = await a_run(session)
    await sample(session, run, ticks=400, devices=[0, 1], utilization={0: 95.0, 1: 30.0})

    full = (await client.get(f"/api/runs/{run.id}/telemetry")).json()
    thin = (await client.get(f"/api/runs/{run.id}/telemetry?max_samples=100")).json()

    assert full["stride"] == 1
    assert thin["stride"] > 1
    assert full["sample_count"] == thin["sample_count"] == 400 + 800
    assert len(thin["gpu"]) < len(full["gpu"])


async def test_unasked_returns_everything(client: httpx.AsyncClient, session: AsyncSession) -> None:
    """The chart wants every point. A browser is not the constrained reader here."""
    run = await a_run(session)
    await sample(session, run, ticks=120, devices=[0, 1], utilization={0: 95.0, 1: 30.0})

    body = (await client.get(f"/api/runs/{run.id}/telemetry")).json()

    assert len(body["engine"]) == 120
    assert len(body["gpu"]) == 240
    assert body["stride"] == 1


async def test_a_device_that_sampled_is_listed_even_when_it_sampled_less(
    client: httpx.AsyncClient, session: AsyncSession
) -> None:
    """Devices need not agree on how many readings they produced.

    A failed NVML read on one device is real. One stride across all of them keeps the
    two series on the same time grid, which is what makes them comparable at all.
    """
    run = await a_run(session)
    await sample(session, run, ticks=200, devices=[0, 1], utilization={0: 95.0, 1: 30.0})
    # Device 0 alone keeps going for a while.
    await sample(session, run, ticks=100, devices=[0], utilization={0: 95.0})

    body = (await client.get(f"/api/runs/{run.id}/telemetry?max_samples=60")).json()

    assert body["gpu_indices"] == [0, 1]
    assert {s["gpu_index"] for s in body["gpu"]} == {0, 1}
