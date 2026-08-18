"""The storage surface, and the one route whose purpose is to destroy data.

Most of what matters about retention is tested against the database in
`packages/db/tests/test_retention.py`. What is tested here is the shape of the HTTP
surface — specifically, that the deleting route is harder to invoke than the ones that
create things, and that the reporting route answers the question an operator actually
has about a large table.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator

import httpx
import pytest
from api_world import World
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_db.enums import RunStatus
from vllmbench_db.models import EngineSample, GpuSample, Run

pytestmark = pytest.mark.integration


@pytest.fixture
async def old_run_with_telemetry(session: AsyncSession, world: World) -> AsyncIterator[Run]:
    # Built already old rather than aged afterwards. A terminal run cannot be updated —
    # the immutability trigger refuses even a raw UPDATE, which is exactly what it is for
    # — so "two hundred days have passed" has to be a fact at insert time.
    finished = dt.datetime.now(dt.UTC) - dt.timedelta(days=200)
    host = await world.host("h")
    run = await world.run(
        host, await world.config("c"), await world.a_workload(), finished_at=finished
    )

    for i in range(4):
        at = finished - dt.timedelta(seconds=i)
        session.add(EngineSample(run_id=run.id, sampled_at=at, num_requests_running=2))
        session.add(GpuSample(run_id=run.id, gpu_index=0, sampled_at=at, sm_utilization_pct=80.0))
    await session.commit()
    yield run


class TestReport:
    async def test_it_says_which_tables_can_be_reclaimed_and_why_not(
        self, client: httpx.AsyncClient, old_run_with_telemetry: Run
    ) -> None:
        """The question an operator has about a large table is "can I delete it".

        "gpu_sample is 40 GB" is only actionable next to "and it may be deleted";
        "run_summary is 40 GB, and it may not" is a different conversation. The reason is
        carried too, because "protected" without a why invites someone to override it.
        """
        body = (await client.get("/api/storage")).json()
        tables = {t["name"]: t for t in body["tables"]}

        assert tables["gpu_sample"]["protected"] is False
        assert tables["run_summary"]["protected"] is True
        assert tables["run_summary"]["protected_because"]
        assert body["total_bytes"] > 0
        assert body["gpu_samples"] == 4


class TestPruning:
    async def test_it_is_a_dry_run_unless_confirmed(
        self, client: httpx.AsyncClient, old_run_with_telemetry: Run
    ) -> None:
        """The default answer to "what would this remove" is a number, not a removal.

        This is the only route in the API whose purpose is to destroy data. It should be
        harder to invoke than the ones that create it.
        """
        body = (await client.post("/api/storage/prune?older_than_days=30")).json()

        assert body["dry_run"] is True
        assert body["runs"] == 1
        assert body["gpu_samples"] == 4

        still_there = (await client.get("/api/storage")).json()
        assert still_there["gpu_samples"] == 4

    async def test_confirming_actually_prunes(
        self, client: httpx.AsyncClient, old_run_with_telemetry: Run
    ) -> None:
        body = (await client.post("/api/storage/prune?older_than_days=30&confirm=true")).json()

        assert body["dry_run"] is False
        assert body["runs"] == 1

        after = (await client.get("/api/storage")).json()
        assert after["gpu_samples"] == 0
        assert after["engine_samples"] == 0
        # And the run is still there, still saying what produced it.
        assert after["runs_pruned"] == 1

    async def test_the_run_and_its_result_survive(
        self, client: httpx.AsyncClient, session: AsyncSession, old_run_with_telemetry: Run
    ) -> None:
        """The measurement is not what retention is for.

        Worth asserting through the HTTP surface as well as the database, because this is
        the path a person will actually take, and a route that quietly cascaded into the
        run would be a much worse bug than one that failed to delete anything.
        """
        run_id = old_run_with_telemetry.id
        await client.post("/api/storage/prune?older_than_days=30&confirm=true")

        detail = (await client.get(f"/api/runs/{run_id}")).json()
        assert detail["status"] == RunStatus.SUCCEEDED
        assert detail["summary"] is not None
        assert detail["config_hash"]

    async def test_a_horizon_below_one_day_is_refused(self, client: httpx.AsyncClient) -> None:
        """0 would read as "delete everything" to a caller and "off" to the settings.

        The two readings differ by the entire telemetry history, so the ambiguous value is
        rejected at the edge rather than resolved by guessing.
        """
        response = await client.post("/api/storage/prune?older_than_days=0&confirm=true")
        assert response.status_code == 422


class TestPrunedIsNotMissing:
    async def test_the_telemetry_endpoint_says_why_it_is_empty(
        self, client: httpx.AsyncClient, old_run_with_telemetry: Run
    ) -> None:
        """The reason the pruning is recorded rather than merely done.

        A chart with no series has two readings — the policy removed it, or sampling has
        been failing silently — and they call for opposite responses. Without this the UI
        would show the same "no telemetry" message for both, and someone would eventually
        spend an afternoon debugging a sampler that works fine.
        """
        run_id = old_run_with_telemetry.id
        before = (await client.get(f"/api/runs/{run_id}/telemetry")).json()
        assert before["sample_count"] == 8
        assert before["pruned_at"] is None

        await client.post("/api/storage/prune?older_than_days=30&confirm=true")

        after = (await client.get(f"/api/runs/{run_id}/telemetry")).json()
        assert after["sample_count"] == 0
        assert after["pruned_at"] is not None
        assert after["pruned_horizon_days"] == 30

    async def test_a_run_that_simply_has_none_says_nothing(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        """The distinction is worthless if the marker appears on everything."""
        host = await world.host("bare")
        run = await world.run(host, await world.config("c"), await world.a_workload())

        body = (await client.get(f"/api/runs/{run.id}/telemetry")).json()

        assert body["sample_count"] == 0
        assert body["pruned_at"] is None
