"""Saved analysis views, end to end.

The property under test is that a view stores a *query* and reopens as one — including
that the population it was saved over is the population it comes back with, which is
invariant 7 reaching the one part of the UI that persists a selection.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from vllmbench_api.main import app as api_app
from vllmbench_api.settings import ApiSettings
from vllmbench_db.session import create_engine, create_session_factory
from vllmbench_db.testing import reset_database, test_database_url

pytestmark = pytest.mark.integration

VIEWS = "/api/views"


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    engine = create_engine(test_database_url())
    factory = create_session_factory(engine)
    api_app.state.engine = engine
    api_app.state.sessions = factory
    api_app.state.settings = ApiSettings(token="test-token-not-a-real-secret")

    # reset_database covers saved_view too: its delete order is derived from the mapped
    # metadata, so a new table is included without anyone remembering to add it.
    await reset_database(engine)

    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as c:
        yield c
    await engine.dispose()


async def _save(client: httpx.AsyncClient, **overrides: object) -> dict:
    body = {
        "name": "tp sweep, real",
        "view": "pareto",
        "source": "real",
        "filters": {"sweepId": "abc", "tensorParallelSizes": [1, 2]},
        "options": {"paretoX": "total_token_throughput_per_gpu"},
    }
    body.update(overrides)
    response = await client.post(VIEWS, json=body)
    assert response.status_code == 201, response.text
    return response.json()


async def test_a_view_round_trips_its_selection(client: httpx.AsyncClient) -> None:
    saved = await _save(client)
    (listed,) = (await client.get(VIEWS)).json()

    assert listed["id"] == saved["id"]
    assert listed["view"] == "pareto"
    assert listed["filters"] == {"sweepId": "abc", "tensorParallelSizes": [1, 2]}
    assert listed["options"]["paretoX"] == "total_token_throughput_per_gpu"


async def test_it_stores_no_run_ids(client: httpx.AsyncClient) -> None:
    """A view is a query, not a snapshot.

    Reopening one next month must include the runs measured since. Pinning run ids would
    make it stop tracking reality while continuing to look current — worse than either a
    live view or an obvious snapshot.
    """
    saved = await _save(client)
    assert "run_ids" not in saved
    assert "run_ids" not in saved["filters"]


async def test_the_population_survives_the_round_trip(client: httpx.AsyncClient) -> None:
    # A view saved over the mock agent's runs must not reopen showing real hardware.
    await _save(client, name="mock work", source="synthetic")
    (listed,) = (await client.get(VIEWS)).json()
    assert listed["source"] == "synthetic"


async def test_an_unknown_population_is_refused(client: httpx.AsyncClient) -> None:
    # Otherwise it is stored and later reopened as something no filter can express.
    response = await client.post(
        VIEWS, json={"name": "bad", "view": "pareto", "source": "everything"}
    )
    assert response.status_code == 422
    assert "real" in response.json()["detail"]


async def test_duplicate_names_are_a_conflict(client: httpx.AsyncClient) -> None:
    await _save(client)
    response = await client.post(
        VIEWS, json={"name": "tp sweep, real", "view": "scaling", "source": "real"}
    )
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


async def test_delete_removes_only_the_bookmark(client: httpx.AsyncClient) -> None:
    saved = await _save(client)
    assert (await client.delete(f"{VIEWS}/{saved['id']}")).status_code == 204
    assert (await client.get(VIEWS)).json() == []


async def test_deleting_an_unknown_view_is_404(client: httpx.AsyncClient) -> None:
    import uuid

    response = await client.delete(f"{VIEWS}/{uuid.uuid4()}")
    assert response.status_code == 404


async def test_newest_first(client: httpx.AsyncClient) -> None:
    await _save(client, name="first")
    await _save(client, name="second")
    names = [v["name"] for v in (await client.get(VIEWS)).json()]
    assert names == ["second", "first"]
