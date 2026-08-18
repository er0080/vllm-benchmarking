"""Shared builders for tests that need a populated database.

Extracted so that the four test modules needing a host, a config and a run do not each
grow their own slightly different version — the kind of duplication that ends with two
files disagreeing about what a "normal" run looks like, and a test passing for the wrong
reason.

A ``conftest.py`` so the fixtures arrive by name without being imported. Importing a
fixture and then naming it as a test parameter shadows the import, which every linter
flags — a hundred times over, once per test.

``World`` itself lives in ``api_world`` — see the note there on why it cannot live
here.

Runs are inserted already terminal rather than transitioned into it: a run in
``succeeded`` is immutable by database trigger, so building one any other way would be
fighting the schema for no benefit.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from api_world import World
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_api.main import app as api_app
from vllmbench_api.settings import ApiSettings
from vllmbench_db.session import create_engine, create_session_factory
from vllmbench_db.testing import reset_database, test_database_url


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


@pytest.fixture
def world(session: AsyncSession) -> World:
    return World(session)
