"""Fixtures for database integration tests.

These require a real PostgreSQL: the invariants under test are enforced by triggers and
check constraints, so SQLite or a mock would test nothing that ships.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_db.enums import InitiatedBy, RunStatus
from vllmbench_db.models import GpuHost, Run, ServerConfig, Workload
from vllmbench_db.session import create_engine, create_session_factory
from vllmbench_db.testing import test_database_url

pytestmark = pytest.mark.integration


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_engine(test_database_url())
    factory = create_session_factory(engine)
    async with factory() as s:
        yield s
        await s.rollback()
    await engine.dispose()


@pytest.fixture
async def host(session: AsyncSession) -> GpuHost:
    host = GpuHost(name=f"test-host-{uuid.uuid4().hex[:8]}", agent_url="http://10.0.0.1:9110")
    session.add(host)
    await session.flush()
    return host


@pytest.fixture
async def config(session: AsyncSession) -> ServerConfig:
    digest = uuid.uuid4().hex * 2
    config = ServerConfig(config_hash=digest[:64], name="test-config", yaml="model: opt-125m\n")
    session.add(config)
    await session.flush()
    return config


@pytest.fixture
async def workload(session: AsyncSession) -> Workload:
    digest = uuid.uuid4().hex * 2
    workload = Workload(
        workload_hash=digest[:64],
        name="test-workload",
        dataset_name="random",
        num_prompts=100,
    )
    session.add(workload)
    await session.flush()
    return workload


@pytest.fixture
async def run_factory(
    session: AsyncSession, host: GpuHost, config: ServerConfig, workload: Workload
):
    async def _make(**overrides: object) -> Run:
        fields: dict[str, object] = {
            "server_config_id": config.id,
            "workload_id": workload.id,
            "gpu_host_id": host.id,
            "config_hash": config.config_hash,
            "workload_hash": workload.workload_hash,
            "initiated_by": InitiatedBy.API,
            "status": RunStatus.QUEUED,
            "started_at": dt.datetime.now(dt.UTC),
        }
        fields.update(overrides)
        run = Run(**fields)
        session.add(run)
        await session.flush()
        return run

    return _make
