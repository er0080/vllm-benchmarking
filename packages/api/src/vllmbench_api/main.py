"""FastAPI control plane.

Milestone 0.1.0 scope: prove the service starts, reaches Postgres, and reports its
version. Domain endpoints arrive with the host registry and the first run.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from sqlalchemy import text

from vllmbench_api.routers import hosts, runs, sweeps
from vllmbench_api.settings import ApiSettings
from vllmbench_db.schema_version import check_schema_version
from vllmbench_db.session import create_engine, create_session_factory
from vllmbench_protocol import PROTOCOL_VERSION, __version__

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    engine = create_engine()
    app.state.engine = engine
    app.state.sessions = create_session_factory(engine)
    app.state.settings = ApiSettings()

    # Checked once, at startup. A schema behind the code fails later as an opaque
    # Postgres type error in whatever request happens to write first.
    try:
        app.state.schema = await check_schema_version(engine)
    except Exception as exc:
        log.warning("could not check schema version: %s", exc)
        app.state.schema = None

    log.info("api %s (protocol %d) started", __version__, PROTOCOL_VERSION)
    try:
        yield
    finally:
        await engine.dispose()


app = FastAPI(
    title="vLLM Benchmarking",
    version=__version__,
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

app.include_router(hosts.router)
app.include_router(runs.router)
app.include_router(sweeps.router)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Liveness plus a real database round trip.

    Deliberately not a bare 200. A control plane that answers "healthy" while unable to
    reach Postgres would let a sweep start and then fail at the moment it has results to
    write, which is the most expensive point at which to discover the problem.
    """
    database_ok = False
    detail: str | None = None
    try:
        async with app.state.sessions() as session:
            await session.execute(text("SELECT 1"))
        database_ok = True
    except Exception as exc:
        detail = str(exc)
        log.warning("health check: database unreachable: %s", exc)

    schema = getattr(app.state, "schema", None)
    schema_ok = schema is None or schema.ok

    return {
        # Degraded rather than ok on a stale schema: the service answers, but writes
        # will fail, and reporting healthy would hide that until the first run.
        "status": "ok" if database_ok and schema_ok else "degraded",
        "version": __version__,
        "protocol_version": PROTOCOL_VERSION,
        "database": {"ok": database_ok, "detail": detail},
        "schema": None
        if schema is None
        else {"ok": schema.ok, "applied": schema.applied, "expected": schema.expected},
    }


@app.get("/api/version")
async def version() -> dict[str, Any]:
    return {"version": __version__, "protocol_version": PROTOCOL_VERSION}
