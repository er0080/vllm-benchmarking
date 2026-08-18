"""FastAPI control plane.

Milestone 0.1.0 scope: prove the service starts, reaches Postgres, and reports its
version. Domain endpoints arrive with the host registry and the first run.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from sqlalchemy import text

from vllmbench_api.mcp_server import build_mcp_server, mount_mcp
from vllmbench_api.routers import (
    analysis,
    audit,
    hosts,
    imports,
    runs,
    storage,
    sweeps,
    views,
)
from vllmbench_api.settings import ApiSettings
from vllmbench_db.schema_version import check_schema_version
from vllmbench_db.session import create_engine, create_session_factory, database_password
from vllmbench_protocol import PROTOCOL_VERSION, __version__
from vllmbench_protocol.logging import bound, configure_logging

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    engine = create_engine()
    app.state.engine = engine
    app.state.sessions = create_session_factory(engine)
    settings = ApiSettings()
    app.state.settings = settings

    # Both tokens and the database password, registered before the first request. The
    # database password is the one that matters most here: nobody ever writes a log line
    # containing it, but a connection failure quotes the whole DSN back from inside the
    # driver — including in the health check three lines below.
    configure_logging(
        "api",
        secrets=[settings.token, settings.mcp_token, database_password()],
    )

    # Checked once, at startup. A schema behind the code fails later as an opaque
    # Postgres type error in whatever request happens to write first.
    try:
        app.state.schema = await check_schema_version(engine)
    except Exception as exc:
        log.warning("could not check schema version: %s", exc)
        app.state.schema = None

    log.info("api %s (protocol %d) started", __version__, PROTOCOL_VERSION)

    if not settings.mcp_enabled:
        # Off unless asked for. ADR 0001 puts the MCP surface on this service rather than
        # a separate one, and this flag is what keeps that from meaning "always exposed".
        try:
            yield
        finally:
            await engine.dispose()
        return

    # Built here rather than at import time so the flag is read from the environment the
    # service actually starts in, and so a disabled deployment does not construct it.
    server = build_mcp_server(app.state.sessions, settings)
    mount_mcp(app, server, settings)
    if not settings.mcp_token:
        # Loud, because the alternative is a surface that authenticates nothing. The
        # verifier refuses every token in this state, so the mount is inert rather than
        # open — but an operator who set MCP_ENABLED expects it to work.
        log.warning("MCP is enabled but VLLMBENCH_MCP_TOKEN is unset; every call will 401")
    log.info("MCP mounted at /mcp")

    try:
        # The session manager owns the transport's task group; mounting the app is not
        # enough on its own, and without this every call fails once, obscurely.
        async with server.session_manager.run():
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


@app.middleware("http")
async def bind_request_context(request: Request, call_next: Any) -> Any:
    """Tag everything logged while serving one request.

    The path and method rather than a generated request id: this control plane serves a
    browser and a handful of agents, not a fleet, and "which endpoint was slow" is the
    question that actually gets asked. A correlation id would be the right answer for a
    service behind a load balancer, and is worth adding the day there is one.
    """
    with bound(method=request.method, path=request.url.path):
        return await call_next(request)


app.include_router(analysis.router)
app.include_router(audit.router)
app.include_router(hosts.router)
app.include_router(imports.router)
app.include_router(runs.router)
app.include_router(storage.router)
app.include_router(sweeps.router)
app.include_router(views.router)


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
