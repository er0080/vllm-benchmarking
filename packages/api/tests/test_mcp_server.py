"""The MCP surface, exercised through a real MCP client over the mounted transport.

Through the client rather than by calling the tool functions directly, because the thing
most likely to be wrong is not a tool body — those are thin wrappers over router functions
the REST tests already cover — but the mount: transport, session manager lifespan, and
authentication. Calling the functions in-process would test none of that and would pass
whether or not the surface is reachable at all.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

import httpx
import httpx2
import pytest
from fastapi import FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from vllmbench_api.mcp_server import DEFAULT_PAGE, MAX_PAGE, StaticTokenVerifier, _page, _source
from vllmbench_api.settings import ApiSettings

pytestmark = pytest.mark.integration

TOKEN = "mcp-test-token-not-a-real-secret"


class TestPageCeiling:
    """Context economy: an agent pays for every token it reads."""

    def test_default_when_unasked(self) -> None:
        assert _page(None) == DEFAULT_PAGE

    def test_a_large_request_is_capped(self) -> None:
        # An agent asking for everything is usually defaulting rather than choosing.
        assert _page(100_000) == MAX_PAGE

    def test_zero_and_negative_become_one(self) -> None:
        assert _page(0) == 1
        assert _page(-5) == 1


class TestSourceParsing:
    def test_the_two_populations(self) -> None:
        assert _source("real").value == "real"
        assert _source("synthetic").value == "synthetic"

    def test_anything_else_is_an_error_not_a_default(self) -> None:
        """Silently defaulting would hand back a population nobody asked about.

        Invariant 7 is the reason: an agent that asked for "all" and received real
        measurements has been answered a different question than it put.
        """
        with pytest.raises(ValueError, match="no value meaning both"):
            _source("all")


class TestTokenVerifier:
    async def test_the_configured_token_is_accepted(self) -> None:
        verified = await StaticTokenVerifier(TOKEN).verify_token(TOKEN)
        assert verified is not None and verified.client_id == "vllmbench-mcp"

    async def test_a_wrong_token_is_rejected(self) -> None:
        assert await StaticTokenVerifier(TOKEN).verify_token("nope") is None

    async def test_an_unconfigured_server_accepts_nothing(self) -> None:
        """The failure mode a misconfigured deployment would otherwise have.

        An empty configured token must reject every caller, including the empty one, so
        that enabling MCP without setting a token yields an inert surface rather than an
        open one.
        """
        verifier = StaticTokenVerifier("")
        assert await verifier.verify_token("") is None
        assert await verifier.verify_token("anything") is None


@contextlib.asynccontextmanager
async def _app(**overrides: object) -> AsyncIterator[FastAPI]:
    """A fresh app with MCP configured as asked, run through its real lifespan.

    Fresh each time because the mount happens *inside* lifespan: reusing the module-level
    app would stack a second /mcp mount on the first, and the second would never be
    reached.
    """
    import vllmbench_api.main as main_module

    fields: dict[str, object] = {"mcp_enabled": True, "mcp_token": TOKEN, **overrides}
    settings = ApiSettings(**fields)  # type: ignore[arg-type]
    original = ApiSettings
    main_module.ApiSettings = lambda: settings  # type: ignore[assignment,return-value]
    app = FastAPI(lifespan=main_module.lifespan)
    try:
        async with LifespanManager(app):
            yield app
    finally:
        main_module.ApiSettings = original  # type: ignore[assignment]


class LifespanManager:
    """Runs an ASGI app's lifespan around a block.

    The MCP mount and its session manager both live in lifespan, so a test that skips it
    exercises an app where the surface was never attached — which would pass for the wrong
    reason.
    """

    def __init__(self, app: FastAPI) -> None:
        self._app = app
        self._cm = app.router.lifespan_context(app)

    async def __aenter__(self) -> FastAPI:
        await self._cm.__aenter__()
        return self._app

    async def __aexit__(self, *exc: object) -> None:
        await self._cm.__aexit__(*exc)  # type: ignore[arg-type]


@contextlib.asynccontextmanager
async def _client(app: FastAPI, token: str = TOKEN) -> AsyncIterator[ClientSession]:
    """An MCP client speaking to the app in-process, over its real transport.

    In-process rather than over a socket, but through the actual Streamable HTTP
    transport: the mount, the session manager and the bearer check are all exercised.
    """
    transport = httpx2.ASGITransport(app=app)
    http = httpx2.AsyncClient(
        transport=transport,
        base_url="http://api",
        headers={"Authorization": f"Bearer {token}"},
    )
    async with http:
        async with streamable_http_client("http://api/mcp", http_client=http) as streams:
            read_stream, write_stream, *_ = streams
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session


async def test_the_surface_is_absent_unless_enabled() -> None:
    """Off by default. ADR 0001 puts MCP on this service; the flag is what keeps that
    from meaning always exposed."""
    async with _app(mcp_enabled=False) as app:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
            response = await client.post("/mcp", json={})
            assert response.status_code == 404


async def test_tools_are_listed_and_callable_over_the_transport() -> None:
    async with _app() as app:
        async with _client(app) as session:
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            # The read surface the roadmap names for 0.6.0.
            assert {
                "list_hosts",
                "list_configs",
                "get_config",
                "list_workloads",
                "list_sweeps",
                "get_sweep",
                "query_runs",
                "get_run",
                "get_pareto",
                "compare_runs",
                "get_run_telemetry",
                "server_info",
            } <= names
            # Every tool carries a description; an agent choosing between them has
            # nothing else to go on.
            assert all(tool.description for tool in listed.tools)

            result = await session.call_tool("server_info", {})
            # A dict return is delivered as structured content directly, not wrapped.
            assert result.structured_content is not None
            assert result.structured_content["populations"] == ["real", "synthetic"]
            assert result.structured_content["protocol_version"] > 0


async def test_a_bad_token_is_refused() -> None:
    async with _app() as app:
        with pytest.raises(Exception):  # noqa: B017 - the transport fails before a session
            async with _client(app, token="wrong") as session:
                await session.list_tools()


async def test_a_tool_reads_through_to_the_database() -> None:
    """One tool exercised all the way to Postgres and back.

    server_info touches nothing, so on its own it proves the transport works and not that
    the tools do: the session factory is handed in at build time and a mistake there would
    fail every data tool while leaving the surface apparently healthy.
    """
    from vllmbench_db.models import GpuHost
    from vllmbench_db.session import create_engine, create_session_factory
    from vllmbench_db.testing import reset_database, test_database_url

    engine = create_engine(test_database_url())
    try:
        await reset_database(engine)
        async with create_session_factory(engine)() as db:
            db.add(
                GpuHost(
                    name="mcp-host",
                    agent_url="http://agent",
                    gpu_count=2,
                    synthetic_source="mock_agent",
                )
            )
            await db.commit()
    finally:
        await engine.dispose()

    async with _app() as app:
        async with _client(app) as session:
            result = await session.call_tool("list_hosts", {})

    assert result.structured_content is not None
    (host,) = result.structured_content["result"]
    assert host["name"] == "mcp-host"
    # Carried on every host so an agent never has to infer it: anything this host produces
    # is quarantined from real measurements (invariant 7).
    assert host["synthetic_source"] == "mock_agent"
