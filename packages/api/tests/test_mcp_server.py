"""The MCP surface, exercised through a real MCP client over the mounted transport.

Through the client rather than by calling the tool functions directly, because the thing
most likely to be wrong is not a tool body — those are thin wrappers over router functions
the REST tests already cover — but the mount: transport, session manager lifespan, and
authentication. Calling the functions in-process would test none of that and would pass
whether or not the surface is reachable at all.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator

import httpx
import httpx2
import pytest
from fastapi import FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from vllmbench_api.mcp_server import DEFAULT_PAGE, MAX_PAGE, StaticTokenVerifier, _page, _source
from vllmbench_api.settings import ApiSettings
from vllmbench_db.testing import test_database_url

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


async def test_write_tools_record_that_an_agent_asked() -> None:
    """A full author-a-sweep round trip over MCP, checking the provenance it leaves.

    Invariant 6: a run must be able to state what produced it. Before this surface
    existed every run claimed the UI and the claim happened to be true; the moment a
    second interface could create work, that became a wrong answer sitting in a
    provenance column.
    """
    from vllmbench_db.models import GpuHost, Run
    from vllmbench_db.session import create_engine, create_session_factory
    from vllmbench_db.testing import reset_database, test_database_url

    engine = create_engine(test_database_url())
    try:
        await reset_database(engine)
        factory = create_session_factory(engine)
        async with factory() as db:
            host = GpuHost(name="mcp-write-host", agent_url="http://agent", gpu_count=2)
            db.add(host)
            await db.commit()
            host_id = str(host.id)

        async with _app() as app:
            async with _client(app) as session:
                config = await session.call_tool(
                    "create_config",
                    {"name": "from-mcp", "yaml": "model: m\ntensor-parallel-size: 1\n"},
                )
                workload = await session.call_tool(
                    "create_workload", {"name": "c8", "max_concurrency": 8}
                )
                assert config.structured_content and workload.structured_content

                sweep = await session.call_tool(
                    "create_sweep",
                    {
                        "name": "agent sweep",
                        "gpu_host_id": host_id,
                        "config_hashes": [config.structured_content["config_hash"]],
                        "workload_hashes": [workload.structured_content["workload_hash"]],
                        "replicates": 2,
                    },
                )
                assert sweep.structured_content is not None
                assert sweep.structured_content["total_runs"] == 2
                # Most of a sweep's wall clock is model loading, so this is the number
                # that predicts how long it takes.
                assert sweep.structured_content["engine_starts"] == 1

        async with factory() as db:
            from sqlalchemy import select

            runs = list((await db.execute(select(Run))).scalars())
        assert runs and all(str(run.initiated_by) == "mcp" for run in runs)
        assert all(run.initiated_by_client == "mcp" for run in runs)
    finally:
        await engine.dispose()


async def test_write_tools_refuse_when_switched_off() -> None:
    """Read stays available when writes are off.

    A control plane an operator wants an agent to look at but not touch is a real
    configuration, and it should not mean turning the whole surface off.
    """
    async with _app(mcp_write_enabled=False) as app:
        async with _client(app) as session:
            listed = await session.list_tools()
            # Still advertised: an agent should be told the tool exists and is refused,
            # rather than concluding this server cannot create configs at all.
            assert "create_config" in {tool.name for tool in listed.tools}

            result = await session.call_tool(
                "create_config", {"name": "nope", "yaml": "model: m\n"}
            )
            assert result.is_error
            assert "disabled" in str(result.content)

            # ...and reading is unaffected.
            info = await session.call_tool("server_info", {})
            assert info.structured_content is not None
            assert info.structured_content["write_tools_enabled"] is False


async def test_no_tool_can_mutate_or_delete_a_run() -> None:
    """Runs are immutable once terminal, and the surface offers no way to try.

    Enforced by a database trigger regardless, but a tool that existed and always failed
    would still be an invitation.
    """
    async with _app() as app:
        async with _client(app) as session:
            names = {tool.name for tool in (await session.list_tools()).tools}
    forbidden = {n for n in names if any(w in n for w in ("delete", "update", "edit", "remove"))}
    assert forbidden == set(), f"unexpected mutating tools: {forbidden}"


# ---------------------------------------------------------------------------
# Resources
#
# Both are content the client can cache by URI, which is only safe because both are
# immutable: a config's hash *is* its text, and a finished sweep's runs cannot change.
# ---------------------------------------------------------------------------


async def _seeded_sweep(synthetic: bool = False) -> tuple[str, str, str]:
    """A finished sweep with two measured points, one dominating the other.

    Returns (sweep_id, config_hash, yaml). Runs are inserted already terminal — a
    succeeded run is immutable by trigger, so building one any other way fights the
    schema for nothing.
    """
    import datetime as dt
    import os

    from vllmbench_db.enums import InitiatedBy, ReplicateOrder, RunStatus, SweepStatus
    from vllmbench_db.models import (
        GpuHost,
        Run,
        RunSummary,
        ServerConfig,
        Sweep,
        Workload,
    )
    from vllmbench_db.session import create_engine, create_session_factory
    from vllmbench_db.testing import reset_database, test_database_url

    yaml = "model: Qwen/Qwen3.5-9B\ntensor-parallel-size: 1\nmax-model-len: 8192\n"
    engine = create_engine(test_database_url())
    try:
        await reset_database(engine)
        async with create_session_factory(engine)() as db:
            host = GpuHost(
                name="ubuntu-llm",
                agent_url="http://agent",
                gpu_count=2,
                synthetic_source="mock_agent" if synthetic else None,
            )
            config = ServerConfig(config_hash=os.urandom(32).hex(), name="qwen-9b", yaml=yaml)
            db.add_all([host, config])
            await db.flush()

            sweep = Sweep(
                name="first real measurements",
                description="TP 1 and 2 across three concurrencies.",
                gpu_host_id=host.id,
                status=SweepStatus.SUCCEEDED,
                replicates=2,
                replicate_order=ReplicateOrder.GROUPED,
                initiated_by=InitiatedBy.MCP,
                is_synthetic=synthetic,
            )
            db.add(sweep)
            await db.flush()

            # Two workloads so the two points differ in something the table shows, and
            # so the dominated one is dominated for a legible reason.
            for name, concurrency, per_gpu, tpot in (
                ("c4", 4, 700.0, 20.0),
                ("c64", 64, 500.0, 40.0),
            ):
                workload = Workload(
                    workload_hash=os.urandom(32).hex(),
                    name=name,
                    dataset_name="random",
                    num_prompts=64,
                    max_concurrency=concurrency,
                )
                db.add(workload)
                await db.flush()
                for replicate in range(2):
                    run = Run(
                        sweep_id=sweep.id,
                        replicate_idx=replicate,
                        server_config_id=config.id,
                        workload_id=workload.id,
                        gpu_host_id=host.id,
                        status=RunStatus.SUCCEEDED,
                        finished_at=dt.datetime.now(dt.UTC),
                        config_hash=config.config_hash,
                        workload_hash=workload.workload_hash,
                        vllm_version="0.25.1",
                        gpu_model="NVIDIA GeForce RTX 3090",
                        driver_version="580.95.05",
                        gpu_count=1,
                        tensor_parallel_size=1,
                        is_synthetic=synthetic,
                        synthetic_source="mock_agent" if synthetic else None,
                        initiated_by=InitiatedBy.MCP,
                    )
                    db.add(run)
                    await db.flush()
                    db.add(
                        RunSummary(
                            run_id=run.id,
                            tpot_ms_mean=tpot,
                            tpot_ms_p99=tpot * 2,
                            total_token_throughput_per_gpu=per_gpu + replicate * 10,
                            output_token_throughput_per_gpu=per_gpu / 2,
                            total_token_throughput_tok_sec=per_gpu + replicate * 10,
                            ttft_ms_p99=120.0,
                        )
                    )
            await db.commit()
            return str(sweep.id), config.config_hash, yaml
    finally:
        await engine.dispose()


async def test_resource_templates_are_advertised() -> None:
    async with _app() as app:
        async with _client(app) as session:
            listed = await session.list_resource_templates()
            templates = {t.uri_template for t in listed.resource_templates}
    assert "vllmbench://config/{config_hash}" in templates
    assert "vllmbench://sweep/{sweep_id}/report" in templates


async def test_a_config_resource_is_the_exact_yaml() -> None:
    """Byte for byte, with no envelope.

    Invariant 5: what is stored is what gets passed to `vllm serve --config`. An agent
    that reads this resource and writes it to a file has the configuration that ran, and
    a JSON wrapper or a re-emitted YAML would break that in a way nothing would notice
    until the engine came up differently.
    """
    _, config_hash, yaml = await _seeded_sweep()

    async with _app() as app:
        async with _client(app) as session:
            result = await session.read_resource(f"vllmbench://config/{config_hash}")

    (content,) = result.contents
    assert getattr(content, "text", None) == yaml
    assert content.mime_type == "text/yaml"


async def test_an_unknown_config_hash_is_an_error() -> None:
    """It fails rather than returning an empty document.

    The SDK replaces the reason with a generic one on the way out, so an agent that
    needs to know *why* should use the `get_config` tool, which says. What matters here
    is that a hash nobody stored never reads as a config with no settings in it.
    """
    async with _app() as app:
        async with _client(app) as session:
            with pytest.raises(Exception, match=r"config/0{64}"):
                await session.read_resource("vllmbench://config/" + "0" * 64)


async def test_a_sweep_report_carries_the_frontier_and_its_provenance() -> None:
    """The report has to be readable *and* not invite an invalid comparison."""
    sweep_id, _, _ = await _seeded_sweep()

    async with _app() as app:
        async with _client(app) as session:
            result = await session.read_resource(f"vllmbench://sweep/{sweep_id}/report")

    (content,) = result.contents
    report = getattr(content, "text", "")

    assert "first real measurements" in report
    # Provenance in the section heading, not a footnote: the reason two tables exist is
    # that they may not be read as one.
    assert "NVIDIA GeForce RTX 3090" in report
    assert "vLLM 0.25.1" in report
    # The dominated point is present, and the frontier point is marked.
    assert "c4" in report and "c64" in report
    assert "★" in report
    # Per-GPU throughput is the leftmost metric column (invariant 8).
    assert report.index("tok/s per GPU") < report.index("tok/s aggregate")
    # A spread accompanies every number that has one, and says what it means.
    assert "±" in report
    assert "not a result" in report


async def test_a_synthetic_sweep_report_leads_with_the_quarantine() -> None:
    """Invariant 7, in the one place an agent cannot skim past it.

    A synthetic result that reads like a measurement is the failure this flag exists to
    prevent, and an agent reading a report will act on the first thing that looks like a
    number unless told otherwise first.
    """
    sweep_id, _, _ = await _seeded_sweep(synthetic=True)

    async with _app() as app:
        async with _client(app) as session:
            result = await session.read_resource(f"vllmbench://sweep/{sweep_id}/report")

    report = getattr(result.contents[0], "text", "")
    banner = report.index("not measurements of any real hardware")
    # Before any measurement in the document.
    assert banner < report.index("tok/s per GPU")


# ---------------------------------------------------------------------------
# The write audit log
# ---------------------------------------------------------------------------


async def _audit_rows() -> list[object]:
    from sqlalchemy import select

    from vllmbench_db.models import McpWriteAudit
    from vllmbench_db.session import create_engine, create_session_factory
    from vllmbench_db.testing import test_database_url

    engine = create_engine(test_database_url())
    try:
        async with create_session_factory(engine)() as db:
            return list(
                (
                    await db.execute(select(McpWriteAudit).order_by(McpWriteAudit.called_at))
                ).scalars()
            )
    finally:
        await engine.dispose()


async def _empty_database() -> None:
    from vllmbench_db.session import create_engine
    from vllmbench_db.testing import reset_database, test_database_url

    engine = create_engine(test_database_url())
    try:
        await reset_database(engine)
    finally:
        await engine.dispose()


async def test_a_successful_write_is_recorded_with_what_it_produced() -> None:
    await _empty_database()

    async with _app() as app:
        async with _client(app) as session:
            await session.call_tool(
                "create_config", {"name": "audited", "yaml": "model: m\n", "notes": "why"}
            )

    (row,) = await _audit_rows()
    assert row.tool == "create_config"  # type: ignore[attr-defined]
    assert row.outcome == "succeeded"  # type: ignore[attr-defined]
    assert row.client == "mcp"  # type: ignore[attr-defined]
    # Enough to join the record to the thing it made.
    assert row.subject and len(row.subject) == 64  # type: ignore[attr-defined]
    # The arguments as they arrived, so a later mishandling can be diagnosed against
    # what was actually asked for rather than what we assume was.
    assert row.arguments["name"] == "audited"  # type: ignore[attr-defined]
    assert row.arguments["yaml"] == "model: m\n"  # type: ignore[attr-defined]


async def test_a_refusal_is_recorded_because_nothing_else_records_it() -> None:
    """The rows that justify the table.

    A refused call writes nothing to any other table. Without this the only evidence it
    happened lives in the agent's context, which does not survive the session — and an
    agent bouncing repeatedly off a read-only control plane is precisely what an operator
    needs to be able to see.
    """
    await _empty_database()

    async with _app(mcp_write_enabled=False) as app:
        async with _client(app) as session:
            result = await session.call_tool("create_config", {"name": "no", "yaml": "model: m\n"})
            assert result.is_error

    (row,) = await _audit_rows()
    assert row.outcome == "refused"  # type: ignore[attr-defined]
    assert "disabled" in (row.error or "")  # type: ignore[attr-defined]
    assert row.subject is None  # type: ignore[attr-defined]


async def test_a_domain_refusal_records_the_reason() -> None:
    """A refusal the surface understood, told apart from a bug.

    Reading the log back, a hundred refusals is an agent that needs better instructions
    and one failure is something to fix; conflating them makes both invisible.
    """
    await _empty_database()

    async with _app() as app:
        async with _client(app) as session:
            result = await session.call_tool(
                "create_sweep",
                {
                    "name": "doomed",
                    "gpu_host_id": str(uuid.uuid4()),
                    "config_hashes": ["0" * 64],
                    "workload_hashes": ["1" * 64],
                },
            )
            assert result.is_error

    (row,) = await _audit_rows()
    assert row.tool == "create_sweep"  # type: ignore[attr-defined]
    assert row.outcome == "refused"  # type: ignore[attr-defined]
    assert "unknown config or workload" in (row.error or "")  # type: ignore[attr-defined]


async def test_reads_are_not_audited() -> None:
    """This log is about what changed, not about what was looked at.

    Recording every query would bury the writes in noise and grow without bound while an
    agent polls a running sweep.
    """
    await _empty_database()

    async with _app() as app:
        async with _client(app) as session:
            await session.call_tool("list_hosts", {})
            await session.call_tool("query_runs", {})
            await session.call_tool("server_info", {})

    assert await _audit_rows() == []


async def test_an_outsized_argument_is_trimmed_not_dropped() -> None:
    """One call must not be able to grow this table without bound.

    Trimmed rather than omitted: a truncated config still identifies what was submitted,
    where a missing key looks like it was never sent.
    """
    from vllmbench_api.mcp_server import MAX_LOGGED_VALUE

    await _empty_database()
    huge = "model: m\n" + ("# padding\n" * 2000)
    assert len(huge) > MAX_LOGGED_VALUE

    async with _app() as app:
        async with _client(app) as session:
            await session.call_tool("create_config", {"name": "big", "yaml": huge})

    (row,) = await _audit_rows()
    logged = row.arguments["yaml"]  # type: ignore[attr-defined]
    assert logged.startswith("model: m\n")
    assert str(len(huge)) in logged
    assert len(logged) < len(huge)


async def test_the_log_is_readable_and_filterable() -> None:
    """Written by the MCP surface, read over HTTP.

    Deliberately tested as one path: a log nothing can read is not an audit trail, and
    the operator who needs it is looking at the control plane, not connected as an agent.
    """
    from vllmbench_api.main import app as rest_app
    from vllmbench_db.session import create_engine, create_session_factory

    await _empty_database()

    async with _app() as app:
        async with _client(app) as session:
            await session.call_tool("create_config", {"name": "kept", "yaml": "model: m\n"})
            await session.call_tool(
                "create_sweep",
                {
                    "name": "doomed",
                    "gpu_host_id": str(uuid.uuid4()),
                    "config_hashes": ["0" * 64],
                    "workload_hashes": ["1" * 64],
                },
            )

    engine = create_engine(test_database_url())
    rest_app.state.engine = engine
    rest_app.state.sessions = create_session_factory(engine)
    rest_app.state.settings = ApiSettings(token="test-token-not-a-real-secret")
    try:
        transport = httpx.ASGITransport(app=rest_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api") as rest:
            everything = (await rest.get("/api/mcp-audit")).json()
            refused = (await rest.get("/api/mcp-audit?outcome=refused")).json()
    finally:
        await engine.dispose()

    assert {row["tool"] for row in everything} == {"create_config", "create_sweep"}
    # Newest first: an operator opening this wants the last thing that happened.
    assert everything[0]["tool"] == "create_sweep"
    # The filter that matters. A refusal writes nothing to any other table, so this is
    # the only view of an agent that has been asking for something it cannot have.
    assert [row["tool"] for row in refused] == ["create_sweep"]
    assert refused[0]["outcome"] == "refused"
