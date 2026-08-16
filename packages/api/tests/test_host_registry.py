"""Host registration, against a real database and a real mock agent.

The failure paths get more attention than the happy path. Registering a host that works
is easy; the value is in refusing to register one that does not, and in saying why
precisely enough that an operator knows which of three different things to fix.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import httpx
import pytest
from sqlalchemy import text

from vllmbench_api.main import app as api_app
from vllmbench_api.settings import ApiSettings
from vllmbench_db.session import create_engine, create_session_factory
from vllmbench_mockagent.main import SYNTHETIC_SOURCE
from vllmbench_mockagent.main import create_app as create_mock_app
from vllmbench_protocol import PROTOCOL_VERSION

pytestmark = pytest.mark.integration

TOKEN = "test-token-not-a-real-secret"
MOCK_URL = "http://mock-agent"
STALE_URL = "http://stale-agent"


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://vllmbench:change-me@localhost:5432/vllmbench",
    )


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[httpx.AsyncClient]:
    """The API, wired to a mock agent reachable only through an in-process transport.

    Routing the agent client at the ASGI mock keeps the test hermetic — no sockets, no
    ports, no ordering dependency on another container — while still exercising the real
    AgentClient, the real handshake, and the real protocol check.
    """
    transports = {
        MOCK_URL: httpx.ASGITransport(app=create_mock_app(token=TOKEN)),
        # A genuinely stale agent: same contract, older protocol number. Closer to the
        # real failure than reaching into module globals to fake it.
        STALE_URL: httpx.ASGITransport(
            app=create_mock_app(token=TOKEN, protocol_version=PROTOCOL_VERSION + 7)
        ),
    }
    real_async_client = httpx.AsyncClient

    def patched(*args: object, **kwargs: object) -> httpx.AsyncClient:
        base_url = str(kwargs.get("base_url", ""))
        for url, transport in transports.items():
            if base_url.startswith(url):
                kwargs["transport"] = transport
                break
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("vllmbench_protocol.client.httpx.AsyncClient", patched)

    engine = create_engine(_database_url())
    api_app.state.engine = engine
    api_app.state.sessions = create_session_factory(engine)
    api_app.state.settings = ApiSettings(token=TOKEN)

    async with engine.begin() as connection:
        await connection.execute(text("DELETE FROM gpu_device"))
        await connection.execute(text("DELETE FROM gpu_host"))

    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as c:
        yield c

    await engine.dispose()


class TestRegistration:
    async def test_registers_a_reachable_host(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/hosts", json={"name": "mock-1", "agent_url": MOCK_URL})
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["gpu_count"] == 2
        assert body["vllm_version"] == "0.25.1"
        assert len(body["devices"]) == 2

    async def test_records_device_inventory_per_device(self, client: httpx.AsyncClient) -> None:
        # Multi-GPU is in scope, so the registry must keep devices individually rather
        # than collapsing them to a count.
        response = await client.post("/api/hosts", json={"name": "mock-2", "agent_url": MOCK_URL})
        devices = response.json()["devices"]
        assert [d["device_index"] for d in devices] == [0, 1]
        assert all(d["vram_bytes"] for d in devices)

    async def test_propagates_the_agents_synthetic_declaration(
        self, client: httpx.AsyncClient
    ) -> None:
        # Invariant 7's chain of custody: the mock declares itself, the control plane
        # records it verbatim. Nothing infers it from how the facts look.
        response = await client.post("/api/hosts", json={"name": "mock-3", "agent_url": MOCK_URL})
        assert response.json()["synthetic_source"] == SYNTHETIC_SOURCE

    async def test_duplicate_name_is_rejected(self, client: httpx.AsyncClient) -> None:
        await client.post("/api/hosts", json={"name": "dup", "agent_url": MOCK_URL})
        second = await client.post("/api/hosts", json={"name": "dup", "agent_url": MOCK_URL})
        assert second.status_code == 409


class TestRegistrationRefusesBadHosts:
    async def test_unreachable_agent_is_not_stored(self, client: httpx.AsyncClient) -> None:
        """The important one.

        Storing a host whose agent cannot be reached would defer the failure to the
        moment a sweep starts. The registry must be a statement that the host worked at
        least once, not a bookmark.
        """
        response = await client.post(
            "/api/hosts",
            json={"name": "ghost", "agent_url": "http://127.0.0.1:9/nope"},
        )
        assert response.status_code == 502
        assert "unreachable" in response.json()["detail"].lower()

        listing = await client.get("/api/hosts")
        assert all(h["name"] != "ghost" for h in listing.json())

    async def test_wrong_token_is_reported_distinctly(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Must be distinguishable from "unreachable": the operator fixes a different
        # thing in each case.
        api_app.state.settings = ApiSettings(token="wrong-token")
        response = await client.post(
            "/api/hosts", json={"name": "bad-token", "agent_url": MOCK_URL}
        )
        assert response.status_code == 502
        assert "token" in response.json()["detail"].lower()
        api_app.state.settings = ApiSettings(token=TOKEN)

    async def test_protocol_mismatch_is_a_conflict_naming_both_versions(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.post("/api/hosts", json={"name": "stale", "agent_url": STALE_URL})

        assert response.status_code == 409
        detail = response.json()["detail"]
        # Both numbers, so an operator knows which side to upgrade.
        assert str(PROTOCOL_VERSION) in detail and str(PROTOCOL_VERSION + 7) in detail

        listing = await client.get("/api/hosts")
        assert all(h["name"] != "stale" for h in listing.json())


class TestVllmVersionPolicy:
    async def test_version_drift_is_reported_but_never_blocks(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLAUDE.md: a vLLM version mismatch warns, never fails.

        The opposite of the protocol check above. Comparing vLLM versions is a supported
        use of this tool, so blocking on drift would remove a feature.
        """
        monkeypatch.setenv("VLLM_REFERENCE_VERSION", "0.99.0")
        from vllmbench_api.reference import reference_vllm_version

        reference_vllm_version.cache_clear()
        try:
            response = await client.post(
                "/api/hosts", json={"name": "drifted", "agent_url": MOCK_URL}
            )
            assert response.status_code == 201
            body = response.json()
            assert body["vllm_version_matches_reference"] is False
            assert body["reference_vllm_version"] == "0.99.0"
        finally:
            reference_vllm_version.cache_clear()


class TestRefresh:
    async def test_refresh_updates_last_seen(self, client: httpx.AsyncClient) -> None:
        created = (
            await client.post("/api/hosts", json={"name": "refreshable", "agent_url": MOCK_URL})
        ).json()
        refreshed = (await client.post(f"/api/hosts/{created['id']}/refresh")).json()
        assert refreshed["last_seen_at"] >= created["last_seen_at"]

    async def test_refresh_of_unknown_host_is_404(self, client: httpx.AsyncClient) -> None:
        missing = "00000000-0000-0000-0000-000000000000"
        assert (await client.post(f"/api/hosts/{missing}/refresh")).status_code == 404
