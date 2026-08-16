"""The agent's HTTP contract, including the parts that must fail.

Runs without a GPU. That is the point: everything the agent does short of launching vLLM
has to be testable on hardware that has no NVIDIA driver, or CI cannot cover it at all.
"""

from __future__ import annotations

import httpx
import pytest

from vllmbench_agent.main import create_app
from vllmbench_agent.settings import AgentSettings
from vllmbench_protocol import PROTOCOL_VERSION, AgentAuthError, AgentClient, ProtocolMismatch
from vllmbench_protocol.wire import HealthResponse, HostInfo

TOKEN = "test-token-not-a-real-secret"


@pytest.fixture
def app():
    return create_app(AgentSettings(token=TOKEN))


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://agent") as c:
        yield c


class TestHealth:
    async def test_health_needs_no_token(self, client: httpx.AsyncClient) -> None:
        # An operator with a wrong token must still be able to tell the agent is alive.
        # Requiring auth here would make "not running" and "wrong token" look identical.
        response = await client.get("/health")
        assert response.status_code == 200
        parsed = HealthResponse.model_validate(response.json())
        assert parsed.protocol_version == PROTOCOL_VERSION

    async def test_health_reports_uptime(self, client: httpx.AsyncClient) -> None:
        parsed = HealthResponse.model_validate((await client.get("/health")).json())
        assert parsed.uptime_seconds >= 0


class TestAuthentication:
    async def test_host_info_requires_a_token(self, client: httpx.AsyncClient) -> None:
        assert (await client.get("/host-info")).status_code == 401

    async def test_wrong_token_is_forbidden(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/host-info", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 403

    @pytest.mark.parametrize(
        "header",
        ["", "Basic abc", f"Token {TOKEN}", "Bearer", "Bearer "],
    )
    async def test_malformed_authorization_headers_are_rejected(
        self, client: httpx.AsyncClient, header: str
    ) -> None:
        response = await client.get("/host-info", headers={"Authorization": header})
        assert response.status_code in (401, 403)

    async def test_correct_token_is_accepted(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/host-info", headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 200


class TestHostInfo:
    async def test_reports_protocol_and_agent_version(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/host-info", headers={"Authorization": f"Bearer {TOKEN}"})
        info = HostInfo.model_validate(response.json())
        assert info.protocol_version == PROTOCOL_VERSION
        assert info.hostname

    async def test_real_agent_never_declares_itself_synthetic(
        self, client: httpx.AsyncClient
    ) -> None:
        # The other half of invariant 7. If the real agent ever set this, genuine runs
        # would be quarantined and silently vanish from every chart.
        response = await client.get("/host-info", headers={"Authorization": f"Bearer {TOKEN}"})
        assert HostInfo.model_validate(response.json()).synthetic_source is None

    async def test_survives_a_host_with_no_gpus(self, client: httpx.AsyncClient) -> None:
        # This test machine has no NVIDIA driver, which is exactly the case that must not
        # crash the agent: developers run it on laptops.
        response = await client.get("/host-info", headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 200
        info = HostInfo.model_validate(response.json())
        assert info.gpu_count == len(info.gpus)


class TestAgentClient:
    async def test_client_reads_host_info(self, app) -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://agent",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as http:
            client = AgentClient("http://agent", TOKEN, client=http)
            info = await client.host_info()
            assert info.protocol_version == PROTOCOL_VERSION

    async def test_client_raises_on_bad_token(self, app) -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://agent",
            headers={"Authorization": "Bearer wrong"},
        ) as http:
            client = AgentClient("http://agent", "wrong", client=http)
            with pytest.raises(AgentAuthError):
                await client.host_info()

    async def test_client_refuses_a_protocol_mismatch(self, app) -> None:
        """A stale agent must fail at connect time, not mid-sweep.

        Simulated by asking a client that believes in a different protocol version. The
        refusal is the point: proceeding and hoping the differences do not matter risks a
        sweep that completes and writes subtly wrong results.
        """
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://agent",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as http:
            client = AgentClient(
                "http://agent",
                TOKEN,
                client=http,
                expected_protocol_version=PROTOCOL_VERSION + 1,
            )
            with pytest.raises(ProtocolMismatch) as exc_info:
                await client.host_info()

            # The message must name both versions; "protocol mismatch" alone tells an
            # operator nothing about which side to upgrade.
            message = str(exc_info.value)
            assert str(PROTOCOL_VERSION) in message
            assert str(PROTOCOL_VERSION + 1) in message

    async def test_protocol_check_can_be_skipped(self, app) -> None:
        # Needed so a mismatched host can still be *inspected* to report what it is
        # running, rather than becoming completely opaque.
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://agent",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as http:
            client = AgentClient(
                "http://agent",
                TOKEN,
                client=http,
                expected_protocol_version=PROTOCOL_VERSION + 1,
            )
            info = await client.host_info(check_protocol=False)
            assert info.protocol_version == PROTOCOL_VERSION
