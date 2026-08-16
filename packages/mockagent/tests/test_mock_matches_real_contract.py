"""The mock must be indistinguishable from the real agent except in what it reports.

CLAUDE.md: "when the real agent's contract changes, the mock changes in the same PR."
That is a rule someone has to remember, so these tests make forgetting it fail loudly.
A mock that has drifted is worse than no mock: it makes tests pass while the real path
is broken.
"""

from __future__ import annotations

import httpx
import pytest

from vllmbench_agent.main import create_app as create_real_app
from vllmbench_agent.settings import AgentSettings
from vllmbench_mockagent.main import SYNTHETIC_SOURCE
from vllmbench_mockagent.main import create_app as create_mock_app
from vllmbench_protocol import PROTOCOL_VERSION, AgentClient
from vllmbench_protocol.wire import HealthResponse, HostInfo

TOKEN = "test-token-not-a-real-secret"


@pytest.fixture
def real_app():
    return create_real_app(AgentSettings(token=TOKEN))


@pytest.fixture
def mock_app():
    return create_mock_app(token=TOKEN)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://agent",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )


class TestContractParity:
    async def test_both_expose_the_same_routes(self, real_app, mock_app) -> None:
        real = {
            (r.path, tuple(sorted(r.methods))) for r in real_app.routes if hasattr(r, "methods")
        }
        mock = {
            (r.path, tuple(sorted(r.methods))) for r in mock_app.routes if hasattr(r, "methods")
        }
        # Only compare the agent contract, not FastAPI's own docs plumbing.
        contract = {"/health", "/host-info"}
        assert {r for r in real if r[0] in contract} == {r for r in mock if r[0] in contract}

    async def test_both_validate_against_the_same_models(self, real_app, mock_app) -> None:
        for app in (real_app, mock_app):
            async with _client(app) as http:
                HealthResponse.model_validate((await http.get("/health")).json())
                HostInfo.model_validate((await http.get("/host-info")).json())

    async def test_both_enforce_authentication_identically(self, real_app, mock_app) -> None:
        for app in (real_app, mock_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://agent"
            ) as http:
                assert (await http.get("/host-info")).status_code == 401
                bad = await http.get("/host-info", headers={"Authorization": "Bearer nope"})
                assert bad.status_code == 403

    async def test_both_report_the_same_protocol_version(self, real_app, mock_app) -> None:
        for app in (real_app, mock_app):
            async with _client(app) as http:
                info = HostInfo.model_validate((await http.get("/host-info")).json())
                assert info.protocol_version == PROTOCOL_VERSION

    async def test_the_shared_client_works_against_the_mock(self, mock_app) -> None:
        async with _client(mock_app) as http:
            info = await AgentClient("http://agent", TOKEN, client=http).host_info()
            assert info.gpu_count == 2


class TestMockDeclaresItselfSynthetic:
    async def test_mock_sets_synthetic_source(self, mock_app) -> None:
        # The load-bearing difference. Everything the control plane does to quarantine
        # these runs keys off this one field being set by the producer.
        async with _client(mock_app) as http:
            info = HostInfo.model_validate((await http.get("/host-info")).json())
            assert info.synthetic_source == SYNTHETIC_SOURCE

    async def test_real_and_mock_differ_only_in_that_field(self, real_app, mock_app) -> None:
        async with _client(real_app) as http:
            real = HostInfo.model_validate((await http.get("/host-info")).json())
        async with _client(mock_app) as http:
            mock = HostInfo.model_validate((await http.get("/host-info")).json())

        assert real.synthetic_source is None
        assert mock.synthetic_source is not None


class TestMockIsMultiGpu:
    async def test_mock_reports_more_than_one_device(self, mock_app) -> None:
        """Two devices, not one.

        A single-device mock would let per-device handling — the TP-imbalance charts,
        the per-GPU normalization — break without any test noticing, because everything
        degenerates correctly at gpu_count=1.
        """
        async with _client(mock_app) as http:
            info = HostInfo.model_validate((await http.get("/host-info")).json())
            assert info.gpu_count >= 2
            assert [g.index for g in info.gpus] == list(range(info.gpu_count))
