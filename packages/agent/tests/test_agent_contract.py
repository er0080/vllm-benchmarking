"""The agent's HTTP contract, including the parts that must fail.

Runs without a GPU. That is the point: everything the agent does short of launching vLLM
has to be testable on hardware that has no NVIDIA driver, or CI cannot cover it at all.
"""

from __future__ import annotations

from pathlib import Path

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


class TestVllmVersionProbe:
    """A null version must explain itself.

    On the first real GPU host this came back null and the payload gave no clue whether
    vLLM was missing, off PATH, or merely slow to import — three problems with three
    different fixes.
    """

    def test_detail_explains_a_missing_binary(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vllmbench_agent.hardware import probe_vllm_version

        probe_vllm_version.cache_clear()
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        version, detail = probe_vllm_version()
        probe_vllm_version.cache_clear()

        assert version is None
        assert "VLLMBENCH_VLLM_BIN" in detail

    def test_version_is_reported_with_its_source(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os
        import stat

        from vllmbench_agent.hardware import probe_vllm_version

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        target = bin_dir / "vllm"
        target.write_text((Path(__file__).parent / "fixtures" / "fake_vllm").read_text())
        target.chmod(target.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

        probe_vllm_version.cache_clear()
        version, detail = probe_vllm_version()
        probe_vllm_version.cache_clear()

        assert version == "0.25.1"
        assert "vllm --version" in detail

    def test_host_info_carries_the_detail(self, client: httpx.AsyncClient) -> None:
        # Synchronous by design — this asserts the field exists on the wire.
        assert "vllm_probe_detail" in HostInfo.model_fields


class TestDeclaredParallelism:
    """The config's request, recorded as a request rather than an outcome."""

    def test_reads_both_forms(self) -> None:
        from vllmbench_agent.vllm_server import _declared_parallelism

        assert _declared_parallelism("tensor_parallel_size: 4\n") == (4, None)
        assert _declared_parallelism("tensor-parallel-size: 2\n") == (2, None)
        assert _declared_parallelism("pipeline_parallel_size: 3\n") == (None, 3)

    def test_absent_means_none_not_one(self) -> None:
        # None and 1 mean different things: "not specified" versus "explicitly single".
        # Defaulting to 1 here would erase the distinction before it reaches the run.
        from vllmbench_agent.vllm_server import _declared_parallelism

        assert _declared_parallelism("model: m\n") == (None, None)

    def test_garbage_is_ignored_rather_than_raising(self) -> None:
        from vllmbench_agent.vllm_server import _declared_parallelism

        assert _declared_parallelism("tensor_parallel_size: lots\n") == (None, None)


class TestDeviceAttribution:
    def test_returns_empty_without_nvml(self) -> None:
        """No GPUs is a legitimate answer, not an error.

        The caller treats an empty list as "could not attribute", and falls back to the
        declared count rather than claiming zero devices.
        """
        import os

        from vllmbench_agent.hardware import devices_for_process

        assert devices_for_process(os.getpid()) == []


class TestProtocolCheckPrecedesValidation:
    """A version mismatch must be reported as one, whatever the payload looks like.

    The wire models forbid unknown fields, so a newer counterpart's payload fails
    validation outright. If validation runs first, the one situation the version check
    exists to explain surfaces as "extra inputs are not permitted" naming some field —
    which is what a stale control plane actually did against a real host.
    """

    async def test_mismatch_wins_over_an_unknown_field(self) -> None:
        from fastapi import FastAPI

        # An agent from the future: newer protocol, and a field this build cannot parse.
        future = FastAPI()

        @future.get("/host-info")
        async def host_info() -> dict[str, object]:
            return {
                "protocol_version": PROTOCOL_VERSION + 1,
                "agent_version": "9.9.9",
                "hostname": "future-host",
                "gpus": [],
                "a_field_from_the_future": True,
            }

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=future), base_url="http://agent"
        ) as http:
            client = AgentClient("http://agent", TOKEN, client=http)
            with pytest.raises(ProtocolMismatch) as exc:
                await client.host_info()

        message = str(exc.value)
        assert str(PROTOCOL_VERSION + 1) in message
        assert str(PROTOCOL_VERSION) in message
        # Not a validation error about some field name the operator has never heard of.
        assert "extra" not in message.lower()


class TestFailureKindHeader:
    """The agent's verdict has to survive the trip, and cost nothing when it does not.

    The header exists instead of a new wire field precisely so that neither side has to
    be upgraded in lockstep for the other to keep working — no protocol bump, no
    redeployment to every GPU host. That claim is worth a test on both halves: it is
    present and correct when the agent sends it, and its absence changes nothing.
    """

    async def test_a_failed_start_names_its_kind_in_the_header(
        self, app, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vllmbench_agent.vllm_server import ServerError
        from vllmbench_protocol.failures import FAILURE_KIND_HEADER, FailureKind

        async def refuse(_self, **_kwargs: object):
            raise ServerError("no memory left", FailureKind.ENGINE_OUT_OF_MEMORY)

        # Patched on the class, which is the object `create_app` closed over an
        # instance of — reaching for the instance would mean reaching into the closure.
        from vllmbench_agent.vllm_server import VllmServer

        monkeypatch.setattr(VllmServer, "start", refuse)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://agent") as raw:
            response = await raw.post(
                "/server/start",
                json={"config_yaml": "model: m\n", "config_hash": "a" * 64, "port": 8000},
                headers={"Authorization": f"Bearer {TOKEN}"},
            )

        assert response.status_code == 409
        assert response.headers[FAILURE_KIND_HEADER] == FailureKind.ENGINE_OUT_OF_MEMORY
        # The message is still the whole message. The header adds a heading; it does not
        # replace the only text that says *how much* memory was wanted.
        assert "no memory left" in response.json()["detail"]

    async def test_the_client_carries_it_onto_the_error(
        self, app, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vllmbench_agent.vllm_server import ServerError
        from vllmbench_protocol import AgentError
        from vllmbench_protocol.failures import FailureKind
        from vllmbench_protocol.wire import StartServerRequest

        async def refuse(_self, **_kwargs: object):
            raise ServerError(
                "vllm: error: unrecognized arguments: --nope", FailureKind.ENGINE_CONFIG_REJECTED
            )

        from vllmbench_agent.vllm_server import VllmServer

        monkeypatch.setattr(VllmServer, "start", refuse)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://agent",
            # An injected client brings its own headers; the constructor only sets them
            # on a client it makes itself.
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as http_client:
            agent = AgentClient("http://agent", TOKEN, client=http_client)
            with pytest.raises(AgentError) as exc:
                await agent.start_server(
                    StartServerRequest(config_yaml="model: m\n", config_hash="a" * 64, port=8000)
                )

        assert exc.value.reported_kind == FailureKind.ENGINE_CONFIG_REJECTED
