"""Server supervision, including every way it can go wrong.

The failure paths matter more than the happy path here. A supervisor that starts servers
correctly but occasionally leaks one is worse than useless on a GPU host: the leaked
process holds VRAM, nothing identifies it as ours, and the next sweep either fails to
allocate or quietly competes with a ghost for the device.
"""

from __future__ import annotations

import os
import socket
import stat
from pathlib import Path

import pytest

from vllmbench_agent.reaper import ProcessRegistry, _process_exists
from vllmbench_agent.vllm_server import ServerError, VllmServer
from vllmbench_protocol.wire import ServerState

FIXTURES = Path(__file__).parent / "fixtures"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def fake_vllm_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a `vllm` executable on PATH that we can make misbehave on demand."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    target = bin_dir / "vllm"
    target.write_text((FIXTURES / "fake_vllm").read_text())
    target.chmod(target.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return target


@pytest.fixture
def server(tmp_path: Path, fake_vllm_path: Path) -> VllmServer:
    return VllmServer(registry=ProcessRegistry(state_dir=tmp_path / "state"))


CONFIG = "model: facebook/opt-125m\n"
CONFIG_HASH = "a" * 64


class TestStartup:
    async def test_starts_and_becomes_ready(self, server: VllmServer) -> None:
        status = await server.start(
            config_yaml=CONFIG,
            config_hash=CONFIG_HASH,
            port=_free_port(),
            readiness_timeout_seconds=30,
        )
        try:
            assert status.state is ServerState.READY
            assert status.pid is not None
            assert status.ready_at is not None
        finally:
            await server.stop()

    async def test_waits_for_the_model_not_just_health(
        self, server: VllmServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """/health answering 200 is not readiness.

        In never_ready mode the fake serves /health immediately but never registers a
        model. A supervisor that trusted /health would declare this ready and benchmark
        a server with nothing loaded — producing numbers that look real.
        """
        monkeypatch.setenv("FAKE_VLLM_MODE", "never_ready")
        with pytest.raises(ServerError, match="did not become ready"):
            await server.start(
                config_yaml=CONFIG,
                config_hash=CONFIG_HASH,
                port=_free_port(),
                readiness_timeout_seconds=6,
            )
        assert server.state is ServerState.STOPPED

    async def test_crash_during_load_reports_the_reason(
        self, server: VllmServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_VLLM_MODE", "crash")
        with pytest.raises(ServerError) as exc:
            await server.start(
                config_yaml=CONFIG,
                config_hash=CONFIG_HASH,
                port=_free_port(),
                readiness_timeout_seconds=30,
            )
        message = str(exc.value)
        assert "exited with code 3" in message
        # The log tail is the only thing that distinguishes OOM from a bad config, and
        # without it the operator gets "it failed" and nothing actionable.
        assert "out of memory" in message.lower()

    async def test_a_failed_start_leaves_nothing_running(
        self, server: VllmServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("FAKE_VLLM_MODE", "never_ready")
        with pytest.raises(ServerError):
            await server.start(
                config_yaml=CONFIG,
                config_hash=CONFIG_HASH,
                port=_free_port(),
                readiness_timeout_seconds=6,
            )
        # The dangerous case: a server that failed readiness is still a process holding
        # VRAM. It must be gone, and its ownership record with it.
        assert ProcessRegistry(state_dir=tmp_path / "state").read() is None

    async def test_refuses_a_second_server(self, server: VllmServer) -> None:
        await server.start(
            config_yaml=CONFIG,
            config_hash=CONFIG_HASH,
            port=_free_port(),
            readiness_timeout_seconds=30,
        )
        try:
            # Two engines on one host compete for the same VRAM, and the resulting
            # numbers measure the contention rather than either configuration.
            with pytest.raises(ServerError, match="already"):
                await server.start(
                    config_yaml=CONFIG,
                    config_hash="b" * 64,
                    port=_free_port(),
                    readiness_timeout_seconds=30,
                )
        finally:
            await server.stop()

    async def test_missing_vllm_binary_is_explained(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        server = VllmServer(registry=ProcessRegistry(state_dir=tmp_path / "state"))
        with pytest.raises(ServerError, match="same environment as vLLM"):
            await server.start(
                config_yaml=CONFIG,
                config_hash=CONFIG_HASH,
                port=_free_port(),
                readiness_timeout_seconds=5,
            )


class TestShutdown:
    async def test_stop_kills_the_process(self, server: VllmServer) -> None:
        status = await server.start(
            config_yaml=CONFIG,
            config_hash=CONFIG_HASH,
            port=_free_port(),
            readiness_timeout_seconds=30,
        )
        pid = status.pid
        assert pid is not None

        await server.stop()
        assert not _process_exists(pid)
        assert server.state is ServerState.STOPPED

    async def test_stop_escalates_to_sigkill(
        self, server: VllmServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A process that ignores SIGTERM must still die.

        Real engines do occasionally wedge during shutdown. Politeness that gives up
        leaves VRAM held indefinitely, so the escalation path is mandatory rather than
        best-effort.
        """
        monkeypatch.setenv("FAKE_VLLM_MODE", "ignore_sigterm")
        monkeypatch.setattr("vllmbench_agent.reaper.TERM_GRACE_SECONDS", 1.0)
        status = await server.start(
            config_yaml=CONFIG,
            config_hash=CONFIG_HASH,
            port=_free_port(),
            readiness_timeout_seconds=30,
        )
        pid = status.pid
        assert pid is not None

        await server.stop()
        assert not _process_exists(pid)

    async def test_stopping_when_nothing_runs_is_harmless(self, server: VllmServer) -> None:
        assert (await server.stop()).state is ServerState.STOPPED

    async def test_ownership_record_is_cleared_on_stop(
        self, server: VllmServer, tmp_path: Path
    ) -> None:
        registry = ProcessRegistry(state_dir=tmp_path / "state")
        await server.start(
            config_yaml=CONFIG,
            config_hash=CONFIG_HASH,
            port=_free_port(),
            readiness_timeout_seconds=30,
        )
        assert registry.read() is not None
        await server.stop()
        # A stale record would make the next agent startup try to kill a dead pid, and
        # with PID recycling that is how you kill something unrelated.
        assert registry.read() is None


class TestOrphanReaping:
    async def test_reaps_a_server_from_a_previous_agent_lifetime(
        self, tmp_path: Path, fake_vllm_path: Path
    ) -> None:
        """The scenario: agent dies, vLLM keeps running, agent restarts.

        Without this, the leaked process holds VRAM and nothing on the box identifies it
        as ours to clean up.
        """
        state_dir = tmp_path / "state"
        first = VllmServer(registry=ProcessRegistry(state_dir=state_dir))
        status = await first.start(
            config_yaml=CONFIG,
            config_hash=CONFIG_HASH,
            port=_free_port(),
            readiness_timeout_seconds=30,
        )
        pid = status.pid
        assert pid is not None and _process_exists(pid)

        # Simulate the agent process vanishing without stopping the server: a new
        # supervisor over the same state directory, with the old process still alive.
        second = VllmServer(registry=ProcessRegistry(state_dir=state_dir))
        second.reap_orphans()

        assert not _process_exists(pid)
        assert ProcessRegistry(state_dir=state_dir).read() is None

    def test_no_record_means_nothing_to_do(self, tmp_path: Path) -> None:
        assert ProcessRegistry(state_dir=tmp_path / "state").reap_orphan() is None

    def test_stale_record_for_a_dead_pid_is_discarded(self, tmp_path: Path) -> None:
        registry = ProcessRegistry(state_dir=tmp_path / "state")
        # A pid that is almost certainly not running and not ours.
        registry.record(pid=999_999, pgid=999_999, port=8000, config_hash=CONFIG_HASH)
        assert registry.reap_orphan() is None
        assert registry.read() is None

    def test_refuses_to_kill_a_recycled_pid(self, tmp_path: Path) -> None:
        """The guard that keeps this from being dangerous.

        PIDs are recycled. A record naming a pid that now belongs to someone else must
        never be acted on — on a shared GPU host that would mean killing another
        person's job.
        """
        registry = ProcessRegistry(state_dir=tmp_path / "state")
        registry.record(pid=os.getpid(), pgid=os.getpgid(0), port=8000, config_hash=CONFIG_HASH)

        # Rewrite the record with a create_time that cannot match this process, which is
        # exactly what a recycled pid looks like.
        record = registry.state_file.read_text().replace('"create_time":', '"create_time_orig":')
        import json

        data = json.loads(record.replace('"create_time_orig":', '"create_time":'))
        data["create_time"] = 1.0
        registry.state_file.write_text(json.dumps(data))

        assert registry.reap_orphan() is None
        # Still alive: this test process was not killed by its own reaper.
        assert _process_exists(os.getpid())
