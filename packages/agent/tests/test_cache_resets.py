"""Cache resets between sweep points, which for a long time did not happen.

`VllmServer.reset_caches` has called `/reset_prefix_cache` and `/reset_mm_cache` since the
agent was written. Both are gated behind `VLLM_SERVER_DEV_MODE`, the agent did not set it,
and so both returned 404 — which the method treated as "this vLLM version does not have that
endpoint" and carried on from. Every sweep this project ever ran carried its prefix cache
across every point, and logged `reset caches: none available` while doing it (issue #87).

CLAUDE.md names the consequence: "prefix cache carryover across sweep points silently
invalidates results". Silently is the operative word — the later points of a matrix come out
faster than they are, so the *ordering* of the matrix decides its winner and nothing about
the chart looks wrong.

The fake `vllm` shim reads the real `VLLM_SERVER_DEV_MODE` rather than faking dev mode into
being always on. That is what makes these regression tests rather than descriptions: stop
setting the variable and they fail here, on a laptop, instead of on a GPU host months later.
"""

from __future__ import annotations

import os
import socket
import stat
from pathlib import Path

import pytest

from vllmbench_agent.reaper import ProcessRegistry
from vllmbench_agent.vllm_server import ENGINE_ENV, RESET_ENDPOINTS, ServerError, VllmServer

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = "model: facebook/opt-125m\n"
CONFIG_HASH = "b" * 64


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def fake_vllm_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    target = bin_dir / "vllm"
    target.write_text((FIXTURES / "fake_vllm").read_text())
    target.chmod(target.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return target


async def start(tmp_path: Path) -> VllmServer:
    server = VllmServer(registry=ProcessRegistry(state_dir=tmp_path / "state"))
    await server.start(
        config_yaml=CONFIG,
        config_hash=CONFIG_HASH,
        port=_free_port(),
        readiness_timeout_seconds=30,
    )
    return server


class TestTheEngineIsLaunchedInDevMode:
    def test_the_variable_is_set(self) -> None:
        """Not a preference. `vllm/benchmarks/sweep/server.py` launches its own server with
        `env=os.environ | {"VLLM_SERVER_DEV_MODE": "1"}` and the comment "Need
        `VLLM_SERVER_DEV_MODE=1` for `_reset_caches`". Invariant 4 says we orchestrate
        upstream's benchmark rather than reimplement it, and dropping this is how we ended
        up reimplementing the part around it badly."""
        assert ENGINE_ENV["VLLM_SERVER_DEV_MODE"] == "1"

    def test_upstreams_three_endpoints_are_all_called(self) -> None:
        """`/reset_encoder_cache` is on upstream's list and used to be missing from ours."""
        assert RESET_ENDPOINTS == (
            "/reset_prefix_cache",
            "/reset_mm_cache",
            "/reset_encoder_cache",
        )


class TestResettingCaches:
    async def test_all_three_answer(self, tmp_path: Path, fake_vllm_path: Path) -> None:
        server = await start(tmp_path)
        try:
            assert await server.reset_caches() == list(RESET_ENDPOINTS)
        finally:
            await server.stop()

    async def test_a_refused_reset_raises_instead_of_shrugging(
        self, tmp_path: Path, fake_vllm_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bug, reproduced: an engine healthy in every other way — it answers /health
        and registers a model — that 404s the reset because it was not started in dev mode.

        Raising is the whole point. A benchmark run after a reset that did not happen
        measures a warm cache from the previous sweep point and reports it as this
        configuration's number. There is no later stage that can notice, which is why this
        one has to.
        """
        monkeypatch.setenv("FAKE_VLLM_MODE", "no_dev_mode")
        server = await start(tmp_path)
        try:
            with pytest.raises(ServerError) as caught:
                await server.reset_caches()
            assert "refused a cache reset" in str(caught.value)
            # "404" on its own sends the reader to the vLLM version, which is where the
            # original bug hid for as long as it did. The cause is one variable, so the
            # message names it.
            assert "VLLM_SERVER_DEV_MODE" in str(caught.value)
        finally:
            await server.stop()

    async def test_the_failure_reaches_the_caller_as_a_failed_run(
        self, tmp_path: Path, fake_vllm_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Classified as a host-state failure rather than a benchmark failure: nothing was
        wrong with the request, and the fix is on the host."""
        monkeypatch.setenv("FAKE_VLLM_MODE", "no_dev_mode")
        server = await start(tmp_path)
        try:
            with pytest.raises(ServerError) as caught:
                await server.reset_caches()
            assert caught.value.kind.value == "engine_not_ready"
        finally:
            await server.stop()
