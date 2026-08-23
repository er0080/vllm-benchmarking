"""Whether a run was speculating, asked of the engine rather than of its config file.

Until protocol 7 a run could not say. Grouping a four-arm MTP sweep by drafting depth meant
regexing the *name* somebody had given each configuration — which is exactly what invariant 8
forbids for parallelism topology, and forbids for the same reason: a config saying
`num_speculative_tokens: 3` is not proof the engine drafted three tokens (issue #86).

The answer comes from `/server_info`, which sits behind the same `VLLM_SERVER_DEV_MODE` gate
as the cache-reset endpoints — so the fix for #87 is what made this reachable at all. The
fake `vllm` shim reads that variable rather than faking dev mode into being always on, which
is why `no_dev_mode` below produces a genuinely unanswerable engine rather than a mocked one.
"""

from __future__ import annotations

import os
import socket
import stat
from pathlib import Path

import pytest

from vllmbench_agent.reaper import ProcessRegistry
from vllmbench_agent.vllm_server import VllmServer
from vllmbench_protocol import NO_SPECULATION
from vllmbench_protocol.wire import ServerState

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


@pytest.fixture
async def started(tmp_path: Path, fake_vllm_path: Path):
    """A running engine, torn down whatever the test does to it."""
    server = VllmServer(registry=ProcessRegistry(state_dir=tmp_path / "state"))
    await server.start(
        config_yaml=CONFIG,
        config_hash=CONFIG_HASH,
        port=_free_port(),
        readiness_timeout_seconds=30,
    )
    try:
        yield server
    finally:
        await server.stop()


class TestSpeculationIsReadFromTheEngine:
    async def test_an_engine_that_is_not_speculating_says_so(self, started: VllmServer) -> None:
        """An answer, not a silence. NULL is reserved for nobody having asked."""
        status = started.status()
        assert status.speculative_method == NO_SPECULATION
        assert status.speculative_tokens == 0

    @pytest.mark.parametrize(("method", "depth"), [("ngram", 3), ("mtp", 1), ("eagle3", 5)])
    async def test_a_speculating_engine_reports_method_and_depth(
        self,
        tmp_path: Path,
        fake_vllm_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        depth: int,
    ) -> None:
        monkeypatch.setenv("FAKE_VLLM_SPECULATIVE", f"{method}:{depth}")
        server = VllmServer(registry=ProcessRegistry(state_dir=tmp_path / "state"))
        status = await server.start(
            config_yaml=CONFIG,
            config_hash=CONFIG_HASH,
            port=_free_port(),
            readiness_timeout_seconds=30,
        )
        try:
            assert (status.speculative_method, status.speculative_tokens) == (method, depth)
        finally:
            await server.stop()

    async def test_the_config_text_is_not_consulted(
        self, tmp_path: Path, fake_vllm_path: Path
    ) -> None:
        """Invariant 8's rule, applied to speculation: the YAML states an intention.

        This config asks for MTP at depth 3 and the engine is not speculating — which is
        exactly what a silently-failed drafter looks like. The run must record what the
        engine did, or the measurement is filed under an arm it does not belong to.
        """
        server = VllmServer(registry=ProcessRegistry(state_dir=tmp_path / "state"))
        status = await server.start(
            config_yaml=CONFIG
            + 'speculative-config: {"method":"mtp","num_speculative_tokens":3}\n',
            config_hash=CONFIG_HASH,
            port=_free_port(),
            readiness_timeout_seconds=30,
        )
        try:
            assert status.speculative_method == NO_SPECULATION
        finally:
            await server.stop()

    async def test_an_engine_that_will_not_answer_leaves_it_null(
        self, tmp_path: Path, fake_vllm_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`/server_info` 404s and the run records that it does not know.

        Distinct from `"none"` deliberately. A run measured against an engine we could not
        ask must not claim the engine denied speculating: an analysis grouping on this
        column would then put it in the non-speculative arm and compare it as evidence.
        """
        monkeypatch.setenv("FAKE_VLLM_MODE", "no_dev_mode")
        server = VllmServer(registry=ProcessRegistry(state_dir=tmp_path / "state"))
        status = await server.start(
            config_yaml=CONFIG,
            config_hash=CONFIG_HASH,
            port=_free_port(),
            readiness_timeout_seconds=30,
        )
        try:
            assert status.speculative_method is None
            assert status.speculative_tokens is None
            # Provenance is worth a NULL, never a failed run.
            assert status.state is ServerState.READY
        finally:
            await server.stop()
