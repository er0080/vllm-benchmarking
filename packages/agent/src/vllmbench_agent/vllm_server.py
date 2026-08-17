"""Launching, waiting on, and tearing down a vLLM server.

Two rules from CLAUDE.md drive the shape of this:

*Wait for real readiness, not a sleep.* Benchmarking a server that is still loading
weights produces a TTFT number that measures the loader, not the engine — and it looks
like a legitimate measurement. So readiness means ``/health`` answered 200 **and** the
model is actually being served, checked while watching the process stay alive.

*Every ``vllm serve`` this agent starts, it kills.* Including on crash and restart, which
is why ownership is recorded in :mod:`vllmbench_agent.reaper` before the process exists.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path

import httpx

from vllmbench_agent.hardware import (
    devices_for_process,
    resolve_vllm_binary,
    vllm_binary_search_detail,
)
from vllmbench_agent.reaper import TERM_GRACE_SECONDS, ProcessRegistry
from vllmbench_protocol.wire import ServerState, ServerStatus

log = logging.getLogger(__name__)

LOG_TAIL_LINES = 200
READINESS_POLL_SECONDS = 2.0


class ServerError(RuntimeError):
    """The server could not be started, or died while starting."""


def _spawn(argv: list[str]) -> subprocess.Popen[str]:
    """Launch vLLM in its own process group.

    ``start_new_session`` is what makes teardown reliable: vLLM forks a worker per
    device, and signalling only the parent leaves those workers alive holding VRAM.
    """
    return subprocess.Popen(  # noqa: S603 - argv is built from a resolved executable
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )


def _declared_parallelism(config_yaml: str) -> tuple[int | None, int | None]:
    """Read the parallelism the config *asks* for.

    Narrow line reads rather than a YAML parse, for the same reason the model name is
    read this way: invariant 5 keeps the config opaque, and a parser would mean having
    opinions about vLLM options that rot with every release. These values are recorded
    as the request, never as the outcome.
    """
    tp = pp = None
    for line in config_yaml.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        name = key.strip().replace("-", "_")
        try:
            if name == "tensor_parallel_size":
                tp = int(value.strip())
            elif name == "pipeline_parallel_size":
                pp = int(value.strip())
        except ValueError:
            continue
    return tp, pp


class VllmServer:
    """Owns at most one vLLM process.

    One at a time, deliberately. A GPU host runs one engine per sweep; allowing two
    would mean two engines competing for the same VRAM and producing numbers that
    measure the contention rather than either configuration.
    """

    def __init__(self, registry: ProcessRegistry | None = None, vllm_bin: str = "") -> None:
        self._registry = registry or ProcessRegistry()
        self._vllm_bin = vllm_bin
        self._lock = asyncio.Lock()

        self._process: subprocess.Popen[str] | None = None
        self._log: deque[str] = deque(maxlen=LOG_TAIL_LINES)
        self._reader: asyncio.Task[None] | None = None
        self._config_dir: Path | None = None

        self._state = ServerState.STOPPED
        self._config_hash: str | None = None
        self._port: int | None = None
        self._started_at: float | None = None
        self._ready_at: float | None = None
        self._error: str | None = None
        self._engine_version: str | None = None
        self._served_model_name: str | None = None
        self._device_indices: list[int] | None = None
        self._declared_tp: int | None = None
        self._declared_pp: int | None = None

    # -- introspection -------------------------------------------------------------

    @property
    def state(self) -> ServerState:
        return self._state

    def status(self) -> ServerStatus:
        return ServerStatus(
            state=self._state,
            config_hash=self._config_hash,
            pid=self._process.pid if self._process else None,
            port=self._port,
            started_at=self._started_at,
            ready_at=self._ready_at,
            error=self._error,
            log_tail=list(self._log),
            vllm_version=self._engine_version,
            served_model_name=self._served_model_name,
            device_indices=self._device_indices,
            tensor_parallel_size=self._declared_tp,
            pipeline_parallel_size=self._declared_pp,
        )

    def base_url(self) -> str:
        if self._port is None:
            raise ServerError("no server is running")
        # Loopback, always. Invariant 2: the benchmark client talks to 127.0.0.1 so that
        # network RTT and jitter never enter TTFT or ITL.
        return f"http://127.0.0.1:{self._port}"

    # -- lifecycle -----------------------------------------------------------------

    def reap_orphans(self) -> None:
        """Kill anything left over from a previous agent lifetime. Call at startup."""
        self._registry.reap_orphan()

    async def start(
        self,
        *,
        config_yaml: str,
        config_hash: str,
        port: int,
        readiness_timeout_seconds: float,
    ) -> ServerStatus:
        async with self._lock:
            if self._state in (ServerState.STARTING, ServerState.READY):
                raise ServerError(
                    f"a server is already {self._state} (config {self._config_hash}); "
                    "stop it before starting another"
                )

            executable = resolve_vllm_binary(self._vllm_bin)
            if executable is None:
                raise ServerError(
                    f"no `vllm` executable found: {vllm_binary_search_detail(self._vllm_bin)}"
                )

            self._reset()
            self._state = ServerState.STARTING
            self._config_hash = config_hash
            self._port = port
            self._started_at = time.time()

            self._declared_tp, self._declared_pp = _declared_parallelism(config_yaml)
            config_path = self._write_config(config_yaml)
            argv = [executable, "serve", "--config", str(config_path), "--port", str(port)]
            log.info("starting vLLM: %s", " ".join(argv))

            try:
                # Off the event loop: fork/exec of a large process can block for long
                # enough to stall the agent's own health endpoint, and an agent that
                # looks unresponsive while starting a server is indistinguishable from
                # one that has died.
                process = await asyncio.to_thread(_spawn, argv)
            except OSError as exc:
                self._fail(f"could not launch vLLM: {exc}")
                raise ServerError(str(exc)) from exc

            self._process = process
            # Recorded immediately. A crash between here and readiness must still leave
            # a trail the next agent lifetime can follow.
            self._registry.record(
                pid=process.pid,
                pgid=os.getpgid(process.pid),
                port=port,
                config_hash=config_hash,
            )
            self._reader = asyncio.create_task(self._drain_output(process))

        try:
            await self._await_readiness(readiness_timeout_seconds)
        except Exception:
            # Never leave a half-started engine holding VRAM because startup failed.
            await self.stop()
            raise

        return self.status()

    async def stop(self) -> ServerStatus:
        async with self._lock:
            process = self._process
            if process is None:
                self._state = ServerState.STOPPED
                return self.status()

            self._state = ServerState.STOPPING
            pid = process.pid
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                pgid = pid

            log.info("stopping vLLM pid=%d", pid)
            await asyncio.to_thread(self._terminate_child, process, pgid)

            if self._reader is not None:
                self._reader.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._reader

            self._registry.clear()
            self._cleanup_config_dir()

            self._process = None
            self._reader = None
            self._state = ServerState.STOPPED
            self._ready_at = None
            return self.status()

    # -- internals -----------------------------------------------------------------

    @staticmethod
    def _terminate_child(process: subprocess.Popen[str], pgid: int) -> None:
        """Signal the group, then reap our own child.

        The ``wait()`` is not tidiness. A killed child that is never waited on becomes a
        zombie, and a zombie still answers ``os.kill(pid, 0)`` — so a supervisor that
        polls for liveness instead of reaping concludes it failed to kill its own
        process, escalates to SIGKILL against a corpse, and then reports that VRAM may
        still be held. Reaping is what actually makes the pid go away.

        :func:`vllmbench_agent.reaper.terminate_group` keeps the poll-based approach
        because there it is signalling a process it did not fork and cannot wait on.
        """
        for sig, grace in ((signal.SIGTERM, TERM_GRACE_SECONDS), (signal.SIGKILL, 10.0)):
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(pgid, sig)
            try:
                process.wait(timeout=grace)
                return
            except subprocess.TimeoutExpired:
                if sig is signal.SIGTERM:
                    log.warning(
                        "pid %d did not exit within %.0fs; escalating to SIGKILL",
                        process.pid,
                        grace,
                    )
        log.error("pid %d survived SIGKILL; VRAM may still be held", process.pid)

    def _reset(self) -> None:
        self._log.clear()
        self._error = None
        self._ready_at = None

    def _write_config(self, config_yaml: str) -> Path:
        # Written verbatim and handed to --config. Invariant 5: what runs is what was
        # stored, with no intermediate representation to lose anything in.
        self._config_dir = Path(tempfile.mkdtemp(prefix="vllmbench-config-"))
        path = self._config_dir / "server.yaml"
        path.write_text(config_yaml)
        return path

    def _cleanup_config_dir(self) -> None:
        if self._config_dir is not None:
            shutil.rmtree(self._config_dir, ignore_errors=True)
            self._config_dir = None

    def _fail(self, message: str) -> None:
        self._state = ServerState.FAILED
        self._error = message
        log.error("vLLM server failed: %s", message)

    async def _drain_output(self, process: subprocess.Popen[str]) -> None:
        """Keep a bounded tail of the server's output.

        Bounded because a model load can emit tens of thousands of lines and the agent
        runs on the machine under test — unbounded buffering would be a memory leak on
        the box whose behaviour we are trying not to perturb.
        """
        assert process.stdout is not None
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, process.stdout.readline)
            if not line:
                return
            self._log.append(line.rstrip("\n"))

    async def _await_readiness(self, timeout_seconds: float) -> None:
        """Poll until the server is genuinely serving, or fail with a real reason."""
        deadline = time.monotonic() + timeout_seconds
        url = f"http://127.0.0.1:{self._port}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.monotonic() < deadline:
                process = self._process
                if process is None:
                    raise ServerError("server was stopped while starting")

                exit_code = process.poll()
                if exit_code is not None:
                    # Died during startup. The tail is the only useful thing we have, and
                    # without it the operator gets "it failed" and nothing else.
                    tail = "\n".join(list(self._log)[-25:])
                    message = f"vLLM exited with code {exit_code} during startup:\n{tail}"
                    self._fail(message)
                    raise ServerError(message)

                with contextlib.suppress(httpx.HTTPError):
                    health = await client.get(f"{url}/health")
                    if health.status_code == 200:
                        # /health alone is not enough: it can answer before a model is
                        # registered, and benchmarking then measures the loader.
                        models = await client.get(f"{url}/v1/models")
                        if models.status_code == 200 and models.json().get("data"):
                            # Captured here rather than parsed from the config: a config
                            # with `served-model-name` answers under the alias, and
                            # requesting the HF id would 404 every benchmark.
                            self._served_model_name = models.json()["data"][0]["id"]
                            self._state = ServerState.READY
                            self._ready_at = time.time()
                            elapsed = self._ready_at - (self._started_at or self._ready_at)

                            await self._capture_engine_facts(client, url, process.pid)

                            log.info(
                                "vLLM ready after %.1fs (version=%s devices=%s)",
                                elapsed,
                                self._engine_version,
                                self._device_indices,
                            )
                            return

                await asyncio.sleep(READINESS_POLL_SECONDS)

        message = f"vLLM did not become ready within {timeout_seconds:.0f}s"
        self._fail(message)
        raise ServerError(message)

    async def _capture_engine_facts(self, client: httpx.AsyncClient, url: str, pid: int) -> None:
        """Record what actually came up, as opposed to what was requested.

        Both facts are read from the running engine rather than derived from the config,
        because the config states an intention and this records an outcome. A config
        asking for four devices on a two-device host is not an error the engine reports
        loudly — it is a run that would otherwise claim a topology that never existed.
        """
        with contextlib.suppress(httpx.HTTPError, ValueError):
            response = await client.get(f"{url}/version")
            if response.status_code == 200:
                self._engine_version = response.json().get("version")

        devices = await asyncio.to_thread(devices_for_process, pid)
        self._device_indices = devices or None

        if devices and self._declared_tp and len(devices) != self._declared_tp:
            # Not fatal, but never silent: per-GPU normalization uses the observed
            # count, so this changes what the numbers mean.
            log.warning(
                "config asked for tensor_parallel_size=%d but the engine is on %d "
                "device(s) %s; per-GPU figures will use the observed count",
                self._declared_tp,
                len(devices),
                devices,
            )

    async def reset_caches(self) -> list[str]:
        """Call every ``/reset_*_cache`` endpoint the server exposes.

        Upstream does this between benchmark runs. Skipping it lets a warm prefix cache
        carry across sweep points, which inflates the later ones and makes the ordering
        of a matrix silently affect its results.

        Endpoints vary by version, so a 404 is a normal answer rather than an error.
        """
        reset: list[str] = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for endpoint in ("/reset_prefix_cache", "/reset_mm_cache"):
                with contextlib.suppress(httpx.HTTPError):
                    response = await client.post(f"{self.base_url()}{endpoint}")
                    if response.status_code < 400:
                        reset.append(endpoint)
        log.info("reset caches: %s", reset or "none available")
        return reset
