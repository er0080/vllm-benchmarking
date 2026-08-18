"""Running ``vllm bench serve`` and returning its result.

Invariant 4: ``vllm bench serve`` is the atomic unit. We build its argument list, run it,
and hand back what it wrote. We do not parse its stdout for numbers — the ``--save-result``
JSON is the contract, and scraping the human-readable table would be a second, worse
parser that drifts independently.

The client runs here, on the GPU host, against loopback (invariant 2), so no network hop
enters the latency figures.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import signal
import tempfile
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from vllmbench_agent.hardware import (
    child_environment,
    resolve_vllm_binary,
    vllm_binary_search_detail,
)
from vllmbench_protocol.failures import FailureKind
from vllmbench_protocol.wire import BenchRequest, BenchResponse

log = logging.getLogger(__name__)

OUTPUT_TAIL_LINES = 100


class BenchError(RuntimeError):
    """The benchmark did not produce a usable result.

    Names its own kind for the same reason :class:`ServerError` does: only this side
    knows whether the client was killed by our deadline or exited on its own.
    """

    def __init__(self, message: str, kind: FailureKind = FailureKind.BENCHMARK_FAILED) -> None:
        super().__init__(message)
        self.kind = kind


class BenchCancelled(RuntimeError):
    """The benchmark was stopped on request, before it produced a result.

    Distinct from BenchError because the two mean opposite things about the run. A failed
    benchmark is a result worth recording and investigating; a cancelled one is work the
    operator asked to stop, and recording it as a failure would put a red row in a sweep
    that behaved exactly as instructed.
    """


def build_argv(
    request: BenchRequest, *, base_url: str, result_path: Path, vllm_bin: str = ""
) -> list[str]:
    """Translate a request into a ``vllm bench serve`` command line.

    Split out from execution so the mapping is testable without running anything — the
    argument list is where a silent mistake (a flag that stops existing, a rate that
    never gets applied) turns into a benchmark that measures something other than what
    was asked for.
    """
    executable = resolve_vllm_binary(vllm_bin)
    if executable is None:
        raise BenchError(f"no `vllm` executable found: {vllm_binary_search_detail(vllm_bin)}")

    argv = [
        executable,
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        base_url,
        # --model is the weights identifier; vLLM loads the tokenizer from it.
        "--model",
        request.model,
        "--dataset-name",
        request.dataset_name,
        "--num-prompts",
        str(request.num_prompts),
        "--save-result",
        "--result-filename",
        str(result_path),
    ]

    # Omitted when equal to the model, matching vLLM's own default, so the command line
    # stays the shortest thing that reproduces the run.
    if request.served_model_name and request.served_model_name != request.model:
        argv += ["--served-model-name", request.served_model_name]

    if request.dataset_path:
        argv += ["--dataset-path", request.dataset_path]
    if request.hf_name:
        argv += ["--hf-name", request.hf_name]

    # None means unbounded. `inf` is what the CLI expects for an unthrottled rate, and
    # omitting --max-concurrency entirely is how you say "no cap" — passing a large
    # number instead would quietly become a cap.
    argv += ["--request-rate", "inf" if request.request_rate is None else str(request.request_rate)]
    if request.max_concurrency is not None:
        argv += ["--max-concurrency", str(request.max_concurrency)]
    if request.burstiness is not None:
        argv += ["--burstiness", str(request.burstiness)]

    if request.random_input_len is not None:
        argv += ["--random-input-len", str(request.random_input_len)]
    if request.random_output_len is not None:
        argv += ["--random-output-len", str(request.random_output_len)]

    argv += request.extra_args
    return argv


class BenchRunner:
    """Owns at most one benchmark process, so that it can be stopped.

    State exists here for exactly one reason: cancelling a sweep must not mean waiting
    for the current point to finish. A twenty-minute benchmark that keeps running after
    the operator said stop is burning the GPU time they asked to reclaim, and killing the
    engine out from under a client that is still sending requests leaves the client
    thrashing against a dead socket until it times out.

    So the client is stopped first, by process group, and the engine after it. One
    benchmark at a time, matching the one-engine-per-host rule.
    """

    def __init__(self, vllm_bin: str = "") -> None:
        self._vllm_bin = vllm_bin
        self._process: asyncio.subprocess.Process | None = None
        self._cancelled = False

    @property
    def running(self) -> bool:
        return self._process is not None

    async def cancel(self) -> bool:
        """Stop the running benchmark. Returns whether there was one to stop.

        Signals the process group: `vllm bench serve` spawns workers, and killing only
        the parent would leave them sending requests at an engine we are about to tear
        down — the orphan case this project cares about, one level up from the engine
        itself.
        """
        process = self._process
        if process is None:
            return False

        self._cancelled = True
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        log.info("cancelling benchmark pid=%d", process.pid)
        return True

    async def run(self, request: BenchRequest, *, base_url: str) -> BenchResponse:
        if self._process is not None:
            raise BenchError("a benchmark is already running on this host")
        self._cancelled = False
        try:
            return await _run_benchmark(
                request,
                base_url=base_url,
                vllm_bin=self._vllm_bin,
                register=self._register,
                was_cancelled=lambda: self._cancelled,
            )
        finally:
            self._process = None

    def _register(self, process: asyncio.subprocess.Process) -> None:
        self._process = process


async def run_benchmark(
    request: BenchRequest, *, base_url: str, vllm_bin: str = ""
) -> BenchResponse:
    """Run one benchmark and return its verbatim result."""
    return await _run_benchmark(request, base_url=base_url, vllm_bin=vllm_bin)


async def _run_benchmark(
    request: BenchRequest,
    *,
    base_url: str,
    vllm_bin: str = "",
    register: Callable[[asyncio.subprocess.Process], None] | None = None,
    was_cancelled: Callable[[], bool] | None = None,
) -> BenchResponse:
    workdir = Path(tempfile.mkdtemp(prefix="vllmbench-bench-"))
    result_path = workdir / "result.json"
    argv = build_argv(request, base_url=base_url, result_path=result_path, vllm_bin=vllm_bin)

    log.info("running benchmark: %s", " ".join(argv))
    started = time.monotonic()

    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            # Same reasoning as the server: the benchmark client is vLLM too, and it
            # loads a tokenizer and may compile.
            env=child_environment(argv[0]),
        )
    except OSError as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        raise BenchError(f"could not launch the benchmark: {exc}") from exc

    if register is not None:
        register(process)

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=request.timeout_seconds
        )
    except TimeoutError:
        # Kill the group: the benchmark client spawns workers, and leaving them running
        # would keep loading the server we are about to declare idle.
        process.kill()
        await process.wait()
        shutil.rmtree(workdir, ignore_errors=True)
        raise BenchError(
            f"benchmark exceeded its {request.timeout_seconds:.0f}s timeout",
            FailureKind.BENCHMARK_TIMEOUT,
        ) from None

    out_tail = _tail(stdout.decode(errors="replace"))
    err_tail = _tail(stderr.decode(errors="replace"))
    duration = time.monotonic() - started

    # Checked before the exit code, because a cancelled client exits non-zero and
    # reporting that as a failed benchmark would mark a sweep the operator stopped as
    # broken rather than as stopped.
    if was_cancelled is not None and was_cancelled():
        shutil.rmtree(workdir, ignore_errors=True)
        raise BenchCancelled("benchmark was cancelled before it produced a result")

    try:
        if process.returncode != 0:
            raise BenchError(
                f"`vllm bench serve` exited with code {process.returncode}:\n"
                + "\n".join(err_tail[-20:] or out_tail[-20:])
            )

        if not result_path.is_file():
            # Exit code 0 with no result file means the CLI changed its output
            # behaviour. Failing here is essential: the alternative is recording a run
            # with no metrics and no indication anything went wrong.
            raise BenchError(
                f"benchmark reported success but wrote no result at {result_path}. "
                "This usually means --save-result or --result-filename changed."
            )

        try:
            raw = json.loads(result_path.read_text())
        except json.JSONDecodeError as exc:
            raise BenchError(f"benchmark result was not valid JSON: {exc}") from exc

        if not isinstance(raw, dict):
            raise BenchError(f"benchmark result was {type(raw).__name__}, expected an object")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    log.info("benchmark completed in %.1fs", duration)
    return BenchResponse(
        raw_result=raw,
        duration_seconds=duration,
        stdout_tail=out_tail,
        stderr_tail=err_tail,
    )


def _tail(text: str) -> list[str]:
    return list(deque(text.splitlines(), maxlen=OUTPUT_TAIL_LINES))
