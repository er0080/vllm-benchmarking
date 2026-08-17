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
import json
import logging
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path

from vllmbench_agent.hardware import resolve_vllm_binary, vllm_binary_search_detail
from vllmbench_protocol.wire import BenchRequest, BenchResponse

log = logging.getLogger(__name__)

OUTPUT_TAIL_LINES = 100


class BenchError(RuntimeError):
    """The benchmark did not produce a usable result."""


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


async def run_benchmark(
    request: BenchRequest, *, base_url: str, vllm_bin: str = ""
) -> BenchResponse:
    """Run one benchmark and return its verbatim result."""
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
        )
    except OSError as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        raise BenchError(f"could not launch the benchmark: {exc}") from exc

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
        raise BenchError(f"benchmark exceeded its {request.timeout_seconds:.0f}s timeout") from None

    out_tail = _tail(stdout.decode(errors="replace"))
    err_tail = _tail(stderr.decode(errors="replace"))
    duration = time.monotonic() - started

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
