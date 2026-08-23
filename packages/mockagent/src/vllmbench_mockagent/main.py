"""Mock agent: the real agent's HTTP contract, backed by synthetic data.

This exists so the control plane and the frontend are fully developable on a laptop, and
so integration tests have a fixture that does not need hardware. It is also, per
CLAUDE.md, the integration-test fixture — when the real agent's contract changes, this
changes in the same PR.

Everything it reports is fake, and it says so: ``synthetic_source`` is set on every
response. The control plane marks runs from this agent as synthetic on that basis alone.
That is invariant 7 working as designed — the producer declares it, and no consumer ever
has to infer it from circumstantial evidence like "the GPU model looks made up".
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from collections import defaultdict
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status

from vllmbench_agent.auth import token_dependency
from vllmbench_agent.dataset import identify_dataset
from vllmbench_mockagent.synthetic import synthesize_bench_result, synthesize_telemetry
from vllmbench_protocol import NO_SPECULATION, PROTOCOL_VERSION, __version__
from vllmbench_protocol.failures import FAILURE_KIND_HEADER, FailureKind
from vllmbench_protocol.wire import (
    BenchRequest,
    BenchResponse,
    CancelResponse,
    EngineSampleWire,
    EnvironmentCheck,
    EnvironmentStatus,
    GpuInfo,
    GpuSampleWire,
    HealthResponse,
    HostInfo,
    ServerState,
    ServerStatus,
    StartServerRequest,
)

log = logging.getLogger("vllmbench.mockagent")

SYNTHETIC_SOURCE = "mock_agent"

# A plausible two-GPU host. Two rather than one on purpose: multi-GPU and tensor
# parallelism are in scope, and a single-device mock would let per-device handling break
# without any test noticing.
MOCK_GPUS = [
    GpuInfo(
        index=0,
        name="NVIDIA A100-SXM4-80GB",
        uuid="GPU-00000000-0000-0000-0000-000000000000",
        vram_bytes=85_899_345_920,
    ),
    GpuInfo(
        index=1,
        name="NVIDIA A100-SXM4-80GB",
        uuid="GPU-11111111-1111-1111-1111-111111111111",
        vram_bytes=85_899_345_920,
    ),
]

MOCK_VLLM_VERSION = "0.25.1"
#: A plausible size for a vLLM environment, so the number reads as a real one.
MOCK_DISTRIBUTIONS = 247

# A model "load" long enough that progress UI has something to show, short enough that
# development stays fast. Real loads are minutes; pretending to take minutes would make
# the mock useless for its actual purpose.
MOCK_LOAD_SECONDS = float(os.environ.get("VLLMBENCH_MOCK_LOAD_SECONDS", "2.0"))
MOCK_BENCH_SECONDS = float(os.environ.get("VLLMBENCH_MOCK_BENCH_SECONDS", "3.0"))
MOCK_DRIVER_VERSION = "550.54.15"
MOCK_CUDA_VERSION = "12.4"


# Failure injection. Set to a FailureKind value to make the next server start, or the
# next benchmark, fail that way — the same status code and the same
# X-Vllmbench-Failure-Kind header the real agent would send.
#
# CLAUDE.md names failure injection as part of what the mock is for, and this is the part
# of the contract that is otherwise untestable without a GPU: an out-of-memory engine
# start is trivial to provoke on real hardware and impossible to provoke on a laptop.
# Environment variables rather than an endpoint so that a compose-based dev stack can be
# put into a failing state without a client that knows how to ask.
# Read when the app is built rather than at import, and overridable as arguments to
# `create_app`. A module-level snapshot would make the setting sticky for the life of the
# process, which is fine for a container and wrong for a test suite: one test putting the
# mock into a failing state would leave every later test in it.
def _default_start_failure() -> str:
    return os.environ.get("VLLMBENCH_MOCK_START_FAILURE", "")


def _default_bench_failure() -> str:
    return os.environ.get("VLLMBENCH_MOCK_BENCH_FAILURE", "")


def _default_empty_result() -> bool:
    return os.environ.get("VLLMBENCH_MOCK_EMPTY_RESULT", "").lower() in ("1", "true", "yes")


def _environment() -> EnvironmentCheck:
    """The mock's environment report, injectable like every other mock failure.

    A real dependency conflict cannot be produced on a laptop without deliberately
    breaking a virtualenv, so injection is the only way the control plane's handling of one
    is reachable in a test — the same argument as the start and bench failure kinds.

    Read per call rather than at import, so a test can change it between requests.
    """
    forced = os.environ.get("VLLMBENCH_MOCK_ENVIRONMENT", "").strip().lower()
    if forced in ("conflicts", "1", "true", "yes"):
        return EnvironmentCheck(
            status=EnvironmentStatus.CONFLICTS,
            # Lifted from the `uv pip check` output that opened #60, so what a developer
            # sees locally is the shape of what a real GPU host will send.
            conflicts=[
                "vllm 0.25.1 requires fastapi[standard]<0.137.0,>=0.133.0, "
                "but fastapi 0.141.1 is installed"
            ],
            distributions=MOCK_DISTRIBUTIONS,
        )
    if forced == "unavailable":
        return EnvironmentCheck(
            status=EnvironmentStatus.UNAVAILABLE, detail="synthetic: the check was not run"
        )
    return EnvironmentCheck(status=EnvironmentStatus.OK, distributions=MOCK_DISTRIBUTIONS)


#: What vLLM returns when every request failed, which it reports as a *success*.
#:
#: Captured shape, not an invention: `bench_serve_all_requests_failed_v0.25.1.json` in the
#: protocol fixtures is the real thing, produced by benchmarking a model name the server
#: does not serve. The client exits 0, writes a result file, and fills every metric with
#: 0.00 — which on a latency axis is the best value there is.
#:
#: Reproducible here because it is otherwise unreachable without a live engine and a
#: deliberate misconfiguration, and it is the most dangerous payload this system can be
#: handed.
def _empty_result(raw: dict[str, Any]) -> dict[str, Any]:
    zeroed = {
        key: (0 if isinstance(value, int) else 0.0 if isinstance(value, float) else value)
        for key, value in raw.items()
    }
    zeroed["completed"] = 0
    zeroed["failed"] = raw.get("num_prompts", 8)
    return zeroed


# What the real agent's message looks like for each injectable kind. Taken from the same
# captured vLLM output the classifier's patterns were read off, so a control plane that
# ignores the header and reads the text reaches the same conclusion — which is exactly
# the older-agent path, and would otherwise never be exercised.
_FAILURE_DETAIL: dict[str, str] = {
    FailureKind.ENGINE_OUT_OF_MEMORY: (
        "vLLM exited with code 1 during startup.\n\nLikely cause:\n"
        "  ValueError: No available memory for the cache blocks. Try increasing "
        "`gpu_memory_utilization` when initializing the engine."
    ),
    FailureKind.ENGINE_CONFIG_REJECTED: (
        "vLLM exited with code 2 during startup.\n\nLikely cause:\n"
        "  vllm: error: unrecognized arguments: --no-such-flag"
    ),
    FailureKind.ENGINE_NOT_READY: "vLLM did not become ready within 900s",
    FailureKind.BENCHMARK_TIMEOUT: "benchmark exceeded its 3600s timeout",
    FailureKind.BENCHMARK_FAILED: "`vllm bench serve` exited with code 1",
}


def _injected(kind: str, status_code: int) -> HTTPException:
    """Build the failure the real agent would have raised."""
    return HTTPException(
        status_code=status_code,
        detail=_FAILURE_DETAIL.get(kind, f"injected failure: {kind}"),
        headers={FAILURE_KIND_HEADER: kind},
    )


def create_app(
    token: str | None = None,
    protocol_version: int = PROTOCOL_VERSION,
    *,
    start_failure: str | None = None,
    bench_failure: str | None = None,
    empty_result: bool | None = None,
) -> FastAPI:
    """Build the mock agent.

    ``protocol_version`` is overridable so tests can stand up a genuinely stale agent
    and exercise the control plane's refusal path, rather than reaching into module
    globals to simulate it.
    """
    token = token or os.environ.get("VLLMBENCH_TOKEN", "dev-token")
    started_at = time.monotonic()
    require_token = token_dependency(token)
    injected_start_failure = _default_start_failure() if start_failure is None else start_failure
    injected_bench_failure = _default_bench_failure() if bench_failure is None else bench_failure
    injected_empty_result = _default_empty_result() if empty_result is None else empty_result

    app = FastAPI(title="vLLM Benchmarking Mock Agent", version=__version__, docs_url="/docs")

    # Mirrors the real agent's single-server rule, so control-plane code that handles
    # "already running" is exercised here too.
    # Set while a synthetic benchmark is "running", so cancellation can interrupt it the
    # way the real agent interrupts a subprocess.
    cancel = asyncio.Event()

    # How many benchmarks this process has already run against each (config, workload).
    # Fed to the synthesizer as the replicate seed so a point's replicates differ from
    # each other, which is what makes the spread rendering in the analysis views
    # exercisable without a GPU. Still deterministic: the same sequence of requests to a
    # fresh mock produces the same sequence of numbers, so tests stay stable.
    #
    # The synthesizer has taken a replicate seed since it was written; nothing ever
    # passed one, so every replicate came back byte-identical and the error bars in the
    # Pareto view had zero width against the only data available in development.
    replicate_counts: dict[str, int] = defaultdict(int)

    state: dict[str, object] = {
        "state": ServerState.STOPPED,
        "config_hash": None,
        "config_yaml": None,
        "port": None,
        "started_at": None,
        "ready_at": None,
    }

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            agent_version=__version__,
            protocol_version=PROTOCOL_VERSION,
            uptime_seconds=time.monotonic() - started_at,
        )

    @app.get("/host-info", response_model=HostInfo, dependencies=[Depends(require_token)])
    async def host_info() -> HostInfo:
        return HostInfo(
            protocol_version=protocol_version,
            agent_version=__version__,
            hostname="mock-gpu-host",
            vllm_version=MOCK_VLLM_VERSION,
            vllm_probe_detail="synthetic: no vLLM is installed for the mock agent",
            driver_version=MOCK_DRIVER_VERSION,
            cuda_version=MOCK_CUDA_VERSION,
            gpus=MOCK_GPUS,
            environment=_environment(),
            synthetic_source=SYNTHETIC_SOURCE,
        )

    def _status() -> ServerStatus:
        return ServerStatus(
            state=state["state"],  # type: ignore[arg-type]
            config_hash=state["config_hash"],  # type: ignore[arg-type]
            pid=4242 if state["state"] is ServerState.READY else None,
            port=state["port"],  # type: ignore[arg-type]
            started_at=state["started_at"],  # type: ignore[arg-type]
            ready_at=state["ready_at"],  # type: ignore[arg-type]
            log_tail=["INFO synthetic vLLM: no real engine was started"],
            vllm_version=MOCK_VLLM_VERSION if state["state"] is ServerState.READY else None,
            served_model_name=_served_name_from_config(state["config_yaml"])  # type: ignore[arg-type]
            if state["state"] is ServerState.READY
            else None,
            tensor_parallel_size=_tp_from_config(state["config_yaml"]),  # type: ignore[arg-type]
            pipeline_parallel_size=1,
            speculative_method=_speculation_from_config(state["config_yaml"])[0]  # type: ignore[arg-type]
            if state["state"] is ServerState.READY
            else None,
            speculative_tokens=_speculation_from_config(state["config_yaml"])[1]  # type: ignore[arg-type]
            if state["state"] is ServerState.READY
            else None,
            # Mirrors the real agent: devices are the ones "observed" to be in use, and
            # the mock pretends the request was honoured up to its two-GPU inventory.
            device_indices=list(
                range(min(_tp_from_config(state["config_yaml"]), len(MOCK_GPUS)))  # type: ignore[arg-type]
            )
            if state["state"] is ServerState.READY
            else None,
        )

    @app.get("/server", response_model=ServerStatus, dependencies=[Depends(require_token)])
    async def server_status() -> ServerStatus:
        return _status()

    @app.post("/server/start", response_model=ServerStatus, dependencies=[Depends(require_token)])
    async def server_start(request: StartServerRequest) -> ServerStatus:
        if state["state"] in (ServerState.STARTING, ServerState.READY):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"a server is already {state['state']}; stop it before starting another",
            )
        if injected_start_failure:
            state.update(state=ServerState.STOPPED, ready_at=None)
            raise _injected(injected_start_failure, status.HTTP_409_CONFLICT)

        state.update(
            state=ServerState.STARTING,
            config_hash=request.config_hash,
            config_yaml=request.config_yaml,
            port=request.port,
            started_at=time.time(),
            ready_at=None,
        )
        await asyncio.sleep(MOCK_LOAD_SECONDS)
        state.update(state=ServerState.READY, ready_at=time.time())
        return _status()

    @app.post("/server/stop", response_model=ServerStatus, dependencies=[Depends(require_token)])
    async def server_stop() -> ServerStatus:
        state.update(state=ServerState.STOPPED, ready_at=None)
        return _status()

    @app.post("/bench", response_model=BenchResponse, dependencies=[Depends(require_token)])
    async def bench(request: BenchRequest) -> BenchResponse:
        if state["state"] is not ServerState.READY:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"no ready server to benchmark (state={state['state']})",
            )
        if injected_bench_failure:
            raise _injected(injected_bench_failure, status.HTTP_422_UNPROCESSABLE_ENTITY)

        cancel.clear()
        # Race the synthetic duration against a cancellation, mirroring the real agent
        # where the wait is on a subprocess that can be signalled.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(cancel.wait(), timeout=MOCK_BENCH_SECONDS)
        if cancel.is_set():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="benchmark was cancelled before it produced a result",
            )

        tp = _tp_from_config(state["config_yaml"])  # type: ignore[arg-type]
        point = f"{state['config_hash']}:{request.max_concurrency}:{request.num_prompts}"
        replicate = replicate_counts[point]
        replicate_counts[point] = replicate + 1
        raw = synthesize_bench_result(
            model=request.model,
            served_model_name=request.served_model_name,
            num_prompts=request.num_prompts,
            max_concurrency=request.max_concurrency,
            request_rate=request.request_rate,
            input_len=request.random_input_len or 512,
            output_len=request.random_output_len or 128,
            config_hash=str(state["config_hash"]),
            replicate_seed=str(replicate),
            tensor_parallel_size=tp,
        )
        if injected_empty_result:
            raw = _empty_result(raw)
        devices = list(range(tp))
        interval = request.telemetry_interval_seconds or 1.0
        # The synthetic benchmark takes seconds while a real one takes minutes, so the
        # window is stretched to a realistic length. A three-sample timeline would let
        # the chart look finished without ever having rendered a series.
        engine_samples, gpu_samples = synthesize_telemetry(
            duration_seconds=max(MOCK_BENCH_SECONDS, 120.0),
            interval_seconds=interval,
            device_indices=devices,
            max_concurrency=request.max_concurrency,
            config_hash=str(state["config_hash"]),
            tensor_parallel_size=tp,
        )

        method, tokens = _speculation_from_config(state["config_yaml"])  # type: ignore[arg-type]
        return BenchResponse(
            raw_result=raw,
            duration_seconds=MOCK_BENCH_SECONDS,
            stdout_tail=["Serving Benchmark Result (synthetic)"],
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            device_indices=devices,
            speculative_method=method,
            speculative_tokens=tokens,
            dataset_identity=identify_dataset(request),
            engine_samples=[EngineSampleWire(**s) for s in engine_samples],
            gpu_samples=[GpuSampleWire(**s) for s in gpu_samples],
            telemetry_interval_seconds=interval,
        )

    @app.post("/bench/cancel", response_model=CancelResponse, dependencies=[Depends(require_token)])
    async def bench_cancel() -> CancelResponse:
        running = not cancel.is_set()
        cancel.set()
        return CancelResponse(
            cancelled=running,
            detail="synthetic benchmark signalled" if running else "no benchmark was running",
        )

    return app


def _served_name_from_config(config_yaml: str | None) -> str | None:
    """Echo the alias a real engine would serve under.

    The real agent reads this from /v1/models. The mock has no engine, so it derives the
    same answer from the config — which keeps the control plane's "prefer the served
    name" path exercised without hardware.
    """
    if not config_yaml:
        return None
    found: dict[str, str] = {}
    for line in config_yaml.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        name = key.strip().replace("-", "_")
        if name in ("model", "served_model_name"):
            cleaned = value.strip().strip("\"'")
            if cleaned:
                found[name] = cleaned
    return found.get("served_model_name") or found.get("model")


def _speculation_from_config(config_yaml: str | None) -> tuple[str, int]:
    """Echo the speculative settings the caller's config asked for.

    Reading this out of the YAML is exactly what invariant 8 forbids the *real* agent from
    doing, and the reason is that the YAML states an intention while the engine states an
    outcome. The mock has no engine, so it has no outcome to report and nothing better to
    echo — and every run it produces is quarantined under invariant 7 anyway. That is the
    whole difference: this is a fake being transparently a fake, not a shortcut that would
    let a real measurement claim a topology nothing ran.

    Matching `speculative-config: {"method": "mtp", "num_speculative_tokens": 3}`, the
    single-line JSON form vLLM's own config files use.
    """
    if not config_yaml:
        return NO_SPECULATION, 0
    for line in config_yaml.splitlines():
        key, _, value = line.partition(":")
        if key.strip() not in ("speculative_config", "speculative-config"):
            continue
        try:
            parsed = json.loads(value.strip())
            method = str(parsed["method"])
            tokens = int(parsed["num_speculative_tokens"])
        except (ValueError, KeyError, TypeError):
            # A config we cannot read is not a config that says "off". The real agent
            # would answer None here; the mock has to answer something, and answering
            # "not speculating" would be inventing a fact.
            return NO_SPECULATION, 0
        return method, max(1, tokens)
    return NO_SPECULATION, 0


def _tp_from_config(config_yaml: str | None) -> int:
    """Read tensor_parallel_size out of the config the caller sent.

    The real agent reports what the engine actually did; the mock has no engine, so it
    echoes the request. Doing this at all matters because it lets TP sweeps — and the
    per-GPU normalization that makes them comparable — be exercised without hardware.
    """
    if not config_yaml:
        return 1
    for line in config_yaml.splitlines():
        key, _, value = line.partition(":")
        if key.strip() in ("tensor_parallel_size", "tensor-parallel-size"):
            try:
                return max(1, int(value.strip()))
            except ValueError:
                return 1
    return 1


app = create_app()
