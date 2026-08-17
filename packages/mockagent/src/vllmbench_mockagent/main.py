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
import logging
import os
import time

from fastapi import Depends, FastAPI, HTTPException, status

from vllmbench_agent.auth import token_dependency
from vllmbench_mockagent.synthetic import synthesize_bench_result, synthesize_telemetry
from vllmbench_protocol import PROTOCOL_VERSION, __version__
from vllmbench_protocol.wire import (
    BenchRequest,
    BenchResponse,
    EngineSampleWire,
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

# A model "load" long enough that progress UI has something to show, short enough that
# development stays fast. Real loads are minutes; pretending to take minutes would make
# the mock useless for its actual purpose.
MOCK_LOAD_SECONDS = float(os.environ.get("VLLMBENCH_MOCK_LOAD_SECONDS", "2.0"))
MOCK_BENCH_SECONDS = float(os.environ.get("VLLMBENCH_MOCK_BENCH_SECONDS", "3.0"))
MOCK_DRIVER_VERSION = "550.54.15"
MOCK_CUDA_VERSION = "12.4"


def create_app(token: str | None = None, protocol_version: int = PROTOCOL_VERSION) -> FastAPI:
    """Build the mock agent.

    ``protocol_version`` is overridable so tests can stand up a genuinely stale agent
    and exercise the control plane's refusal path, rather than reaching into module
    globals to simulate it.
    """
    token = token or os.environ.get("VLLMBENCH_TOKEN", "dev-token")
    started_at = time.monotonic()
    require_token = token_dependency(token)

    app = FastAPI(title="vLLM Benchmarking Mock Agent", version=__version__, docs_url="/docs")

    # Mirrors the real agent's single-server rule, so control-plane code that handles
    # "already running" is exercised here too.
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
        await asyncio.sleep(MOCK_BENCH_SECONDS)
        tp = _tp_from_config(state["config_yaml"])  # type: ignore[arg-type]
        raw = synthesize_bench_result(
            model=request.model,
            served_model_name=request.served_model_name,
            num_prompts=request.num_prompts,
            max_concurrency=request.max_concurrency,
            request_rate=request.request_rate,
            input_len=request.random_input_len or 512,
            output_len=request.random_output_len or 128,
            config_hash=str(state["config_hash"]),
            tensor_parallel_size=tp,
        )
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

        return BenchResponse(
            raw_result=raw,
            duration_seconds=MOCK_BENCH_SECONDS,
            stdout_tail=["Serving Benchmark Result (synthetic)"],
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            device_indices=devices,
            engine_samples=[EngineSampleWire(**s) for s in engine_samples],
            gpu_samples=[GpuSampleWire(**s) for s in gpu_samples],
            telemetry_interval_seconds=interval,
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
