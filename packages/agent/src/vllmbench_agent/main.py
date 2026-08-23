"""The GPU-host agent.

Runs natively on the machine under test, installed by uv into the same environment as
vLLM (invariant 3: no container runtime there). Its resource footprint is part of its
contract — everything it does must be cheap enough not to perturb the measurement it is
taking.

Milestone 0.1.0 scope: health, host facts, and authentication. Server lifecycle and
benchmark execution land in 0.2.0.
"""

from __future__ import annotations

import logging
import socket
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status

from vllmbench_agent.auth import token_dependency
from vllmbench_agent.bench import BenchCancelled, BenchError, BenchRunner
from vllmbench_agent.dataset import identify_dataset
from vllmbench_agent.environment import log_environment, probe_environment
from vllmbench_agent.hardware import (
    probe_cuda_version,
    probe_driver_version,
    probe_gpus,
    probe_peer_access,
    probe_vllm_version,
)
from vllmbench_agent.settings import AgentSettings
from vllmbench_agent.telemetry import TelemetrySampler
from vllmbench_agent.vllm_server import ServerError, VllmServer
from vllmbench_agent.workspace import DiskFull, disk_space, require_headroom, sweep_stale_workdirs
from vllmbench_protocol import PROTOCOL_VERSION, __version__
from vllmbench_protocol.failures import FAILURE_KIND_HEADER, FailureKind
from vllmbench_protocol.logging import configure_logging
from vllmbench_protocol.wire import (
    BenchRequest,
    BenchResponse,
    CancelResponse,
    HealthResponse,
    HostInfo,
    ServerState,
    ServerStatus,
    StartServerRequest,
)

log = logging.getLogger("vllmbench.agent")


def create_app(settings: AgentSettings | None = None) -> FastAPI:
    settings = settings or AgentSettings()  # type: ignore[call-arg]
    started_at = time.monotonic()
    require_token = token_dependency(settings.token)
    server = VllmServer(vllm_bin=settings.vllm_bin)
    bench_runner = BenchRunner(vllm_bin=settings.vllm_bin)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Before serving. If a previous agent lifetime left a vLLM server holding VRAM,
        # the next start fails to allocate and nothing on the box explains why.
        server.reap_orphans()
        # Same principle, one level down: a SIGKILL skips the `finally` that removes a
        # working directory, and nothing else on the box knows those were ours.
        sweep_stale_workdirs()
        space = disk_space()
        log.info(
            "%.1f GB free on %s (%.0f%%)",
            space.free_bytes / 1e9,
            space.path,
            space.free_fraction * 100,
        )
        # Said once at startup, because the moment somebody can act on a broken
        # environment is the moment they finish installing into it — not the moment a
        # sweep produces a number nobody trusts. Never fatal: see environment.py.
        log_environment(probe_environment())
        try:
            yield
        finally:
            # A clean shutdown must not leak VRAM. The reaper covers the unclean case.
            await server.stop()

    app = FastAPI(
        title="vLLM Benchmarking Agent",
        version=__version__,
        docs_url="/docs",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Unauthenticated by design.

        A wrong token is among the likeliest setup failures, and an operator needs to
        distinguish it from "agent not running" without guessing. Nothing returned here
        is sensitive on the trusted LAN this agent is meant to live on.
        """
        return HealthResponse(
            status="ok",
            agent_version=__version__,
            protocol_version=PROTOCOL_VERSION,
            uptime_seconds=time.monotonic() - started_at,
        )

    @app.get("/host-info", response_model=HostInfo, dependencies=[Depends(require_token)])
    async def host_info() -> HostInfo:
        gpus = probe_gpus()
        if not gpus:
            # Not an error: the agent is expected to run on a developer laptop. The
            # control plane decides what to do with a host that reports no devices.
            log.warning("no NVIDIA devices detected on this host")

        # Host-wide, across every device. What a run records is narrower and is computed
        # when the benchmark finishes, over the devices the engine actually used.
        peer_access, peer_access_detail = probe_peer_access()

        vllm_version, probe_detail = probe_vllm_version(settings.vllm_bin)
        # Recomputed per call rather than cached from startup: installing something into
        # this virtualenv while the agent runs is exactly how the environment changes, and
        # a cached answer would keep reporting the world as it was at boot. It costs a few
        # hundred milliseconds against a run that takes minutes, and it happens before the
        # engine starts, so it perturbs nothing being measured.
        return HostInfo(
            environment=probe_environment(),
            protocol_version=PROTOCOL_VERSION,
            agent_version=__version__,
            hostname=socket.gethostname(),
            vllm_version=vllm_version,
            vllm_probe_detail=probe_detail,
            driver_version=probe_driver_version(),
            cuda_version=probe_cuda_version(),
            peer_access=peer_access,
            peer_access_detail=peer_access_detail,
            gpus=gpus,
            # The real agent is never synthetic. The mock overrides this, and the
            # control plane trusts the producer rather than inferring (invariant 7).
            synthetic_source=None,
        )

    # --- vLLM server lifecycle ---------------------------------------------------

    @app.get("/server", response_model=ServerStatus, dependencies=[Depends(require_token)])
    async def server_status() -> ServerStatus:
        return server.status()

    @app.post("/server/start", response_model=ServerStatus, dependencies=[Depends(require_token)])
    async def server_start(request: StartServerRequest) -> ServerStatus:
        try:
            # Before the model load, not after. vLLM writes a compile cache and may pull
            # weights; discovering there is no room for either an hour in is the failure
            # this converts into an immediate refusal.
            require_headroom(settings.min_free_disk_bytes)
            return await server.start(
                config_yaml=request.config_yaml,
                config_hash=request.config_hash,
                port=request.port,
                readiness_timeout_seconds=request.readiness_timeout_seconds,
            )
        except DiskFull as exc:
            raise HTTPException(
                status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
                detail=str(exc),
                headers={FAILURE_KIND_HEADER: FailureKind.HOST_DISK_FULL.value},
            ) from exc
        except ServerError as exc:
            # 409 rather than 500: the request was well-formed, the host was not in a
            # state to satisfy it. The detail carries the server's own log tail, which
            # is the only thing that explains a model-load failure, and the header
            # carries this side's verdict on what kind of failure it was.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
                headers={FAILURE_KIND_HEADER: exc.kind.value},
            ) from exc

    @app.post("/server/stop", response_model=ServerStatus, dependencies=[Depends(require_token)])
    async def server_stop() -> ServerStatus:
        return await server.stop()

    # --- benchmarking --------------------------------------------------------------

    @app.post("/bench", response_model=BenchResponse, dependencies=[Depends(require_token)])
    async def bench(request: BenchRequest) -> BenchResponse:
        if server.state is not ServerState.READY:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"no ready server to benchmark (state={server.state}). "
                    "Benchmarking a server that is still loading measures the loader."
                ),
            )

        try:
            require_headroom(settings.min_free_disk_bytes)
        except DiskFull as exc:
            raise HTTPException(
                status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
                detail=str(exc),
                headers={FAILURE_KIND_HEADER: FailureKind.HOST_DISK_FULL.value},
            ) from exc

        if request.reset_caches_first:
            try:
                await server.reset_caches()
            except ServerError as exc:
                # Refusing before measuring, not after. The alternative is a run that
                # completes, looks fine, and quietly reports a warm cache from the
                # previous sweep point as this configuration's number.
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=str(exc),
                    headers={FAILURE_KIND_HEADER: exc.kind.value},
                ) from exc

        # Computed here rather than in the runner: it describes the request, so it is
        # recorded even for a benchmark that goes on to fail, and it costs one stat and a
        # read of a file the benchmark is about to read anyway.
        dataset_identity = identify_dataset(request)

        # Started after the cache reset and stopped after the client exits, so the
        # window the telemetry covers is the window the benchmark measured. Sampling
        # across the reset would put a KV-cache cliff in the series that belongs to
        # neither run.
        sampler = TelemetrySampler(
            base_url=server.base_url(),
            device_indices=server.status().device_indices,
            interval_seconds=request.telemetry_interval_seconds
            or settings.telemetry_interval_seconds,
        )
        sampler.start()

        try:
            response = await bench_runner.run(request, base_url=server.base_url())
        except BenchCancelled as exc:
            # 409 rather than 422: nothing about the request was wrong, the host was
            # asked to stop. The orchestrator records the run as cancelled, not failed.
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except BenchError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
                headers={FAILURE_KIND_HEADER: exc.kind.value},
            ) from exc
        finally:
            # Unconditional: a failed benchmark must not leave a sampler running against
            # the engine that the next one is about to measure.
            await sampler.stop()

        status_now = server.status()
        return response.model_copy(
            update={
                "engine_samples": sampler.engine_samples,
                "gpu_samples": sampler.gpu_samples,
                "telemetry_decimated": sampler.decimated,
                "telemetry_interval_seconds": sampler.effective_interval_seconds,
                # Read from the engine that just served this benchmark, so the run
                # records what ran rather than what its YAML asked for.
                "speculative_method": status_now.speculative_method,
                "speculative_tokens": status_now.speculative_tokens,
                "dataset_identity": dataset_identity,
                # Scoped to the devices this engine actually ran on, not to the host's.
                # A single-device run reports SINGLE_DEVICE even where the host's other
                # pairs are fine, so a TP=1 control keeps comparing against itself across
                # a change to the interconnect.
                "peer_access": probe_peer_access(status_now.device_indices)[0],
            }
        )

    @app.post("/bench/cancel", response_model=CancelResponse, dependencies=[Depends(require_token)])
    async def bench_cancel() -> CancelResponse:
        """Stop the benchmark in flight.

        Deliberately does not stop the server. Cancelling a sweep and tearing down the
        engine are separate decisions — the caller may want to cancel one point and keep
        the loaded model for the next — and the orchestrator issues both when it means
        both.
        """
        cancelled = await bench_runner.cancel()
        return CancelResponse(
            cancelled=cancelled,
            detail="benchmark signalled" if cancelled else "no benchmark was running",
        )

    return app


def main() -> None:
    """Console entry point: ``vllmbench-agent``."""
    import uvicorn

    settings = AgentSettings()  # type: ignore[call-arg]
    # The token is registered before anything else runs, so there is no window in which
    # a startup failure could quote it.
    configure_logging("agent", level=settings.log_level, secrets=[settings.token])
    log.info(
        "agent %s (protocol %d) listening on %s:%d",
        __version__,
        PROTOCOL_VERSION,
        settings.host,
        settings.port,
    )
    # log_config=None: uvicorn otherwise installs its own handlers over ours, and its
    # access log would then bypass the redaction entirely.
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_config=None)
