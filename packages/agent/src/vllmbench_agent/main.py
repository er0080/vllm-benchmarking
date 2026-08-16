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
from vllmbench_agent.bench import BenchError, run_benchmark
from vllmbench_agent.hardware import (
    probe_cuda_version,
    probe_driver_version,
    probe_gpus,
    probe_vllm_version,
)
from vllmbench_agent.settings import AgentSettings
from vllmbench_agent.vllm_server import ServerError, VllmServer
from vllmbench_protocol import PROTOCOL_VERSION, __version__
from vllmbench_protocol.wire import (
    BenchRequest,
    BenchResponse,
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
    server = VllmServer()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Before serving. If a previous agent lifetime left a vLLM server holding VRAM,
        # the next start fails to allocate and nothing on the box explains why.
        server.reap_orphans()
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

        return HostInfo(
            protocol_version=PROTOCOL_VERSION,
            agent_version=__version__,
            hostname=socket.gethostname(),
            vllm_version=probe_vllm_version(),
            driver_version=probe_driver_version(),
            cuda_version=probe_cuda_version(),
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
            return await server.start(
                config_yaml=request.config_yaml,
                config_hash=request.config_hash,
                port=request.port,
                readiness_timeout_seconds=request.readiness_timeout_seconds,
            )
        except ServerError as exc:
            # 409 rather than 500: the request was well-formed, the host was not in a
            # state to satisfy it. The detail carries the server's own log tail, which
            # is the only thing that explains a model-load failure.
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

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

        if request.reset_caches_first:
            await server.reset_caches()

        try:
            return await run_benchmark(request, base_url=server.base_url())
        except BenchError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc

    return app


def main() -> None:
    """Console entry point: ``vllmbench-agent``."""
    import uvicorn

    settings = AgentSettings()  # type: ignore[call-arg]
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    )
    log.info(
        "agent %s (protocol %d) listening on %s:%d",
        __version__,
        PROTOCOL_VERSION,
        settings.host,
        settings.port,
    )
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
