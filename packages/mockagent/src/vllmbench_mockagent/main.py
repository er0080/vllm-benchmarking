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

import logging
import os
import time

from fastapi import Depends, FastAPI

from vllmbench_agent.auth import token_dependency
from vllmbench_protocol import PROTOCOL_VERSION, __version__
from vllmbench_protocol.wire import GpuInfo, HealthResponse, HostInfo

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
            driver_version=MOCK_DRIVER_VERSION,
            cuda_version=MOCK_CUDA_VERSION,
            gpus=MOCK_GPUS,
            synthetic_source=SYNTHETIC_SOURCE,
        )

    return app


app = create_app()
