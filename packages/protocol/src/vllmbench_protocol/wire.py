"""Wire types for the control plane ↔ agent boundary.

These models are the contract. Both sides import them from here rather than each
declaring their own, because two hand-maintained copies of a schema drift, and the drift
shows up as a field silently arriving as None rather than as an error.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

AUTH_HEADER = "Authorization"
AUTH_SCHEME = "Bearer"


class _Wire(BaseModel):
    # Reject unknown fields. A newer agent sending a field this control plane does not
    # know about is a version mismatch we want to hear about, not absorb silently.
    model_config = ConfigDict(extra="forbid", frozen=True)


class GpuInfo(_Wire):
    """One physical device.

    Reported per device rather than summarized because multi-GPU is in scope and
    tensor parallelism is a first-class sweep dimension — a host-level summary cannot
    answer which devices a TP run actually used.
    """

    index: int = Field(ge=0)
    name: str
    uuid: str | None = None
    vram_bytes: int | None = Field(default=None, ge=0)


class HealthResponse(_Wire):
    """Liveness. Deliberately unauthenticated.

    A misconfigured token is one of the likeliest setup failures, and an operator needs
    to be able to tell "wrong token" from "agent not running" without guessing. Nothing
    here is sensitive on a trusted LAN.
    """

    status: str = "ok"
    agent_version: str
    protocol_version: int
    uptime_seconds: float = Field(ge=0)


class HostInfo(_Wire):
    """Everything the control plane needs to record provenance for a run.

    Point-in-time facts: the control plane copies these onto each run rather than
    joining to the host record, so that upgrading a driver tomorrow does not silently
    rewrite what yesterday's measurements claim to have run on.
    """

    protocol_version: int
    agent_version: str
    hostname: str

    vllm_version: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None

    gpus: list[GpuInfo] = Field(default_factory=list)

    # Set by the producer, never inferred by the consumer (invariant 7). A mock or a
    # CPU-backend agent names itself here, and the control plane marks every run it
    # produces accordingly. Inference downstream would eventually mark a synthetic run
    # as real, and the failure is silent.
    synthetic_source: str | None = None

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)
