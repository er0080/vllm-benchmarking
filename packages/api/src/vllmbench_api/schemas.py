"""Request and response models for the control-plane API.

Separate from ``vllmbench_protocol.wire``, which is the agent contract. Conflating them
would mean a change to what the UI displays could alter what the agent must send.
"""

from __future__ import annotations

import datetime as dt
import uuid

from pydantic import BaseModel, ConfigDict, Field


class GpuDeviceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    device_index: int
    name: str
    vram_bytes: int | None = None


class HostCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    agent_url: str = Field(min_length=1)


class HostOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    agent_url: str

    agent_version: str | None = None
    protocol_version: int | None = None
    vllm_version: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    gpu_count: int = 0
    synthetic_source: str | None = None
    last_seen_at: dt.datetime | None = None
    created_at: dt.datetime

    devices: list[GpuDeviceOut] = Field(default_factory=list)


class HostFacts(HostOut):
    """A host plus how its vLLM version compares to this build's reference.

    The comparison is surfaced rather than enforced. Benchmarking one vLLM version
    against another is a supported use of this tool, so a mismatch is information, never
    a blocker (CLAUDE.md, "vLLM version policy").
    """

    reference_vllm_version: str | None = None
    vllm_version_matches_reference: bool | None = None
