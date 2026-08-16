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


# ---------------------------------------------------------------------------
# Configurations and workloads
# ---------------------------------------------------------------------------


class ConfigCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    # Native vLLM YAML, stored and executed verbatim (invariant 5).
    yaml: str = Field(min_length=1)
    notes: str | None = None


class ConfigOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    config_hash: str
    name: str
    yaml: str
    notes: str | None = None
    created_at: dt.datetime


class WorkloadCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    dataset_name: str = "random"
    dataset_path: str | None = None
    hf_name: str | None = None
    num_prompts: int = Field(default=200, ge=1)
    # None means unbounded. See Workload in the data model for why not a sentinel.
    request_rate: float | None = Field(default=None, gt=0)
    max_concurrency: int | None = Field(default=None, ge=1)
    burstiness: float | None = Field(default=None, gt=0)
    input_len: int | None = Field(default=None, ge=1)
    output_len: int | None = Field(default=None, ge=1)


class WorkloadOut(WorkloadCreate):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    workload_hash: str
    created_at: dt.datetime


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


class RunCreate(BaseModel):
    gpu_host_id: uuid.UUID
    server_config_id: uuid.UUID
    workload_id: uuid.UUID


class RunSummaryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    successful_requests: int | None = None
    failed_requests: int | None = None
    benchmark_duration_sec: float | None = None
    total_input_tokens: int | None = None
    total_generated_tokens: int | None = None

    request_throughput_req_sec: float | None = None
    output_token_throughput_tok_sec: float | None = None
    total_token_throughput_tok_sec: float | None = None
    # Invariant 8: the comparable figures. Charts default to these.
    output_token_throughput_per_gpu: float | None = None
    total_token_throughput_per_gpu: float | None = None
    peak_output_token_throughput_tok_sec: float | None = None
    peak_concurrent_requests: float | None = None

    ttft_ms_mean: float | None = None
    ttft_ms_median: float | None = None
    ttft_ms_p99: float | None = None
    ttft_ms_std: float | None = None
    tpot_ms_mean: float | None = None
    tpot_ms_median: float | None = None
    tpot_ms_p99: float | None = None
    tpot_ms_std: float | None = None
    itl_ms_mean: float | None = None
    itl_ms_median: float | None = None
    itl_ms_p99: float | None = None
    itl_ms_std: float | None = None


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: str
    queued_at: dt.datetime
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None

    server_config_id: uuid.UUID
    workload_id: uuid.UUID
    gpu_host_id: uuid.UUID

    # Provenance, denormalized onto the run so that editing a config later cannot
    # retroactively change what a finished run claims to have measured (invariant 6).
    config_hash: str
    workload_hash: str
    vllm_version: str | None = None
    agent_version: str | None = None
    gpu_model: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None

    gpu_count: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    device_indices: list[int] | None = None

    bench_client_location: str
    is_synthetic: bool
    synthetic_source: str | None = None
    initiated_by: str

    error: str | None = None
    log_excerpt: str | None = None

    summary: RunSummaryOut | None = None
