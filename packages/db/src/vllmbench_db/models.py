"""The results store.

Shape follows CLAUDE.md's data model conventions, which exist because this repository
produces measurements: a wrong number is worse than a wrong pixel, because it gets
recorded, charted, compared against last month, and acted on.

Three of those conventions drive most of what looks unusual here:

*Raw before derived.* ``Run.raw_result`` keeps the benchmark JSON verbatim next to the
flattened columns in ``RunSummary``. If the flattening turns out to be wrong — the
highest-consequence bug available in this codebase — the raw record is what lets us
recompute instead of re-running weeks of GPU time.

*Provenance is not optional.* A run that cannot state what produced it is not a valid
result, so the provenance columns on ``Run`` are denormalized copies rather than joins.
Editing a config must not retroactively change what a finished run claims to have run.

*Per-device, never aggregated at write time.* ``GpuSample`` is keyed by device. A
host-level average destroys the imbalance signal that makes a tensor-parallel run
diagnosable, and no amount of later querying can recover it.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from vllmbench_db.base import Base, created_at_column, uuid_pk
from vllmbench_db.enums import (
    BenchClientLocation,
    InitiatedBy,
    RunStatus,
    SweepStatus,
)


def _enum(python_enum: type, name: str) -> Enum:
    # values_callable keeps the stored labels equal to the enum *values* rather than
    # Python attribute names, so the database is readable without the code at hand.
    return Enum(
        python_enum,
        name=name,
        values_callable=lambda e: [member.value for member in e],
        native_enum=True,
    )


# ---------------------------------------------------------------------------
# Hosts and hardware
# ---------------------------------------------------------------------------


class GpuHost(Base):
    __tablename__ = "gpu_host"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(String(128), unique=True)
    agent_url: Mapped[str] = mapped_column(Text)

    # Last-seen facts, refreshed on every successful handshake. Point-in-time values
    # for a run live on Run itself, because these drift.
    agent_version: Mapped[str | None] = mapped_column(String(32))
    protocol_version: Mapped[int | None] = mapped_column(Integer)
    vllm_version: Mapped[str | None] = mapped_column(String(64))
    driver_version: Mapped[str | None] = mapped_column(String(64))
    cuda_version: Mapped[str | None] = mapped_column(String(32))
    gpu_count: Mapped[int] = mapped_column(Integer, default=0)
    last_seen_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    # Declared by the agent itself, never inferred here (invariant 7). A host backed by
    # the mock agent produces synthetic runs, and this is where that propagates from.
    # Inferring it instead — "the GPU model looks made up" — would eventually mark a
    # synthetic run as real, silently.
    synthetic_source: Mapped[str | None] = mapped_column(String(32))

    created_at: Mapped[dt.datetime] = created_at_column()

    # lazy="raise_on_sql" throughout: under the async engine an implicit lazy load
    # raises MissingGreenlet at runtime, in whatever request happens to touch it. This
    # turns that into an immediate, obvious error at the point of the mistake and forces
    # callers to say what they want loaded.
    devices: Mapped[list[GpuDevice]] = relationship(
        back_populates="host",
        cascade="all, delete-orphan",
        order_by="GpuDevice.device_index",
        lazy="raise_on_sql",
    )


class GpuDevice(Base):
    """One physical GPU. Multi-GPU is in scope, so device inventory is a table."""

    __tablename__ = "gpu_device"
    __table_args__ = (UniqueConstraint("gpu_host_id", "device_index"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    gpu_host_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("gpu_host.id", ondelete="CASCADE"), index=True
    )
    device_index: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(128))
    uuid_str: Mapped[str | None] = mapped_column("uuid", String(64))
    vram_bytes: Mapped[int | None] = mapped_column(BigInteger)

    host: Mapped[GpuHost] = relationship(back_populates="devices")


# ---------------------------------------------------------------------------
# What gets benchmarked
# ---------------------------------------------------------------------------


class Model(Base):
    __tablename__ = "model"
    __table_args__ = (UniqueConstraint("hf_id", "revision", "quantization"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    hf_id: Mapped[str] = mapped_column(String(256))
    revision: Mapped[str] = mapped_column(String(128), default="main")
    quantization: Mapped[str] = mapped_column(String(32), default="none")
    local_path: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = created_at_column()


class ServerConfig(Base):
    """A vLLM server configuration, stored as the YAML that actually runs.

    Invariant 5: what is stored is what is written to disk and passed to
    ``vllm serve --config``. No intermediate schema, so nothing can be lost in
    translation and nothing rots when vLLM adds a flag.
    """

    __tablename__ = "server_config"

    id: Mapped[uuid.UUID] = uuid_pk()
    # sha256 of the canonicalized YAML. Two runs claiming the same config must have
    # byte-identical effective configuration.
    config_hash: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(128))
    yaml: Mapped[str] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)

    model_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model.id"), index=True)
    # Lineage: which config this was derived from.
    parent_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("server_config.id"), index=True)

    created_at: Mapped[dt.datetime] = created_at_column()


class Workload(Base):
    """A benchmark workload: the arguments handed to ``vllm bench serve``."""

    __tablename__ = "workload"

    id: Mapped[uuid.UUID] = uuid_pk()
    workload_hash: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(128))

    dataset_name: Mapped[str] = mapped_column(String(64))
    dataset_path: Mapped[str | None] = mapped_column(Text)
    hf_name: Mapped[str | None] = mapped_column(String(256))

    num_prompts: Mapped[int] = mapped_column(Integer)
    # NULL means unbounded: --request-rate inf, --max-concurrency inf. NULL rather than a
    # sentinel because "infinite" is genuinely the absence of a limit, and a magic number
    # would eventually be averaged.
    request_rate: Mapped[float | None] = mapped_column()
    max_concurrency: Mapped[int | None] = mapped_column(Integer)
    burstiness: Mapped[float | None] = mapped_column()

    input_len: Mapped[int | None] = mapped_column(Integer)
    output_len: Mapped[int | None] = mapped_column(Integer)

    extra_args: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[dt.datetime] = created_at_column()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class Sweep(Base):
    __tablename__ = "sweep"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[SweepStatus] = mapped_column(
        _enum(SweepStatus, "sweep_status"), default=SweepStatus.DRAFT, index=True
    )

    gpu_host_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("gpu_host.id"), index=True)
    replicates: Mapped[int] = mapped_column(Integer, default=3)

    initiated_by: Mapped[InitiatedBy] = mapped_column(_enum(InitiatedBy, "initiated_by"))
    initiated_by_client: Mapped[str | None] = mapped_column(String(128))

    is_synthetic: Mapped[bool] = mapped_column(default=False, index=True)

    created_at: Mapped[dt.datetime] = created_at_column()
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)

    runs: Mapped[list[Run]] = relationship(back_populates="sweep", lazy="raise_on_sql")


class Run(Base):
    """One execution of one workload against one server configuration.

    Provenance columns are denormalized copies, deliberately. Joining to ``ServerConfig``
    for the config hash would mean an edit to a config retroactively changing what a
    finished run claims to have run — which is exactly the class of silent corruption
    this schema exists to prevent.
    """

    __tablename__ = "run"
    __table_args__ = (
        Index("ix_run_sweep_id_replicate_idx", "sweep_id", "replicate_idx"),
        CheckConstraint("gpu_count >= 1", name="gpu_count_positive"),
        CheckConstraint("tensor_parallel_size >= 1", name="tp_positive"),
        CheckConstraint("pipeline_parallel_size >= 1", name="pp_positive"),
        # Invariant 7, enforced in DDL: a synthetic run must name its source, and a real
        # run must not have one. Without this, "is_synthetic = false, source = mock_agent"
        # is representable, and whichever column downstream code trusts decides whether
        # fabricated numbers reach a chart.
        CheckConstraint(
            "(is_synthetic AND synthetic_source IS NOT NULL)"
            " OR (NOT is_synthetic AND synthetic_source IS NULL)",
            name="synthetic_source_matches_flag",
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    sweep_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("sweep.id"), index=True)
    replicate_idx: Mapped[int] = mapped_column(Integer, default=0)

    server_config_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("server_config.id"), index=True)
    workload_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workload.id"), index=True)
    gpu_host_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("gpu_host.id"), index=True)

    status: Mapped[RunStatus] = mapped_column(
        _enum(RunStatus, "run_status"), default=RunStatus.QUEUED, index=True
    )

    queued_at: Mapped[dt.datetime] = created_at_column()
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    # -- Provenance (invariant 6) -------------------------------------------------
    config_hash: Mapped[str] = mapped_column(String(64), index=True)
    workload_hash: Mapped[str] = mapped_column(String(64), index=True)
    vllm_version: Mapped[str | None] = mapped_column(String(64), index=True)
    agent_version: Mapped[str | None] = mapped_column(String(32))
    protocol_version: Mapped[int | None] = mapped_column(Integer)
    driver_version: Mapped[str | None] = mapped_column(String(64))
    cuda_version: Mapped[str | None] = mapped_column(String(32))
    gpu_model: Mapped[str | None] = mapped_column(String(128), index=True)
    dataset_identity: Mapped[str | None] = mapped_column(Text)

    # -- Parallelism topology (invariant 8) ---------------------------------------
    # Reported by the agent from what actually ran. Never parsed back out of the config
    # YAML after the fact, because the YAML is not proof of what the engine did.
    gpu_count: Mapped[int] = mapped_column(Integer, default=1)
    device_indices: Mapped[list[int] | None] = mapped_column(ARRAY(Integer))
    tensor_parallel_size: Mapped[int] = mapped_column(Integer, default=1, index=True)
    pipeline_parallel_size: Mapped[int] = mapped_column(Integer, default=1)

    # -- Comparability guards ------------------------------------------------------
    bench_client_location: Mapped[BenchClientLocation] = mapped_column(
        _enum(BenchClientLocation, "bench_client_location"),
        default=BenchClientLocation.LOOPBACK,
    )
    is_synthetic: Mapped[bool] = mapped_column(default=False, index=True)
    # Free text rather than an enum, deliberately. SyntheticSource lists the fakes we
    # know about, but an agent reporting an unfamiliar one must still be quarantined —
    # and with a native enum that insert would fail instead, turning "I do not recognise
    # this fake" into "the run errors". Failing closed here means recording it.
    synthetic_source: Mapped[str | None] = mapped_column(String(32))

    initiated_by: Mapped[InitiatedBy] = mapped_column(_enum(InitiatedBy, "initiated_by"))
    initiated_by_client: Mapped[str | None] = mapped_column(String(128))

    # -- Payload -------------------------------------------------------------------
    raw_result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    log_excerpt: Mapped[str | None] = mapped_column(Text)

    sweep: Mapped[Sweep | None] = relationship(back_populates="runs", lazy="raise_on_sql")
    summary: Mapped[RunSummary | None] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="raise_on_sql",
    )


class RunSummary(Base):
    """Flattened benchmark metrics, one row per run.

    The columns mirror ``vllm bench serve --save-result``. Anything not mapped lands in
    ``extra`` rather than being dropped, and the verbatim payload is still on
    ``Run.raw_result``.
    """

    __tablename__ = "run_summary"

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("run.id", ondelete="CASCADE"), primary_key=True
    )

    successful_requests: Mapped[int | None] = mapped_column(Integer)
    failed_requests: Mapped[int | None] = mapped_column(Integer)
    benchmark_duration_sec: Mapped[float | None] = mapped_column()
    total_input_tokens: Mapped[int | None] = mapped_column(BigInteger)
    total_generated_tokens: Mapped[int | None] = mapped_column(BigInteger)

    request_throughput_req_sec: Mapped[float | None] = mapped_column()
    output_token_throughput_tok_sec: Mapped[float | None] = mapped_column()
    total_token_throughput_tok_sec: Mapped[float | None] = mapped_column()
    peak_output_token_throughput_tok_sec: Mapped[float | None] = mapped_column()
    peak_concurrent_requests: Mapped[float | None] = mapped_column()

    # Invariant 8. Stored rather than computed at read time so that every consumer —
    # UI, MCP tools, CSV export — gets the same normalization instead of each dividing
    # by its own idea of the device count. A TP=4 run trivially out-throughputs TP=1 on
    # the aggregate figures above while being worse per device.
    output_token_throughput_per_gpu: Mapped[float | None] = mapped_column()
    total_token_throughput_per_gpu: Mapped[float | None] = mapped_column()

    ttft_ms_mean: Mapped[float | None] = mapped_column()
    ttft_ms_median: Mapped[float | None] = mapped_column()
    ttft_ms_p99: Mapped[float | None] = mapped_column()
    ttft_ms_std: Mapped[float | None] = mapped_column()

    tpot_ms_mean: Mapped[float | None] = mapped_column()
    tpot_ms_median: Mapped[float | None] = mapped_column()
    tpot_ms_p99: Mapped[float | None] = mapped_column()
    tpot_ms_std: Mapped[float | None] = mapped_column()

    itl_ms_mean: Mapped[float | None] = mapped_column()
    itl_ms_median: Mapped[float | None] = mapped_column()
    itl_ms_p99: Mapped[float | None] = mapped_column()
    itl_ms_std: Mapped[float | None] = mapped_column()

    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    run: Mapped[Run] = relationship(back_populates="summary", lazy="raise_on_sql")


# ---------------------------------------------------------------------------
# Telemetry — append-only
# ---------------------------------------------------------------------------


class EngineSample(Base):
    """vLLM ``/metrics`` sampled during a run.

    This is the table that answers *why* a configuration lost. A p99 TTFT says a config
    was slow; KV cache pressure and a growing waiting queue say what to change.
    """

    __tablename__ = "engine_sample"
    __table_args__ = (Index("ix_engine_sample_run_id_sampled_at", "run_id", "sampled_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("run.id", ondelete="CASCADE"))
    sampled_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))

    # A 0..1 fraction, stored exactly as vLLM emits it. vLLM calls the metric
    # `kv_cache_usage_perc` but reports 0.112 for 11.2%, so inheriting that suffix here
    # would invite a chart that renders an eighth-full cache as "0.1%".
    kv_cache_usage_fraction: Mapped[float | None] = mapped_column()
    num_requests_running: Mapped[int | None] = mapped_column(Integer)
    num_requests_waiting: Mapped[int | None] = mapped_column(Integer)

    # Counters, stored raw. vLLM exposes no hit-rate gauge — only these two totals — and
    # storing a rate computed at sample time would discard the information needed to
    # difference it across an arbitrary window later.
    num_preemptions_total: Mapped[int | None] = mapped_column(BigInteger)
    prefix_cache_queries_total: Mapped[int | None] = mapped_column(BigInteger)
    prefix_cache_hits_total: Mapped[int | None] = mapped_column(BigInteger)

    raw: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class GpuSample(Base):
    """NVML sampled per device during a run.

    Keyed by device on purpose. Averaging at write time would be cheaper and would
    destroy the only signal that makes a tensor-parallel run diagnosable — one device at
    60% while three sit at 95% is a finding; the mean of those four numbers is not.
    """

    __tablename__ = "gpu_sample"
    __table_args__ = (
        Index("ix_gpu_sample_run_id_gpu_index_sampled_at", "run_id", "gpu_index", "sampled_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("run.id", ondelete="CASCADE"))
    gpu_index: Mapped[int] = mapped_column(Integer)
    sampled_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))

    sm_utilization_pct: Mapped[float | None] = mapped_column()
    memory_used_bytes: Mapped[int | None] = mapped_column(BigInteger)
    power_watts: Mapped[float | None] = mapped_column()
    temperature_c: Mapped[float | None] = mapped_column()
    sm_clock_mhz: Mapped[int | None] = mapped_column(Integer)
    memory_clock_mhz: Mapped[int | None] = mapped_column(Integer)
