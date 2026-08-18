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
    ReplicateOrder,
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

    # -- Operational budgets -------------------------------------------------------
    # How long this host is allowed to take, in seconds. Defaults, overridable per
    # sweep. They live on the host because the host is what determines them: model load
    # time is disk and PCIe, and both limits are enforced there.
    #
    # These are limits, not measurements. Nothing downstream reads them; they exist so
    # that a hung run ends as a failure with a reason instead of sitting in
    # `benchmarking` forever, and so that raising one does not mean editing code.
    model_load_timeout_seconds: Mapped[int] = mapped_column(Integer, default=900)
    benchmark_timeout_seconds: Mapped[int] = mapped_column(Integer, default=3600)

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

    #: How this configuration first came to exist: the config it was edited from.
    #:
    #: "First" is load-bearing. Content addressing means the text is the identity, so
    #: submitting YAML that already exists returns the existing row — and if two people
    #: edit two different parents into byte-identical results, the second submission does
    #: not overwrite the first's parent. This records the derivation that created the row,
    #: which is a true and useful thing, and is deliberately not a claim to be a complete
    #: derivation graph.
    #:
    #: It is *not* what groups a config with its tensor-parallel variants; that is
    #: computed from the text (see `sweep_plan.config_family_text`), so it works for
    #: configs nobody derived from anything.
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
    # Recorded, not inferred: whether a point's replicates ran back-to-back or spread
    # across the sweep decides whether the spread means "repeatability" or "run-to-run
    # variance", and a chart drawing error bars is claiming one of them.
    replicate_order: Mapped[ReplicateOrder] = mapped_column(
        _enum(ReplicateOrder, "replicate_order"), default=ReplicateOrder.GROUPED
    )

    initiated_by: Mapped[InitiatedBy] = mapped_column(_enum(InitiatedBy, "initiated_by"))
    initiated_by_client: Mapped[str | None] = mapped_column(String(128))

    is_synthetic: Mapped[bool] = mapped_column(default=False, index=True)

    # Null means "use the host's default". A sweep knows things the host cannot: that
    # these points load a 70B, or that this workload sends fifty thousand prompts. Null
    # rather than a copy of the host value so that raising the host default also raises
    # every sweep that never had an opinion.
    model_load_timeout_seconds: Mapped[int | None] = mapped_column(Integer)
    benchmark_timeout_seconds: Mapped[int | None] = mapped_column(Integer)

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
        # A failed run names why. Not "should" — the orchestrator's floor is
        # FailureKind.INTERNAL, so there is no path that fails without a kind, and this
        # is what keeps that true if someone adds one.
        #
        # Added NOT VALID in the migration, so it binds new and updated rows but leaves
        # history alone. Runs that failed before this column existed keep an honest NULL;
        # backfilling them would mean guessing a kind from old free text, which is the
        # one thing FailureKind is written not to do.
        CheckConstraint(
            "status <> 'failed' OR failure_kind IS NOT NULL",
            name="failed_run_names_its_failure",
            postgresql_not_valid=True,
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    sweep_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("sweep.id"), index=True)
    replicate_idx: Mapped[int] = mapped_column(Integer, default=0)
    # Position in the sweep's plan. Explicit rather than derived from queued_at, because
    # the order is a decision — it is chosen to keep runs sharing a server config
    # adjacent, so the engine restarts once per config instead of once per run — and an
    # ordering that matters should not depend on timestamp ties.
    sweep_seq: Mapped[int | None] = mapped_column(Integer)

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

    #: Where this run came from, when it was not measured by this framework. Null for a
    #: run this control plane orchestrated; otherwise a description of the source, e.g.
    #: "vllm bench sweep serve · <experiment>".
    #:
    #: Never null for an imported run, because the provenance on such a run was *declared
    #: by an operator rather than observed* — the upstream output carries no vLLM version,
    #: GPU model or device count at all (ADR 0003). A GPU model NVML reported and one a
    #: person typed are different kinds of fact, and a chart that cannot tell them apart
    #: will eventually be asked to explain a discrepancy nobody can resolve.
    imported_from: Mapped[str | None] = mapped_column(String(200), index=True)

    initiated_by: Mapped[InitiatedBy] = mapped_column(_enum(InitiatedBy, "initiated_by"))
    initiated_by_client: Mapped[str | None] = mapped_column(String(128))

    # -- Payload -------------------------------------------------------------------
    raw_result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)

    #: Which class of failure this was, for a run that has one. See
    #: :class:`vllmbench_db.enums.FailureKind`.
    #:
    #: Free text and nullable rather than a native enum: an agent or a vLLM version we
    #: do not yet have a name for must still be *recorded*, and a native enum would fail
    #: the insert instead — losing the failure altogether, which is worse than filing it
    #: under the wrong heading. Indexed because the question it exists to answer is
    #: "what went wrong across these eleven runs", which is a GROUP BY.
    #:
    #: Never a substitute for `error`, which always holds the full text. The kind throws
    #: away exactly the detail an operator needs to act.
    failure_kind: Mapped[str | None] = mapped_column(String(32), index=True)

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


class RunTelemetryPruned(Base):
    """That a run's telemetry was deleted under a retention policy, rather than missing.

    Its own table, not a column on ``run``, for two independent reasons. A terminal run is
    immutable and a database trigger enforces it; and this is data *about* a run rather
    than a correction to what the run measured.

    Without this row, a run detail page with an empty timeline is indistinguishable from a
    run whose telemetry failed to record — and those call for opposite responses. One is
    the retention policy working; the other is a sampling bug that has been silently
    losing diagnostic data.
    """

    __tablename__ = "run_telemetry_pruned"

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("run.id", ondelete="CASCADE"), primary_key=True
    )
    pruned_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    #: The horizon in force when this ran, so a later reader can tell whether the absence
    #: is explained by the policy they are looking at or by an older, stricter one.
    horizon_days: Mapped[int] = mapped_column(Integer)


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


# ---------------------------------------------------------------------------
# Saved views
# ---------------------------------------------------------------------------


class SavedView(Base):
    """A named analysis view: which chart, over which runs, on which axes.

    Deliberately stores a *query*, never a set of run ids. A saved view reopened next
    month should include the runs measured since it was saved — that is what makes it a
    view of the data rather than a snapshot of it. Pinning run ids would produce something
    that silently stops tracking reality while continuing to look current, which is the
    worst of both.

    ``filters`` and ``options`` are ``jsonb`` rather than columns because they are
    interface state, not measurements. Nothing queries across them, no chart is drawn from
    them, and giving each a column would mean a migration every time a view gains a
    control. The "raw before derived" rule that governs results does not apply to a
    record of what somebody had selected.

    ``source`` is a column of its own, though, because it is the one field whose meaning
    is load-bearing: a view saved over synthetic runs must reopen as synthetic. Buried in
    a JSON blob it would be one typo away from a saved view quietly showing real numbers
    where a developer expected the mock's (invariant 7).
    """

    __tablename__ = "saved_view"
    __table_args__ = (UniqueConstraint("name"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text)

    #: Which analysis view this reopens — "pareto", "scaling", "load", "balance",
    #: "compare". Free text rather than a native enum so adding a view is not a
    #: migration; an unrecognised value falls back to the default view rather than
    #: failing to load.
    view: Mapped[str] = mapped_column(String(32))

    #: The population, held as a column for the reason in the class docstring.
    source: Mapped[str] = mapped_column(String(16), default="real")

    filters: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    options: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    created_at: Mapped[dt.datetime] = created_at_column()


class McpWriteAudit(Base):
    """Every write call the MCP surface received, including the ones it refused.

    Named for the surface it covers rather than "audit log", because it covers exactly
    one interface and a name that implied otherwise would be a trap: somebody would read
    an empty table and conclude nothing had changed anything.

    The MCP surface is the one that needs this, and the reason is not suspicion. A person
    who clicks *create sweep* remembers doing it, and `initiated_by` on the resulting rows
    records the rest. An agent working unattended remembers nothing, and — the part no
    other record covers — **a refused call leaves no trace anywhere else at all**. A
    `create_sweep` rejected because the host was busy, or because the matrix exceeded the
    bound, writes nothing to any table; without this, the only evidence it ever happened
    is in the agent's own context, which is gone by morning.

    Append-only in the same sense the sample tables are: rows are inserted and never
    updated. There is no endpoint that edits one.
    """

    __tablename__ = "mcp_write_audit"

    id: Mapped[uuid.UUID] = uuid_pk()
    called_at: Mapped[dt.datetime] = created_at_column()

    #: The tool name as the caller invoked it.
    tool: Mapped[str] = mapped_column(String(64), index=True)

    #: Who asked, as far as the surface can tell. Declared by the client rather than
    #: sniffed — the same trust model as `initiated_by`, and for the same reason: HTTP
    #: cannot honestly identify the thing on the other end, so a confident guess would be
    #: a wrong answer in an audit column.
    client: Mapped[str | None] = mapped_column(String(128))

    #: What was asked for, verbatim, as it arrived. Raw before derived: if the surface
    #: later turns out to have mishandled an argument, this is what says so.
    arguments: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    #: "succeeded", "refused" or "failed". Refused is the interesting one — it is the
    #: outcome that writes nothing anywhere else.
    outcome: Mapped[str] = mapped_column(String(16), index=True)

    #: Why, when it was not "succeeded".
    error: Mapped[str | None] = mapped_column(Text)

    #: What the call produced, when it produced something identifiable — a sweep id, a
    #: config hash. Enough to join this record to the thing it created.
    subject: Mapped[str | None] = mapped_column(String(128), index=True)


class ConfigJustification(Base):
    """The measurement somebody would point at to defend a configuration.

    A configuration on its own says what was set, never why. Six months later the YAML is
    a list of numbers with no argument attached, and the sweep that produced them is one
    of forty. This is the link back to the evidence, and it is the difference between a
    config you can defend and one you are afraid to touch.

    **A table rather than a column on ``server_config``, to avoid a circular foreign key.**
    ``run`` already points at ``server_config``; pointing back would make the two mutually
    dependent, which Postgres tolerates but which defeats any topological ordering over the
    schema — including the metadata-derived delete order the test fixtures rely on, which
    exists precisely because a hand-written one drifted. As a separate table this is a
    leaf, and the graph stays acyclic.

    One per configuration, enforced by a unique constraint: the question is "what is the
    argument for this config", and a list of five runs is not an argument.
    """

    __tablename__ = "config_justification"
    __table_args__ = (UniqueConstraint("server_config_id"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    server_config_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("server_config.id", ondelete="CASCADE"), index=True
    )
    #: The run must be one that actually used this configuration. Checked in the API rather
    #: than by constraint — the comparison is against ``run.config_hash``, which a foreign
    #: key cannot express — because a link that can point at the wrong evidence is worse
    #: than no link.
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("run.id"), index=True)
    #: Why this run settles it, in the author's words.
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = created_at_column()
