"""Request and response models for the control-plane API.

Separate from ``vllmbench_protocol.wire``, which is the agent contract. Conflating them
would mean a change to what the UI displays could alter what the agent must send.
"""

from __future__ import annotations

import datetime as dt
import uuid

from pydantic import BaseModel, ConfigDict, Field

from vllmbench_db.enums import ReplicateOrder


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


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class EngineSampleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sampled_at: dt.datetime
    num_requests_running: int | None = None
    num_requests_waiting: int | None = None
    # 0..1, exactly as vLLM emits it. The client multiplies for display; the API does not
    # pre-scale, so a consumer reading the raw series is never guessing about units.
    kv_cache_usage_fraction: float | None = None
    num_preemptions_total: int | None = None
    prefix_cache_queries_total: int | None = None
    prefix_cache_hits_total: int | None = None


class GpuSampleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sampled_at: dt.datetime
    gpu_index: int
    sm_utilization_pct: float | None = None
    memory_used_bytes: int | None = None
    power_watts: float | None = None
    temperature_c: float | None = None
    sm_clock_mhz: int | None = None
    memory_clock_mhz: int | None = None


class RunTelemetryOut(BaseModel):
    """Everything sampled during one run.

    Returned as two flat series rather than pre-joined per timestamp: engine and GPU
    sampling can legitimately disagree about which ticks succeeded (a scrape can fail
    while NVML answers), and zipping them here would either invent readings or drop
    good ones.
    """

    run_id: uuid.UUID
    engine: list[EngineSampleOut]
    gpu: list[GpuSampleOut]
    # The devices that actually produced samples, so a client can build one series per
    # device without scanning the whole payload first.
    gpu_indices: list[int]
    # Present so a chart can say "no telemetry" rather than drawing an empty axis and
    # leaving the reader to wonder whether the engine was idle.
    sample_count: int


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


class SweepCreate(BaseModel):
    """A matrix of server configs by workloads, run `replicates` times each.

    ``tensor_parallel_sizes`` turns TP into an axis: each base config is derived into one
    variant per value, and each variant is stored as an ordinary content-addressed
    config. Omit it to sweep the configs exactly as written.
    """

    name: str = Field(min_length=1, max_length=128)
    description: str | None = None
    gpu_host_id: uuid.UUID

    server_config_ids: list[uuid.UUID] = Field(min_length=1)
    workload_ids: list[uuid.UUID] = Field(min_length=1)

    # None means "leave each config's own tensor-parallel-size alone". An empty list would
    # mean the same thing but reads like "no TP values", so it is rejected rather than
    # silently treated as absent.
    tensor_parallel_sizes: list[int] | None = Field(default=None, min_length=1)

    # Three by default: CLAUDE.md asks charts to render the spread, and two points make a
    # range rather than a distribution.
    replicates: int = Field(default=3, ge=1, le=25)
    replicate_order: ReplicateOrder = ReplicateOrder.GROUPED


class SweepProgress(BaseModel):
    """Counts by run status, so a caller does not have to fetch every run to draw a bar."""

    total: int = 0
    queued: int = 0
    starting: int = 0
    benchmarking: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0

    @property
    def terminal(self) -> int:
        return self.succeeded + self.failed + self.cancelled


class SweepOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None = None
    status: str
    gpu_host_id: uuid.UUID
    replicates: int
    replicate_order: str
    initiated_by: str
    is_synthetic: bool
    created_at: dt.datetime
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    error: str | None = None

    progress: SweepProgress = Field(default_factory=SweepProgress)
    # How many engine restarts the plan implies. Most of a sweep's wall clock is model
    # loading, so this is the number that predicts how long it will take — and the one
    # that makes the cost of interleaving replicates visible before committing to it.
    engine_starts: int = 0


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


class MetricOut(BaseModel):
    """A chartable measurement and how to read it.

    Shipped with the data rather than duplicated in the frontend so that axis direction
    and per-GPU status have one definition. A view that had to remember on its own which
    metrics are "lower is better" would eventually draw a Pareto frontier upside down.
    """

    key: str
    label: str
    unit: str
    better: str
    per_gpu: bool = False
    description: str = ""


class SpreadOut(BaseModel):
    """One metric across a point's replicates.

    ``median`` is the value; the rest is the uncertainty CLAUDE.md requires rendering.
    The raw ``values`` are included because with three replicates a band is not enough
    to see whether the odd one out was high or low.
    """

    n: int
    median: float
    mean: float
    min: float
    max: float
    values: list[float]
    relative_range: float | None = None


class PointOut(BaseModel):
    point_id: str
    config_hash: str
    # Points sharing this are the same engine configuration at different tensor-parallel
    # widths — the only grouping under which a scaling curve means anything.
    config_family: str = ""
    config_name: str
    workload_hash: str
    workload_name: str

    tensor_parallel_size: int
    pipeline_parallel_size: int
    gpu_count: int
    max_concurrency: int | None = None
    request_rate: float | None = None
    num_prompts: int = 0

    replicates: int
    run_ids: list[uuid.UUID] = Field(default_factory=list)
    sweep_ids: list[uuid.UUID] = Field(default_factory=list)
    # What the band means — repeatability, run-to-run variance, or drift between
    # sittings — and a sentence saying so, for the tooltip that has to justify it.
    spread_basis: str
    spread_note: str
    latest_finished_at: dt.datetime | None = None

    on_pareto_frontier: bool = False
    metrics: dict[str, SpreadOut] = Field(default_factory=dict)


class GroupOut(BaseModel):
    """Points that may legitimately share a chart.

    The API partitions rather than the client, so a view cannot overlay two vLLM
    versions by forgetting to check: it is never handed them in one series.
    """

    group_id: str
    label: str
    gpu_host_id: uuid.UUID
    gpu_host_name: str
    gpu_model: str | None = None
    vllm_version: str | None = None
    bench_client_location: str

    warnings: list[str] = Field(default_factory=list)
    run_count: int = 0
    points: list[PointOut] = Field(default_factory=list)
    pareto_point_ids: list[str] = Field(default_factory=list)


class ExcludedOut(BaseModel):
    """Runs the filters matched that no chart will show.

    A missing point and a point whose every replicate failed look identical on a chart
    and mean opposite things, so the counts travel with the data.
    """

    failed: int = 0
    cancelled: int = 0
    unfinished: int = 0
    succeeded_without_summary: int = 0
    other_source: int = 0
    other_source_name: str = "synthetic"


class AnalysisOut(BaseModel):
    source: str
    run_count: int
    truncated: bool = False
    limit: int = 0
    pareto_x: str
    pareto_y: str
    metrics: list[MetricOut] = Field(default_factory=list)
    excluded: ExcludedOut = Field(default_factory=ExcludedOut)
    groups: list[GroupOut] = Field(default_factory=list)


class ScalingStepOut(BaseModel):
    """One tensor-parallel width on a scaling curve."""

    tensor_parallel_size: int
    gpu_count: int
    point_id: str
    config_name: str
    is_baseline: bool = False

    #: Aggregate throughput relative to the baseline width — what an operator feels.
    speedup: float | None = None
    #: Per-GPU throughput relative to the baseline — parallel efficiency. 1.0 means every
    #: added device pulled its weight.
    efficiency: float | None = None

    per_gpu: SpreadOut | None = None
    aggregate_median: float | None = None


class ScalingCurveOut(BaseModel):
    """One configuration's response to more devices, holding the workload fixed."""

    family: str
    config_name: str
    workload_hash: str
    workload_name: str
    max_concurrency: int | None = None
    baseline_tp: int
    #: False when the narrowest width measured was itself already parallel. An efficiency
    #: figure against a TP=2 baseline is not the parallel efficiency a reader assumes,
    #: and the view has to say so rather than print a number that reads as one.
    baseline_is_single_gpu: bool = True
    steps: list[ScalingStepOut] = Field(default_factory=list)


class ScalingGroupOut(BaseModel):
    group_id: str
    label: str
    gpu_host_id: uuid.UUID
    gpu_host_name: str
    gpu_model: str | None = None
    vllm_version: str | None = None
    bench_client_location: str
    warnings: list[str] = Field(default_factory=list)
    run_count: int = 0
    curves: list[ScalingCurveOut] = Field(default_factory=list)


class ScalingOut(BaseModel):
    source: str
    run_count: int
    truncated: bool = False
    limit: int = 0
    #: Which per-GPU metric the curves were built from. Efficiency is only meaningful for
    #: a per-GPU figure, so the endpoint refuses anything else rather than dividing two
    #: aggregates and calling the ratio efficiency.
    metric: str
    metrics: list[MetricOut] = Field(default_factory=list)
    excluded: ExcludedOut = Field(default_factory=ExcludedOut)
    groups: list[ScalingGroupOut] = Field(default_factory=list)
    #: Families that appeared at only one width. Not an error — most configs are only
    #: ever run one way — but a reader looking for a config that is missing from the
    #: chart deserves to be told why rather than assuming it was never measured.
    single_width_families: int = 0


class DeviceSummaryOut(BaseModel):
    """One GPU's behaviour over one run."""

    gpu_index: int
    samples: int
    sm_utilization_pct: float | None = None
    sm_utilization_max: float | None = None
    memory_used_bytes: float | None = None
    power_watts: float | None = None


class BalanceMetricOut(BaseModel):
    key: str
    label: str


class RunBalanceOut(BaseModel):
    run_id: uuid.UUID
    config_name: str
    workload_name: str
    tensor_parallel_size: int
    gpu_count: int
    replicate_idx: int = 0
    finished_at: dt.datetime | None = None
    devices: list[DeviceSummaryOut] = Field(default_factory=list)
    #: Per metric, ``(max - min) / max`` across devices. Null where fewer than two
    #: devices reported, or where the busiest sat at zero.
    imbalances: dict[str, float | None] = Field(default_factory=dict)
    worst_imbalance: float | None = None
    #: A one-device run has no balance to report, which is not the same as balanced.
    is_single_device: bool = False


class DeviceBalanceGroupOut(BaseModel):
    group_id: str
    label: str
    gpu_host_id: uuid.UUID
    gpu_host_name: str
    gpu_model: str | None = None
    vllm_version: str | None = None
    bench_client_location: str
    warnings: list[str] = Field(default_factory=list)
    run_count: int = 0
    runs: list[RunBalanceOut] = Field(default_factory=list)


class DeviceBalanceOut(BaseModel):
    source: str
    run_count: int
    truncated: bool = False
    limit: int = 0
    metrics: list[BalanceMetricOut] = Field(default_factory=list)
    excluded: ExcludedOut = Field(default_factory=ExcludedOut)
    groups: list[DeviceBalanceGroupOut] = Field(default_factory=list)
    #: Runs whose telemetry was never sampled. A run with no per-device rows is not a
    #: balanced run; it is one this view cannot speak about, and saying so beats an
    #: empty chart that looks like a clean bill of health.
    runs_without_telemetry: int = 0


class DiffLineOut(BaseModel):
    kind: str
    text: str
    left_no: int | None = None
    right_no: int | None = None


class ProvenanceDifferenceOut(BaseModel):
    field: str
    label: str
    left: str | None = None
    right: str | None = None
    #: True for differences that would stop these two sharing a chart series. They do not
    #: stop a side-by-side the reader explicitly asked for — comparing vLLM versions is a
    #: supported use — but they must be stated.
    invalidating: bool = False


class ComparisonSideOut(BaseModel):
    point_id: str
    config_hash: str
    config_name: str
    config_yaml: str
    workload_name: str
    tensor_parallel_size: int
    gpu_count: int
    replicates: int
    spread_basis: str
    spread_note: str
    gpu_host_name: str
    gpu_model: str | None = None
    vllm_version: str | None = None
    bench_client_location: str
    metrics: dict[str, SpreadOut] = Field(default_factory=dict)


class MetricComparisonOut(BaseModel):
    key: str
    label: str
    unit: str
    better: str
    left: float | None = None
    right: float | None = None
    #: Relative change from left to right. Null when either side is missing or the left
    #: is zero.
    change: float | None = None
    #: Null for "no change" as well as "unmeasurable", so a view is never forced to
    #: render an unchanged metric as a win.
    is_improvement: bool | None = None


class ComparisonOut(BaseModel):
    source: str
    left: ComparisonSideOut
    right: ComparisonSideOut
    config_diff: list[DiffLineOut] = Field(default_factory=list)
    configs_identical: bool = False
    provenance_differences: list[ProvenanceDifferenceOut] = Field(default_factory=list)
    metrics: list[MetricComparisonOut] = Field(default_factory=list)
