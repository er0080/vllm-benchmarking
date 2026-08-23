"""Request and response models for the control-plane API.

Separate from ``vllmbench_protocol.wire``, which is the agent contract. Conflating them
would mean a change to what the UI displays could alter what the agent must send.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vllmbench_db.enums import InitiatedBy, ReplicateOrder
from vllmbench_protocol import EnvironmentStatus


class GpuDeviceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    device_index: int
    name: str
    vram_bytes: int | None = None


class HostCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    agent_url: str = Field(min_length=1)

    # Operational limits, not measurement parameters. The defaults match what the agent
    # has always enforced; they are settable so that a host loading a 70B, or a workload
    # sending fifty thousand prompts, does not require a code change to finish.
    model_load_timeout_seconds: int = Field(default=900, ge=30, le=86_400)
    benchmark_timeout_seconds: int = Field(default=3600, ge=30, le=86_400)


def _reported_status(value: str | None) -> str:
    """A stored NULL is an agent that could not say, which is its own answer.

    Every row written before protocol 6 has NULL here, and so would any row written from
    an agent that failed to report. Letting that surface as null puts the decision on
    every consumer, and the tempting default — treat missing as fine — is the one that
    turns "nobody checked" into "checked and clean".
    """
    return value or EnvironmentStatus.NOT_REPORTED.value


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
    # Never null on the wire: a stored NULL means an agent that could not say, which is
    # "not_reported" and not "ok". Rounding one to the other would turn silence into a
    # clean bill of health.
    environment_status: str = EnvironmentStatus.NOT_REPORTED.value
    environment_conflicts: list[str] = Field(default_factory=list)
    model_load_timeout_seconds: int = 900
    benchmark_timeout_seconds: int = 3600
    last_seen_at: dt.datetime | None = None
    created_at: dt.datetime

    @field_validator("environment_status", mode="before")
    @classmethod
    def _status_reported(cls, value: str | None) -> str:
        return _reported_status(value)

    @field_validator("environment_conflicts", mode="before")
    @classmethod
    def _conflicts_listed(cls, value: list[str] | None) -> list[str]:
        return value or []

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
    #: The config this one was edited from, when it was. Recorded only when this call
    #: creates a new row: submitting text that already exists returns the existing config
    #: and leaves its lineage alone, because that row's history already happened.
    parent_id: uuid.UUID | None = None


class ConfigAnnotate(BaseModel):
    """Metadata about a config, never its text.

    Nothing here can change what `vllm serve --config` receives. The YAML is the
    identity — editing it produces a different config with a different hash, which is a
    creation rather than an update — so these are the only fields a config can have
    changed after the fact.
    """

    name: str | None = Field(default=None, min_length=1, max_length=128)
    notes: str | None = None
    #: The run somebody would point at to defend this configuration. It must be a run
    #: that actually used it — a run of some other config is not evidence for this one.
    justified_by_run_id: uuid.UUID | None = None
    #: Why that run settles it, in the author's words.
    justification_note: str | None = None
    #: Distinguishes "leave the justification alone" from "withdraw it", which an omitted
    #: nullable field cannot express on its own.
    clear_justification: bool = False


class ConfigOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    config_hash: str
    name: str
    yaml: str
    notes: str | None = None
    parent_id: uuid.UUID | None = None
    #: Set when a run has been recorded as the evidence for this configuration. Populated
    #: from `config_justification`, which is a table rather than a column here to keep
    #: `run` and `server_config` from referencing each other in a cycle.
    justified_by_run_id: uuid.UUID | None = None
    justification_note: str | None = None
    created_at: dt.datetime


class LineageNode(BaseModel):
    """One config in a derivation chain, without its YAML.

    The text is omitted deliberately: a lineage view shows how a configuration came to
    be, and carrying every ancestor's full YAML would make the common case — a chain of
    five near-identical files — mostly duplication.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    config_hash: str
    name: str
    created_at: dt.datetime


class LineageOut(BaseModel):
    """Where a configuration came from and what came from it."""

    config_hash: str
    #: Nearest parent first, back to the original. Empty for a config nobody derived.
    ancestors: list[LineageNode] = Field(default_factory=list)
    #: Configs derived directly from this one, newest first.
    children: list[LineageNode] = Field(default_factory=list)
    #: True when the chain was cut short by a cycle. Cannot happen through the API — a
    #: parent always predates its child — but the column is a plain self-reference and a
    #: traversal that could hang on bad data is not worth shipping.
    truncated: bool = False


#: Flags the framework sets itself. Passing one through ``extra_args`` would win, because
#: argparse takes the last occurrence — and the damage is not symmetric. Overriding
#: ``--result-filename`` sends the benchmark's output somewhere nothing reads, producing a
#: summary row of NULLs that is indistinguishable from a run which legitimately measured
#: nothing. Refused at the boundary rather than trusted to care.
FRAMEWORK_OWNED_FLAGS = frozenset(
    {
        "--backend",
        "--base-url",
        "--model",
        "--served-model-name",
        "--dataset-name",
        "--dataset-path",
        "--hf-name",
        "--num-prompts",
        "--request-rate",
        "--max-concurrency",
        "--burstiness",
        "--save-result",
        "--result-filename",
    }
)

#: An escape hatch, not a scripting language. Bounded so a workload cannot carry a command
#: line longer than the thing it is describing.
MAX_EXTRA_ARGS = 32
MAX_EXTRA_ARG_LENGTH = 256


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
    #: Appended verbatim to `vllm bench serve`. Part of the workload's identity: two
    #: workloads that pass different flags send different traffic and are not the same
    #: workload, whatever else they share.
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("extra_args", mode="before")
    @classmethod
    def _tolerate_older_rows(cls, value: object) -> object:
        """Rows written before this was a list hold ``{}``, including the CI seed.

        WorkloadOut inherits this model, so reading one of those rows would otherwise fail
        validation on a value that means exactly what an empty list means.
        """
        return [] if value in (None, {}) else value

    @field_validator("extra_args")
    @classmethod
    def _no_framework_flags(cls, value: list[str]) -> list[str]:
        if len(value) > MAX_EXTRA_ARGS:
            raise ValueError(f"extra_args takes at most {MAX_EXTRA_ARGS} items, got {len(value)}")
        for item in value:
            if len(item) > MAX_EXTRA_ARG_LENGTH:
                raise ValueError(f"extra_args item longer than {MAX_EXTRA_ARG_LENGTH} characters")
            # `--flag=value` too, which argparse accepts and which would otherwise slip past.
            flag = item.split("=", 1)[0]
            if flag in FRAMEWORK_OWNED_FLAGS:
                raise ValueError(
                    f"{flag} is set by this framework; passing it in extra_args would "
                    "override what the run records about itself"
                )
        return value


class WorkloadOut(WorkloadCreate):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    workload_hash: str
    created_at: dt.datetime


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


class InitiatedByFields(BaseModel):
    """Who asked for this work.

    Required provenance under invariant 6, and it only became answerable-and-wrong once a
    second interface could create work: every sweep and run was recorded as ``ui``,
    including the ones an MCP client will now create.

    Declared by the caller rather than sniffed from the request, because HTTP cannot tell
    a browser from a curl invocation and guessing would put a confident wrong answer in a
    provenance column. This is the same trust model the project already uses for
    ``synthetic_source``: the producer declares, and nothing downstream infers.

    The default is ``api`` — the honest answer for an unidentified HTTP caller. The web
    app sends ``ui``; the MCP tools send ``mcp`` and their client's name.
    """

    initiated_by: InitiatedBy = InitiatedBy.API
    initiated_by_client: str | None = Field(default=None, max_length=128)


class RunCreate(InitiatedByFields):
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
    # Whether the host's environment was internally consistent when this ran. See HostOut
    # for why the absent case is a value rather than a null.
    environment_status: str = EnvironmentStatus.NOT_REPORTED.value

    @field_validator("environment_status", mode="before")
    @classmethod
    def _status_reported(cls, value: str | None) -> str:
        return _reported_status(value)

    gpu_count: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    device_indices: list[int] | None = None

    bench_client_location: str
    is_synthetic: bool
    synthetic_source: str | None = None
    initiated_by: str

    error: str | None = None
    # Which class of failure this was. Never a substitute for `error`, which holds the
    # full text — this is what makes "why did nine of these eleven points fail" a
    # question with an answer.
    failure_kind: str | None = None
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

    #: When this run's telemetry was deleted by the retention policy, if it was.
    #:
    #: Without it an empty timeline has two readings — the policy removed it, or sampling
    #: silently failed — and those call for opposite responses. A chart that cannot tell
    #: them apart eventually has someone debugging a sampler that works fine.
    pruned_at: dt.datetime | None = None
    pruned_horizon_days: int | None = None
    # The devices that actually produced samples, so a client can build one series per
    # device without scanning the whole payload first. Counted over the whole series,
    # never over what a downsampled response happens to carry: a device that exists is
    # listed even if the thinning were ever to drop its last sample.
    gpu_indices: list[int]
    # Present so a chart can say "no telemetry" rather than drawing an empty axis and
    # leaving the reader to wonder whether the engine was idle. This is what was
    # *recorded*, not what was returned — compare it against the array lengths to know
    # whether you are looking at all of it.
    sample_count: int
    # 1 means the response is the complete series. Anything higher means every nth
    # sample, per device, and is stated rather than inferred because a thinned series
    # and a sparsely-sampled one look identical once they arrive.
    stride: int = 1


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


class SweepCreate(InitiatedByFields):
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

    # None means "use the host's defaults" — and keeps meaning that, so raising a host's
    # limit also raises every sweep that never had an opinion about it.
    model_load_timeout_seconds: int | None = Field(default=None, ge=30, le=86_400)
    benchmark_timeout_seconds: int | None = Field(default=None, ge=30, le=86_400)


class SweepProgress(BaseModel):
    """Counts by run status, so a caller does not have to fetch every run to draw a bar."""

    total: int = 0
    queued: int = 0
    starting: int = 0
    benchmarking: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0

    # Failed runs by kind, highest count first. The whole reason `failure_kind` exists:
    # a sweep reporting "11 failed" tells a reader to open eleven runs, and one reporting
    # "9 engine_out_of_memory, 2 benchmark_timeout" tells them what to change.
    #
    # Empty for a sweep with no failures, and for one whose failures predate the column.
    failures: dict[str, int] = Field(default_factory=dict)

    @property
    def terminal(self) -> int:
        return self.succeeded + self.failed + self.cancelled


class ConfigValidationRequest(BaseModel):
    """A candidate configuration, and optionally the machine it is meant for."""

    yaml: str
    #: When given, the checks that need to know the hardware are enabled and the host's
    #: own vLLM version selects the argument catalogue. Without it the config is checked
    #: in the abstract against the reference version, which is still worth doing and is
    #: reported as such.
    gpu_host_id: uuid.UUID | None = None
    #: Whether a sweep will be rewriting `tensor-parallel-size` on this config, which
    #: turns whatever is written there into something the engine never sees.
    tensor_parallel_is_swept: bool = False


class FindingOut(BaseModel):
    #: "error" means vLLM will refuse to start. "warning" means it will start and may not
    #: do what was meant — a distinction worth keeping, since the second kind is the one
    #: that quietly produces a valid-looking result.
    severity: str
    message: str
    key: str | None = None
    line: int | None = None
    #: Offered, never applied. Invariant 5 says validate, do not transform.
    suggestion: str | None = None


class ConfigValidationOut(BaseModel):
    valid: bool
    findings: list[FindingOut] = Field(default_factory=list)
    #: The vLLM version whose arguments were used. Stated because validating against a
    #: different version than the target runs is normal here, and changes what a clean
    #: result means.
    checked_against: str
    exact_version_match: bool = True


class McpWriteAuditOut(BaseModel):
    """One recorded write call from the MCP surface."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    called_at: dt.datetime
    tool: str
    client: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    #: "succeeded", "refused" or "failed". Refused is a policy decision the surface made
    #: and understood; failed is a bug. Reading the log back, conflating them hides both.
    outcome: str
    error: str | None = None
    subject: str | None = None


class DurationEstimateOut(BaseModel):
    """How much longer a sweep has, and how much to trust that.

    Structured rather than a single number of seconds because the two components behave
    completely differently: benchmark time scales with the runs left, engine-load time
    scales with the *config changes* left, and a plan with three configs remaining costs
    minutes more than one with three runs of the same config. A caller shown only a total
    cannot tell those apart, and neither can it tell a well-founded estimate from one
    extrapolated off a single completed run.
    """

    runs_remaining: int = 0
    engine_loads_remaining: int = 0
    # Null means "not known", which is different from zero. Nothing here guesses: a
    # fabricated countdown looks measured, and this project's whole posture is that a
    # number nobody derived is worse than a number nobody has.
    seconds_remaining: float | None = None
    median_run_seconds: float | None = None
    median_engine_load_seconds: float | None = None
    basis: str = "none"
    sample_size: int = 0
    caveats: list[str] = Field(default_factory=list)


class SweepOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None = None
    status: str
    gpu_host_id: uuid.UUID
    replicates: int
    replicate_order: str
    model_load_timeout_seconds: int | None = None
    benchmark_timeout_seconds: int | None = None
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

    # Present on sweeps that still have work to do. A terminal sweep reports zero
    # remaining, which is a fact rather than an extrapolation.
    estimated_remaining: DurationEstimateOut | None = None


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


class SavedViewCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str | None = None
    view: str = Field(min_length=1, max_length=32)
    source: str = "real"
    #: The selection, not the results. A view stores a query so that reopening it next
    #: month includes the runs measured since — pinning run ids would produce something
    #: that silently stops tracking reality while still looking current.
    filters: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class SavedViewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None = None
    view: str
    source: str
    filters: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    created_at: dt.datetime


# ---------------------------------------------------------------------------
# Storage and retention
# ---------------------------------------------------------------------------


class TableUsageOut(BaseModel):
    name: str
    rows: int
    total_bytes: int
    #: Whether retention may ever delete from this table, and the reason it may not.
    #: Surfaced rather than kept internal: an operator looking at a large table needs to
    #: know immediately whether it is reclaimable or whether it is the results.
    protected: bool
    protected_because: str | None = None


class StorageOut(BaseModel):
    total_bytes: int
    tables: list[TableUsageOut]

    engine_samples: int
    gpu_samples: int
    oldest_sample_at: dt.datetime | None = None

    #: Telemetry bytes per hour of benchmarking, measured from this database's own
    #: history. Null until there is something to measure — a made-up number here would be
    #: worse than silence, since it is what someone would size a disk against.
    bytes_per_run_hour: int | None = None

    runs_with_telemetry: int = 0
    runs_pruned: int = 0


class PruneOut(BaseModel):
    cutoff: dt.datetime
    runs: int
    engine_samples: int
    gpu_samples: int
    bytes_reclaimed: int
    #: True unless the caller passed `confirm=true`. The default answer to "what would
    #: this remove" is a number, not a removal.
    dry_run: bool
