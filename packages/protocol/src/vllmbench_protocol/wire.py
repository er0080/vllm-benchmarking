"""Wire types for the control plane ↔ agent boundary.

These models are the contract. Both sides import them from here rather than each
declaring their own, because two hand-maintained copies of a schema drift, and the drift
shows up as a field silently arriving as None rather than as an error.
"""

from __future__ import annotations

import enum
from typing import Any

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


class EnvironmentStatus(enum.StrEnum):
    """Three states, because "no conflicts" and "nobody looked" are not the same claim.

    An agent too old to report this leaves the field unset, which reads as NOT_REPORTED
    downstream. Collapsing that into OK would let a silent absence pass for a clean bill
    of health — the failure this whole check exists to prevent, reintroduced one layer up.
    """

    OK = "ok"
    CONFLICTS = "conflicts"
    #: The check could not run — malformed metadata, an unreadable environment.
    UNAVAILABLE = "unavailable"
    #: Never sent. Recorded by the control plane when an agent said nothing.
    NOT_REPORTED = "not_reported"


class PeerAccessStatus(enum.StrEnum):
    """Whether the devices a run used could reach each other's memory directly.

    Peer-to-peer DMA is the difference between an all-reduce that crosses the link once
    and one that stages through host memory, and on consumer hardware whether it is
    available is a property of the *driver build* rather than of the driver version — a
    patched module reports the version it was patched from. So two runs can agree on
    every provenance field a run has ever carried and still have been measured over
    different interconnects. This is what records that they were.

    Five states, for the same reason :class:`EnvironmentStatus` has four. The one this
    adds is SINGLE_DEVICE: a run that used one GPU has no peer access to report, and
    calling that "unsupported" would put a TP=1 run on one side of a boundary it cannot
    be on either side of — which matters most precisely when a single-device run is being
    used as the control for a change to the interconnect.
    """

    #: Every pair of devices this run used can reach every other, both directions.
    OK = "ok"
    #: At least one pair cannot. Partial peer access is not a weaker "ok": the engine
    #: falls back for the whole group, so one broken pair changes what everything measures.
    UNSUPPORTED = "unsupported"
    #: Fewer than two devices. Nothing to report, and nothing that could differ.
    SINGLE_DEVICE = "single_device"
    #: NVML was there but would not answer. Distinct from UNSUPPORTED, which is an answer.
    UNAVAILABLE = "unavailable"
    #: Never sent. Recorded by the control plane when an agent said nothing — every run
    #: measured before protocol 8, and any run by an agent that could not look.
    NOT_REPORTED = "not_reported"


class EnvironmentCheck(_Wire):
    """Whether the agent's Python environment satisfies its own declared constraints.

    The agent lives in vLLM's virtualenv, so its resolution and vLLM's ceilings meet with
    nothing arbitrating between them. This reports the outcome and never blocks on it —
    the same treatment the vLLM version policy gives a version mismatch, and for the same
    reason: a measurement taken on an inconsistent environment is not necessarily wrong,
    it is unattributable, which is a thing to record.
    """

    status: EnvironmentStatus
    #: Human-readable lines, each naming the requirer, the requirement and what is
    #: installed. Bounded by the agent; a wholly broken environment does not get to send
    #: pages of them.
    conflicts: list[str] = Field(default_factory=list)
    #: How many distributions were examined, so an empty result can be told apart from an
    #: environment the check could not see into.
    distributions: int | None = Field(default=None, ge=0)
    detail: str | None = None


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
    # Why vllm_version is null, when it is. "null" alone tells an operator nothing —
    # missing, not on PATH, or slow to import are three different problems with three
    # different fixes, and distinguishing them cost real time on the first real host.
    vllm_probe_detail: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None

    # Host-wide peer-access state, across every pair of devices. The value a run carries
    # is narrower — scoped to the devices that run actually used — and travels on
    # BenchResponse. This one is what an operator looks at when deciding whether the host
    # is set up the way they think it is.
    peer_access: PeerAccessStatus | None = None
    #: The pairwise detail behind that summary, one line per pair that is not OK. Kept on
    #: the host rather than on every run for the reason `environment_conflicts` is: it is
    #: long, identical across a sweep, and only useful while somebody is fixing the host.
    peer_access_detail: list[str] = Field(default_factory=list)

    gpus: list[GpuInfo] = Field(default_factory=list)

    # Whether the agent's own virtualenv is internally consistent. Protocol 6. Optional
    # on the model rather than required so that the *shape* survives an agent that could
    # not run the check — the control plane records the absence as NOT_REPORTED, which is
    # a different claim from "no conflicts" and must stay one.
    environment: EnvironmentCheck | None = None

    # Set by the producer, never inferred by the consumer (invariant 7). A mock or a
    # CPU-backend agent names itself here, and the control plane marks every run it
    # produces accordingly. Inference downstream would eventually mark a synthetic run
    # as real, and the failure is silent.
    synthetic_source: str | None = None

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------


class ServerState(enum.StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    STOPPING = "stopping"


class StartServerRequest(_Wire):
    """Launch a vLLM server from a configuration.

    ``config_yaml`` is passed through to ``vllm serve --config`` verbatim (invariant 5).
    The agent does not parse it beyond what it must write to disk, so a config using
    arguments this build has never heard of still works.
    """

    config_yaml: str = Field(min_length=1)
    # Identifies the config for logging and for matching an already-running server.
    # The control plane computes it; the agent only echoes it back.
    config_hash: str = Field(min_length=8, max_length=64)
    port: int = Field(default=8000, ge=1024, le=65535)
    # Model load can take minutes for a large model on a cold page cache. The default is
    # generous because the failure mode of being too eager — declaring a healthy server
    # dead — wastes far more time than waiting.
    readiness_timeout_seconds: float = Field(default=900.0, gt=0)


class ServerStatus(_Wire):
    state: ServerState
    config_hash: str | None = None
    pid: int | None = None
    port: int | None = None
    started_at: float | None = None
    ready_at: float | None = None
    error: str | None = None
    log_tail: list[str] = Field(default_factory=list)

    # Read from the engine once it is up, so a run records what actually served it.
    # The engine's own /version beats probing the agent's environment: the agent may
    # live in a different venv, and it is the engine that produced the numbers.
    vllm_version: str | None = None

    # What the engine actually serves, read from its own /v1/models. This is NOT the
    # `model:` line in the config: a config setting `served-model-name` serves under
    # that alias instead, and benchmarking the HF id would simply 404. The engine is the
    # only thing that knows which name its API answers to.
    served_model_name: str | None = None

    # Attributed by NVML from the server's process tree — the devices the engine is
    # genuinely occupying, not the ones the config asked for. Per-GPU normalization
    # divides by this count, so a config requesting more devices than the host can give
    # would otherwise silently corrupt every comparison the run appears in.
    device_indices: list[int] | None = None

    # What the config *asked* for. Kept alongside the observed devices rather than
    # instead of them, so a disagreement between request and reality stays visible.
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None

    # What the engine resolved for speculative decoding, read from its own /server_info
    # rather than from the config text. Invariant 8 says parallelism topology is never
    # inferred from the YAML after the fact; speculation is the same kind of fact and
    # gets the same treatment. The YAML can say `num_speculative_tokens: 3` while the
    # engine runs without a drafter, and only the engine can say which happened.
    #
    # None on both means "the engine did not say" — a version without /server_info, or a
    # reachability failure. NOT the same as `speculative_method="none"`, which is the
    # engine stating it is not speculating. Silence and a clean answer are different
    # facts, for the same reason EnvironmentStatus.NOT_REPORTED is not OK.
    speculative_method: str | None = None
    speculative_tokens: int | None = None


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


class BenchRequest(_Wire):
    """Arguments for one ``vllm bench serve`` invocation.

    Deliberately close to the CLI rather than an abstraction over it: invariant 4 says we
    orchestrate the upstream benchmark, we do not reimplement it, and a translation layer
    here would be one more thing to drift.
    """

    # Two names, because vLLM uses them for two different things and conflating them
    # corrupts results rather than failing.
    #
    #   model              -> `--model`, which is the *weights* identifier. vLLM loads
    #                         the tokenizer from it, so it must be a real HF repo id or
    #                         local path.
    #   served_model_name  -> `--served-model-name`, the alias the API answers to, which
    #                         is what goes in the request body.
    #
    # A config with `served-model-name: Qwen3.8-27B` over `model: Qwen/Qwen3.8-27B-FP8`
    # serves under the alias, so requesting `model:` 404s. Sending the alias as `--model`
    # instead fixes the 404 and breaks tokenization: if the alias is not a valid repo id
    # the benchmark dies, and if it happens to be one, it tokenizes with somebody else's
    # tokenizer and reports confident, wrong input-token counts.
    model: str = Field(min_length=1)
    # None means "same as model", which is vLLM's own default for the flag.
    served_model_name: str | None = None

    dataset_name: str = "random"
    dataset_path: str | None = None
    hf_name: str | None = None

    num_prompts: int = Field(default=200, ge=1)
    # None means unbounded — `--request-rate inf` and no `--max-concurrency`. None rather
    # than a sentinel because "no limit" is the absence of a bound, and a magic number
    # eventually gets averaged into a chart.
    request_rate: float | None = None
    max_concurrency: int | None = None
    burstiness: float | None = None

    random_input_len: int | None = None
    random_output_len: int | None = None

    # Passed through verbatim, for flags this build predates.
    extra_args: list[str] = Field(default_factory=list)

    # None means "use the agent's configured default". Per-run because the acceptable
    # sampling cost is a property of the host, but a particular run may want finer
    # resolution (a short saturation test) or coarser (a long soak).
    telemetry_interval_seconds: float | None = Field(default=None, gt=0)

    timeout_seconds: float = Field(default=3600.0, gt=0)
    # Upstream resets caches between runs; carrying a warm prefix cache across sweep
    # points silently invalidates the comparison.
    reset_caches_first: bool = True


class EngineSampleWire(_Wire):
    """One scrape of vLLM's ``/metrics``.

    Every field is optional because a scrape that partly fails must still record what it
    did read. The alternative — dropping the whole sample — leaves a gap in the timeline
    exactly when the engine is under the most stress, which is when the timeline matters.
    """

    # Seconds since the benchmark started, not a wall clock. The control plane converts
    # to absolute time using the run's own start, so a clock skewed between the two hosts
    # cannot slide the telemetry out of alignment with the window it describes.
    offset_seconds: float

    num_requests_running: int | None = None
    num_requests_waiting: int | None = None
    # 0..1, as vLLM emits it. See metrics.py — the upstream name says "perc" and lies.
    kv_cache_usage_fraction: float | None = None

    # Counters, raw. Rates are derived later so any window can be differenced.
    num_preemptions_total: int | None = None
    prefix_cache_queries_total: int | None = None
    prefix_cache_hits_total: int | None = None


class GpuSampleWire(_Wire):
    """One NVML read of one device.

    Per device, never averaged. A host-level mean is the one summary guaranteed to hide
    the imbalance that makes a tensor-parallel run diagnosable.
    """

    offset_seconds: float
    gpu_index: int

    sm_utilization_pct: float | None = None
    memory_used_bytes: int | None = None
    power_watts: float | None = None
    temperature_c: float | None = None
    sm_clock_mhz: int | None = None
    memory_clock_mhz: int | None = None


class CancelResponse(_Wire):
    """The outcome of asking the agent to stop what it is doing."""

    # False means there was nothing running. Not an error: cancelling a sweep races the
    # run finishing on its own, and both orders have to be safe.
    cancelled: bool
    detail: str = ""


class BenchResponse(_Wire):
    """The result of one benchmark, plus what it took to get it."""

    # Verbatim --save-result payload. Stored raw so a wrong flattening can be recomputed
    # rather than re-measured.
    raw_result: dict[str, Any]
    duration_seconds: float
    stdout_tail: list[str] = Field(default_factory=list)
    stderr_tail: list[str] = Field(default_factory=list)

    # Echoed from the server that served it, so the run records the topology that ran.
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    device_indices: list[int] | None = None

    # Echoed from the same place, for the same reason. A run has to be able to say
    # whether it was speculating without anyone reading its config text.
    speculative_method: str | None = None
    speculative_tokens: int | None = None

    # Observed over `device_indices` above — the devices this benchmark actually ran on,
    # not the ones the host happens to have. A single-device run reports SINGLE_DEVICE
    # even on a host where every other pair is fine, so that a TP=1 control stays one
    # series across a change to the interconnect. Protocol 8.
    peer_access: PeerAccessStatus | None = None

    # What the benchmark actually read, computed on the GPU host because that is the
    # only host that can see it (invariant 1). See `vllmbench_agent.dataset` for the
    # forms this takes; the control plane stores the string and never parses it.
    dataset_identity: str | None = None

    # Telemetry sampled across the benchmark window, returned with the result rather
    # than streamed. One round trip, nothing to reconcile, and no partial series left
    # behind if the run fails — the samples arrive with the thing they describe or not
    # at all.
    engine_samples: list[EngineSampleWire] = Field(default_factory=list)
    gpu_samples: list[GpuSampleWire] = Field(default_factory=list)
    # True when sampling was decimated to stay within its cap. The series still spans the
    # whole window, at lower resolution — but a consumer computing a rate from adjacent
    # samples needs to know the spacing changed.
    telemetry_decimated: bool = False
    telemetry_interval_seconds: float | None = None
