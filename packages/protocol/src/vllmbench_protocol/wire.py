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

    gpus: list[GpuInfo] = Field(default_factory=list)

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

    timeout_seconds: float = Field(default=3600.0, gt=0)
    # Upstream resets caches between runs; carrying a warm prefix cache across sweep
    # points silently invalidates the comparison.
    reset_caches_first: bool = True


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
