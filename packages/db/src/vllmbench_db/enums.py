"""Enumerations stored as native PostgreSQL enum types."""

from __future__ import annotations

import enum


class RunStatus(enum.StrEnum):
    QUEUED = "queued"
    STARTING = "starting"
    BENCHMARKING = "benchmarking"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Terminal runs are immutable. Enforced by a database trigger, not by trust."""
        return self in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}


class SweepStatus(enum.StrEnum):
    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InitiatedBy(enum.StrEnum):
    """Which interface asked for this work.

    Required provenance under invariant 6. Once more than one interface can start a
    sweep, "what produced this run" includes which interface asked — the question
    "why did this run at 3am" has no answer without it.
    """

    UI = "ui"
    API = "api"
    MCP = "mcp"


class BenchClientLocation(enum.StrEnum):
    """Where the benchmark client process ran.

    Invariant 2 fixes this at ``loopback``: the client runs on the GPU host so network
    RTT and jitter stay out of TTFT and ITL. The enum exists so that if a remote mode is
    ever added, existing runs can still state which they were, and the two can be kept
    apart in charts. A run whose latency figures include a network hop is not comparable
    with one whose figures do not, and nothing downstream can tell them apart after the
    fact unless it was recorded here.
    """

    LOOPBACK = "loopback"
    REMOTE = "remote"


class SyntheticSource(enum.StrEnum):
    """What fake produced this run, when one did.

    Invariant 7: set by the producer at the moment of creation, never inferred later.
    Inference would eventually mark a synthetic run as real — the failure mode is silent
    and the result is fabricated data in a chart.
    """

    MOCK_AGENT = "mock_agent"
    FAKE_VLLM = "fake_vllm"
    CPU_BACKEND = "cpu_backend"
