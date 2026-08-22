"""Wire types and protocol constants shared by the control plane and the GPU-host agent.

This package is the one place both sides of the host boundary agree on. It must stay
dependency-light: the agent installs it into a user's vLLM environment, so anything added
here lands on the system under test.
"""

from __future__ import annotations

from vllmbench_protocol.client import AgentClient
from vllmbench_protocol.errors import (
    AgentAuthError,
    AgentError,
    AgentUnreachable,
    ProtocolMismatch,
)
from vllmbench_protocol.failures import (
    TRANSIENT_KINDS,
    FailureKind,
    classify_agent_error,
    classify_engine_output,
)
from vllmbench_protocol.placeholders import is_placeholder, warn_about_placeholders
from vllmbench_protocol.version import PROTOCOL_VERSION, __version__
from vllmbench_protocol.wire import (
    AUTH_HEADER,
    AUTH_SCHEME,
    EnvironmentCheck,
    EnvironmentStatus,
    GpuInfo,
    HealthResponse,
    HostInfo,
)

__all__ = [
    "AUTH_HEADER",
    "AUTH_SCHEME",
    "PROTOCOL_VERSION",
    "TRANSIENT_KINDS",
    "AgentAuthError",
    "AgentClient",
    "AgentError",
    "AgentUnreachable",
    "EnvironmentCheck",
    "EnvironmentStatus",
    "FailureKind",
    "GpuInfo",
    "HealthResponse",
    "HostInfo",
    "ProtocolMismatch",
    "__version__",
    "classify_agent_error",
    "classify_engine_output",
    "is_placeholder",
    "warn_about_placeholders",
]
