"""Wire types and protocol constants shared by the control plane and the GPU-host agent.

This package is the one place both sides of the host boundary agree on. It must stay
dependency-light: the agent installs it into a user's vLLM environment, so anything added
here lands on the system under test.
"""

from __future__ import annotations

from vllmbench_protocol.version import PROTOCOL_VERSION, __version__

__all__ = ["PROTOCOL_VERSION", "__version__"]
