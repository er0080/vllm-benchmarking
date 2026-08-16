"""Failures at the control plane ↔ agent boundary.

Distinct types rather than one generic error, because the operator response differs for
each: unreachable means check the network, auth means check the token, protocol mismatch
means upgrade one side.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base for anything that went wrong talking to an agent."""


class AgentUnreachable(AgentError):
    def __init__(self, url: str, detail: str) -> None:
        super().__init__(f"agent at {url} is unreachable: {detail}")
        self.url = url
        self.detail = detail


class AgentAuthError(AgentError):
    def __init__(self, url: str) -> None:
        super().__init__(
            f"agent at {url} rejected the token; check VLLMBENCH_TOKEN matches on both sides"
        )
        self.url = url


class ProtocolMismatch(AgentError):
    """The agent speaks a different version of the agent API.

    This is fatal on purpose. The alternative — proceeding and hoping the differences do
    not matter — risks a sweep that runs to completion and writes results that are subtly
    wrong, which is far more expensive than refusing to start. CLAUDE.md: a stale agent
    must fail loudly at connect time.

    Contrast a *vLLM* version mismatch, which only warns: comparing vLLM versions is a
    supported use of this tool, so blocking on it would prevent a feature.
    """

    def __init__(self, url: str, agent_version: int, expected: int) -> None:
        super().__init__(
            f"agent at {url} speaks protocol {agent_version}, this control plane speaks "
            f"{expected}; upgrade whichever is older — both ship as one version"
        )
        self.url = url
        self.agent_protocol_version = agent_version
        self.expected_protocol_version = expected
