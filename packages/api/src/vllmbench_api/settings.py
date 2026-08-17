"""Control-plane configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLLMBENCH_", extra="ignore")

    # Shared secret for every agent this control plane talks to. Per-host tokens are a
    # post-1.0 concern; one GPU host is the supported topology.
    token: str = ""

    # The MCP surface is off unless asked for. ADR 0001 mounts it on this service rather
    # than a separate one, and the flag is what keeps that from meaning "always exposed".
    mcp_enabled: bool = False
    mcp_write_enabled: bool = True

    # Bearer token clients present to /mcp. Separate from `token`, which is what this
    # service presents to *agents*: one is an inbound credential and the other outbound,
    # and sharing a value would mean an MCP client could impersonate the control plane to
    # a GPU host.
    mcp_token: str = ""

    # Host headers /mcp will accept. Empty turns DNS-rebinding protection off, which is
    # the default because the alternative is a surface that is broken on arrival: the
    # control plane is reached by LAN address and there is no host value we could guess.
    # Rebinding protection defends a *browser* that can be tricked into calling a local
    # service; the bearer token is what actually guards this one, and a rebinding attacker
    # does not have it. Set this to a real list to turn the check on.
    mcp_allowed_hosts: list[str] = []
