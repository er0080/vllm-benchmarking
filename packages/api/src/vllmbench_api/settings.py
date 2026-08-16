"""Control-plane configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLLMBENCH_", extra="ignore")

    # Shared secret for every agent this control plane talks to. Per-host tokens are a
    # post-1.0 concern; one GPU host is the supported topology.
    token: str = ""

    mcp_enabled: bool = False
    mcp_write_enabled: bool = True
