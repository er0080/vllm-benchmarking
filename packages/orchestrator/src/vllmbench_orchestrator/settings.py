"""Orchestrator configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class OrchestratorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLLMBENCH_", extra="ignore")

    # Shared with the API. Empty is allowed so the service starts and reports the
    # problem, rather than crash-looping before anyone can read a log line.
    token: str = ""
