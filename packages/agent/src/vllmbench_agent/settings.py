"""Agent configuration."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLLMBENCH_", extra="ignore")

    # No default. An agent that silently starts with a guessable token is worse than one
    # that refuses to start, because the failure is invisible until someone else finds it.
    token: str = Field(min_length=8)

    host: str = "0.0.0.0"  # noqa: S104 - the control plane is on another machine
    port: int = 9110

    log_level: str = "INFO"
