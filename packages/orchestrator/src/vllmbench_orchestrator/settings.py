"""Orchestrator configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class OrchestratorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLLMBENCH_", extra="ignore")

    # Shared with the API. Empty is allowed so the service starts and reports the
    # problem, rather than crash-looping before anyone can read a log line.
    token: str = ""

    # Delete telemetry for runs that finished more than this many days ago. 0 keeps
    # everything, and is the default.
    #
    # Off by default deliberately. This is a benchmarking tool: telemetry is what turns
    # "which config won" into "why it won", and a framework that silently discards that
    # after ninety days because ninety sounded reasonable would be making a decision that
    # belongs to whoever owns the results. It is also not urgent — measured growth is
    # about 7 MB per run-hour on eight GPUs, so the first year of a busy host is tens of
    # gigabytes. Bounded is the point; choosing the bound is the operator's.
    telemetry_retention_days: int = 0
