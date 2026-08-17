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

    # How often to sample /metrics and NVML during a benchmark. The agent runs on the
    # machine under test, so this is a knob on how much the measurement is disturbed:
    # 1s costs one loopback GET and one NVML read per device per second, against an
    # engine generating thousands of tokens in the same window. Raise it on a host where
    # even that is too much; a request can override it per run.
    telemetry_interval_seconds: float = 1.0

    # Absolute path to the `vllm` executable, for the case where the agent is installed
    # somewhere other than the vLLM environment.
    #
    # Normally leave this unset and install the agent into the vLLM venv — every one of
    # the agent's dependencies is already there (vLLM's server is itself a FastAPI and
    # uvicorn app using pydantic, psutil and NVML), so it adds two pure-Python packages
    # and pulls in nothing new.
    #
    # When isolation is genuinely required, set this rather than manipulating PATH. PATH
    # is invisible in `ps`, silently lost across systemd units, tmux sessions and
    # reboots, and when wrong it produces a confusing null version rather than an error.
    vllm_bin: str = ""
