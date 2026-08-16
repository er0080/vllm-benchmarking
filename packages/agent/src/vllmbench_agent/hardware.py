"""GPU and vLLM facts, probed from the host the agent runs on.

Every function here degrades rather than raises. The agent must start and answer
``/health`` on a machine with no NVIDIA driver — that is how a developer runs it on a
laptop — and an agent that refuses to boot because NVML is missing gives the operator no
way to see *why*. Absence is reported as data.
"""

from __future__ import annotations

import functools
import logging
import shutil
import subprocess

from vllmbench_protocol.wire import GpuInfo

log = logging.getLogger(__name__)

_PROBE_TIMEOUT_SECONDS = 15


def _decode(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def probe_gpus() -> list[GpuInfo]:
    """Enumerate NVIDIA devices via NVML.

    Returns an empty list when no driver is present. An empty list is a legitimate
    answer that the control plane can surface ("this host has no GPUs") rather than an
    error it has to interpret.
    """
    try:
        import pynvml
    except ImportError:
        log.info("pynvml not importable; reporting no GPUs")
        return []

    try:
        pynvml.nvmlInit()
    except Exception as exc:
        log.info("NVML unavailable (%s); reporting no GPUs", exc)
        return []

    gpus: list[GpuInfo] = []
    try:
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = _decode(pynvml.nvmlDeviceGetName(handle)) or f"device-{index}"

            uuid: str | None = None
            try:
                uuid = _decode(pynvml.nvmlDeviceGetUUID(handle))
            except Exception:  # noqa: S110 - optional detail
                pass

            vram: int | None = None
            try:
                vram = int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)
            except Exception:  # noqa: S110 - optional detail
                pass

            gpus.append(GpuInfo(index=index, name=name, uuid=uuid, vram_bytes=vram))
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: S110
            pass

    return gpus


def probe_driver_version() -> str | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            return _decode(pynvml.nvmlSystemGetDriverVersion())
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def probe_cuda_version() -> str | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            raw = pynvml.nvmlSystemGetCudaDriverVersion()
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None
    # NVML packs the version as major * 1000 + minor * 10.
    return f"{raw // 1000}.{(raw % 1000) // 10}"


@functools.cache
def probe_vllm_version() -> str | None:
    """Report the vLLM version in the agent's own environment.

    Read by importing rather than shelling out where possible: the agent is installed
    *into* the vLLM venv, so the import is the authoritative answer for the interpreter
    that will actually launch the server. The subprocess fallback covers an agent
    installed alongside rather than inside.

    Cached because this is asked on every handshake and the answer cannot change without
    restarting the process.
    """
    try:
        # Unresolvable when type-checking, and that is correct: vLLM is never a
        # dependency of this workspace. It exists only in the environment on the GPU
        # host that the agent is installed into (invariants 1 and 3).
        import vllm  # pyright: ignore[reportMissingImports]

        version = getattr(vllm, "__version__", None)
        if version:
            return str(version)
    except ImportError:
        pass

    executable = shutil.which("vllm")
    if executable is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("could not run `vllm --version`: %s", exc)
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
