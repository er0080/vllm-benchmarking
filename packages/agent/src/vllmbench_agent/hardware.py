"""GPU and vLLM facts, probed from the host the agent runs on.

Every function here degrades rather than raises. The agent must start and answer
``/health`` on a machine with no NVIDIA driver — that is how a developer runs it on a
laptop — and an agent that refuses to boot because NVML is missing gives the operator no
way to see *why*. Absence is reported as data.
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

from vllmbench_protocol.wire import GpuInfo

log = logging.getLogger(__name__)

# `vllm --version` imports the whole package — torch, CUDA init, the lot. On a busy
# host that legitimately takes far longer than a naive timeout allows. 15s was the
# original value and it produced a null version on a real dual-3090 box, which under
# invariant 6 makes every run from that host invalid.
_PROBE_TIMEOUT_SECONDS = 120


def _executable(path: Path) -> str | None:
    return str(path) if path.is_file() and os.access(path, os.X_OK) else None


def vllm_binary_search_detail(configured: str = "") -> str:
    """Say where we looked, for the error raised when we found nothing.

    "No `vllm` executable found" on a host that visibly has vLLM installed is a message
    that sends people to the wrong place — usually to PATH, which is the one answer
    README tells them not to use.
    """
    if configured:
        return f"VLLMBENCH_VLLM_BIN is set to {configured!r}, which is not an executable file"
    scripts = sysconfig.get_path("scripts") or "(unknown)"
    return (
        f"looked in this interpreter's script directory ({scripts}) and on PATH. "
        "Install the agent into the vLLM environment (it adds no new dependencies), "
        "or set VLLMBENCH_VLLM_BIN to the absolute path of the vllm executable."
    )


def resolve_vllm_binary(configured: str = "") -> str | None:
    """Locate the `vllm` executable.

    Order: the explicit setting, then this interpreter's own script directory, then PATH.

    The middle step is the one that matters, and it was missing. The documented, and
    recommended, deployment installs the agent *into* the vLLM environment, where `vllm`
    sits beside the very interpreter running this code. Looking only at PATH meant that
    arrangement worked for everything except the thing it exists for: the agent imported
    vLLM successfully, reported the correct version on `/host-info`, presented as
    entirely healthy, and then failed every single run with "no `vllm` executable found".

    Nothing about that failure points at PATH, and the fix people reach for is to put the
    venv on PATH — which README explicitly warns against, because PATH is invisible in
    `ps`, lost across systemd units and tmux sessions, and silently wrong after a reboot.
    Asking the interpreter where its own scripts live needs no environment at all.
    """
    if configured:
        return _executable(Path(configured))

    # sysconfig rather than Path(sys.executable).parent: it is correct for a venv, for a
    # system install, and for the Windows Scripts/ layout, and it does not care whether
    # sys.executable happens to be a symlink into a uv-managed interpreter.
    scripts = sysconfig.get_path("scripts")
    if scripts:
        found = _executable(Path(scripts) / "vllm")
        if found:
            return found

    return shutil.which("vllm")


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
def probe_vllm_version(configured_bin: str = "") -> tuple[str | None, str]:
    """Report the vLLM version in the agent's own environment.

    Read by importing rather than shelling out where possible: the agent is installed
    *into* the vLLM venv, so the import is the authoritative answer for the interpreter
    that will actually launch the server. The subprocess fallback covers an agent
    installed alongside rather than inside.

    Returns (version, detail). The detail explains a null version, because "null" on its
    own tells an operator nothing — is vLLM missing, not on PATH, or just slow to import?
    That question cost real time on the first real host, so the answer travels with the
    result.

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
            return str(version), "imported from the agent's own environment"
    except ImportError:
        pass

    executable = resolve_vllm_binary(configured_bin)
    if executable is None:
        return None, (
            "vLLM is not importable here and no `vllm` executable was found. "
            "Install the agent into the vLLM environment (it adds no new dependencies), "
            "or set VLLMBENCH_VLLM_BIN to the absolute path of the vllm executable."
        )

    try:
        result = subprocess.run(  # noqa: S603
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, (
            f"`{executable} --version` did not finish within {_PROBE_TIMEOUT_SECONDS}s. "
            "The engine's own /version endpoint is used for run provenance regardless."
        )
    except OSError as exc:
        return None, f"could not run `{executable} --version`: {exc}"

    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-1:] or [""]
        return None, f"`vllm --version` exited {result.returncode}: {tail[0]}"

    version = result.stdout.strip()
    if not version:
        return None, "`vllm --version` printed nothing"
    # Newer CLIs print just the number; older ones prefix it.
    return version.split()[-1], f"read from `{executable} --version`"


def devices_for_process(pid: int) -> list[int]:
    """Which GPUs a process (and its children) actually occupies, per NVML.

    Ground truth for device attribution, and better than anything derivable from the
    config: a config asking for TP=4 on a host that could only give it 2 would otherwise
    produce a run claiming four devices. Since per-GPU normalization divides by this
    number, getting it wrong silently corrupts every comparison the run takes part in.
    """
    try:
        import psutil
        import pynvml
    except ImportError:
        return []

    try:
        family = {pid}
        try:
            family |= {child.pid for child in psutil.Process(pid).children(recursive=True)}
        except Exception:  # noqa: S110 - the parent alone is still a useful answer
            pass

        pynvml.nvmlInit()
        try:
            found: list[int] = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                for accessor in (
                    "nvmlDeviceGetComputeRunningProcesses_v3",
                    "nvmlDeviceGetComputeRunningProcesses",
                ):
                    getter = getattr(pynvml, accessor, None)
                    if getter is None:
                        continue
                    try:
                        if any(p.pid in family for p in getter(handle)):
                            found.append(index)
                        break
                    except Exception as exc:
                        log.debug("%s failed on device %d: %s", accessor, index, exc)
                        continue
            return sorted(found)
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        log.info("could not attribute devices to pid %d: %s", pid, exc)
        return []
