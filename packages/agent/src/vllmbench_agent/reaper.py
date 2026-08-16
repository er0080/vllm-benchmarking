"""Durable record of the vLLM process this agent owns, so it can always be killed.

The failure this exists to prevent: the agent dies — crash, SIGKILL, a careless
``systemctl restart`` — while a vLLM server it launched is still holding many gigabytes
of VRAM. Nothing else on the box knows that process was ours, the GPU looks busy, and the
next sweep either fails to allocate or silently competes with a ghost for the device.

So ownership is written to disk *before* the process is launched, not after. A crash in
the window between fork and record would otherwise produce exactly the orphan this file
is meant to make impossible.

Identity is (pid, create_time), never pid alone. PIDs are recycled, and a stale record
naming a recycled pid would have the agent kill an unrelated process — plausibly
something important, on a machine whose whole job is running other people's models.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_STATE_DIR = Path(
    os.environ.get("VLLMBENCH_STATE_DIR", Path.home() / ".local" / "state" / "vllmbench")
)

# How long to wait for a polite shutdown before escalating. vLLM must free VRAM and tear
# down worker processes; a too-short grace period turns a clean exit into an orphan.
TERM_GRACE_SECONDS = 30.0


@dataclass(frozen=True)
class OwnedProcess:
    pid: int
    pgid: int
    create_time: float
    port: int
    config_hash: str
    recorded_at: float


def _proc_create_time(pid: int) -> float | None:
    """Process start time, used to tell a live process from a recycled PID.

    Returns None when the answer cannot be determined, which callers must treat as
    "identity unverifiable" and therefore "do not kill". psutil is a hard dependency
    precisely so this rarely happens: without it there is no portable way to distinguish
    our process from whatever inherited its pid, and the safe fallback (never reap)
    would mean never cleaning up orphaned VRAM.
    """
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:  # noqa: S110 - fall through to the /proc reader below
        pass

    # Fallback for a stripped environment. Field 22 of /proc/<pid>/stat, in clock ticks
    # since boot. The comm field can contain spaces and parentheses, so split after the
    # final ')' rather than from the left.
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        raw = stat_path.read_text()
        fields = raw[raw.rindex(")") + 2 :].split()
        return float(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _process_exists(pid: int) -> bool:
    """Is there a live process at this pid?

    A zombie does not count. ``os.kill(pid, 0)`` succeeds against one, so a caller
    polling for liveness would wait forever on a process that is already dead and merely
    unreaped.
    """
    try:
        import psutil

        process = psutil.Process(pid)
        return process.status() != psutil.STATUS_ZOMBIE
    except Exception:  # noqa: S110 - fall through to the os.kill probe below
        pass

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but belongs to someone else. Not ours to kill.
        return False
    return True


class ProcessRegistry:
    """Records which vLLM process this agent owns, across agent restarts."""

    def __init__(self, state_dir: Path | None = None) -> None:
        self.state_dir = state_dir or DEFAULT_STATE_DIR
        self.state_file = self.state_dir / "owned-process.json"

    def record(self, *, pid: int, pgid: int, port: int, config_hash: str) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        owned = OwnedProcess(
            pid=pid,
            pgid=pgid,
            create_time=_proc_create_time(pid) or 0.0,
            port=port,
            config_hash=config_hash,
            recorded_at=time.time(),
        )
        # Write-then-rename: a half-written record read after a crash would be worse
        # than no record, because it would parse as a different pid.
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(owned)))
        tmp.replace(self.state_file)
        log.debug("recorded ownership of pid %d (pgid %d)", pid, pgid)

    def clear(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.state_file.unlink()

    def read(self) -> OwnedProcess | None:
        try:
            return OwnedProcess(**json.loads(self.state_file.read_text()))
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return None

    def reap_orphan(self) -> OwnedProcess | None:
        """Kill a server left behind by a previous agent lifetime.

        Called at startup. Returns the process it killed, or None if there was nothing
        to do — which is the overwhelmingly common case, and must stay silent enough not
        to look alarming in logs.
        """
        owned = self.read()
        if owned is None:
            return None

        if not _process_exists(owned.pid):
            log.info("stale ownership record for pid %d; process is gone", owned.pid)
            self.clear()
            return None

        # PID recycling guard, and the most safety-critical branch in the agent.
        #
        # This must fail closed. An earlier version required *both* timestamps to be
        # truthy before comparing, so an unverifiable identity — either side missing —
        # skipped the check and killed on pid alone. On a host where pids had been
        # recycled that is how you kill someone else's job; in testing it killed the
        # test runner.
        #
        # So: no verifiable identity, no kill. An orphan that survives is a VRAM leak an
        # operator can see and fix. A wrongly-killed process is silent and unrecoverable.
        current_create_time = _proc_create_time(owned.pid)
        if not owned.create_time or current_create_time is None:
            log.error(
                "cannot verify that pid %d is the process we started "
                "(recorded start %.2f, current %s); refusing to kill it. "
                "If a vLLM server is holding VRAM, stop it by hand.",
                owned.pid,
                owned.create_time,
                "unknown" if current_create_time is None else f"{current_create_time:.2f}",
            )
            self.clear()
            return None

        if abs(current_create_time - owned.create_time) > 1.0:
            log.warning(
                "pid %d exists but started at a different time (%.2f vs recorded %.2f); "
                "refusing to kill a process that is not ours",
                owned.pid,
                current_create_time,
                owned.create_time,
            )
            self.clear()
            return None

        log.warning(
            "reaping orphaned vLLM server from a previous agent lifetime: "
            "pid=%d pgid=%d port=%d config=%s",
            owned.pid,
            owned.pgid,
            owned.port,
            owned.config_hash[:12],
        )
        terminate_group(owned.pgid, owned.pid)
        self.clear()
        return owned


def terminate_group(pgid: int, pid: int, grace: float = TERM_GRACE_SECONDS) -> None:
    """Stop a process group: SIGTERM, wait, then SIGKILL.

    The *group* rather than the process. vLLM spawns worker subprocesses per device, and
    signalling only the parent leaves the workers alive holding VRAM — the exact orphan
    this module exists to prevent, just one level down.
    """
    _signal_group(pgid, pid, signal.SIGTERM)

    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            log.info("pid %d exited on SIGTERM", pid)
            return
        time.sleep(0.25)

    log.warning("pid %d did not exit within %.0fs; sending SIGKILL", pid, grace)
    _signal_group(pgid, pid, signal.SIGKILL)

    # Give the kernel a moment, then report honestly rather than assuming success.
    for _ in range(20):
        if not _process_exists(pid):
            return
        time.sleep(0.25)
    log.error("pid %d survived SIGKILL; VRAM may still be held", pid)


def _signal_group(pgid: int, pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
        return
    except ProcessLookupError:
        return
    except (PermissionError, OSError) as exc:
        log.warning("could not signal process group %d (%s); falling back to pid", pgid, exc)

    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, sig)
