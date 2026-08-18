"""The agent's footprint on the GPU host's disk.

Its resource footprint is part of its contract (CLAUDE.md), and disk is the part that
fails quietly. VRAM leaks are loud — the next run cannot allocate. A slowly filling
filesystem produces a benchmark that dies forty minutes in with a write error, on a host
whose whole purpose is to be running something expensive.

Two things to keep bounded, and they are different problems.

**Leftover working directories.** Every benchmark and every server start writes into a
private temp directory and removes it in a ``finally``. A ``finally`` does not run when
the process is SIGKILLed, so a crash or an abrupt ``systemctl restart`` leaves one behind
— the same class of failure the process reaper exists for, one level down. So they are
swept at startup, on the same principle: the agent cleans up after its previous lifetime
because nothing else on the box knows those directories were ours.

**Headroom before starting.** A disk with no room left turns a long benchmark into a
write error at the end of it. Checking first costs a ``statvfs`` and converts forty
wasted minutes into an immediate, actionable refusal.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: Prefixes this agent creates under the system temp directory. Anything matching one of
#: these is ours by construction, which is what makes sweeping them safe — unlike a
#: pattern that could match a directory some other tool created.
WORKDIR_PREFIXES = ("vllmbench-bench-", "vllmbench-config-")

#: How old a leftover must be before it is swept, in seconds.
#:
#: Not zero. Two agent processes can briefly overlap during a restart, and deleting a
#: directory the outgoing one is still writing into would corrupt a benchmark that was
#: about to succeed. An hour is far longer than that overlap and far shorter than the
#: interval between the crashes this exists to clean up after.
STALE_AFTER_SECONDS = 3600.0

#: Free space required before starting a benchmark or an engine.
#:
#: Sized for what actually gets written, not as a round number: a `--save-result` payload
#: is kilobytes, but vLLM's torch.compile cache and any dataset download share the same
#: filesystem, and those are the things that fill it. A gigabyte is enough headroom that
#: hitting this means something is genuinely wrong rather than merely tight.
DEFAULT_MIN_FREE_BYTES = 1_073_741_824


class DiskFull(RuntimeError):
    """There is not enough room to do this safely."""


@dataclass(frozen=True, slots=True)
class DiskSpace:
    path: Path
    free_bytes: int
    total_bytes: int

    @property
    def free_fraction(self) -> float:
        return self.free_bytes / self.total_bytes if self.total_bytes else 0.0


def disk_space(path: Path | None = None) -> DiskSpace:
    target = path or Path(tempfile.gettempdir())
    usage = shutil.disk_usage(target)
    return DiskSpace(path=target, free_bytes=usage.free, total_bytes=usage.total)


def require_headroom(minimum_bytes: int, path: Path | None = None) -> DiskSpace:
    """Refuse to start work that the disk cannot hold. Returns the space it found.

    Raises rather than warning. A warning here is a line in a log nobody reads until they
    are debugging the write error it predicted.
    """
    space = disk_space(path)
    if space.free_bytes < minimum_bytes:
        raise DiskFull(
            f"{space.path} has {space.free_bytes / 1e9:.2f} GB free, below the "
            f"{minimum_bytes / 1e9:.2f} GB required to start. Free space on the GPU host "
            "before running; a benchmark that fills the disk fails partway through and "
            "wastes the whole run."
        )
    return space


def sweep_stale_workdirs(*, older_than_seconds: float = STALE_AFTER_SECONDS) -> list[Path]:
    """Remove working directories left by a previous agent lifetime.

    Returns what was removed, so startup can say so. Silence here would make a host that
    had been crash-looping for a week look identical to one that had never crashed.
    """
    removed: list[Path] = []
    cutoff = time.time() - older_than_seconds
    root = Path(tempfile.gettempdir())

    for prefix in WORKDIR_PREFIXES:
        for candidate in root.glob(f"{prefix}*"):
            if not candidate.is_dir():
                continue
            try:
                if candidate.stat().st_mtime > cutoff:
                    continue
                shutil.rmtree(candidate, ignore_errors=True)
            except OSError as exc:
                # Never fatal. A directory we cannot remove — a permissions change, a
                # busy mount — is a smaller problem than an agent that refuses to start.
                log.warning("could not remove stale workdir %s: %s", candidate, exc)
                continue
            removed.append(candidate)

    if removed:
        log.info(
            "removed %d stale working director%s left by a previous lifetime",
            len(removed),
            "y" if len(removed) == 1 else "ies",
        )
    return removed
