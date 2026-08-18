"""The agent's disk footprint on the machine under test.

The agent's resource footprint is part of its contract, and disk is the part that fails
quietly: a VRAM leak stops the next run immediately, while a slowly filling filesystem
produces a benchmark that dies forty minutes in, on a host whose whole purpose is running
something expensive.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

from vllmbench_agent.workspace import (
    STALE_AFTER_SECONDS,
    WORKDIR_PREFIXES,
    DiskFull,
    disk_space,
    require_headroom,
    sweep_stale_workdirs,
)


@pytest.fixture
def temp_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the sweeper at a directory of our own.

    Emphatically not the real temp directory: this deletes things, and a test that
    deletes from `/tmp` on a developer's laptop is a test that eventually deletes
    something it should not.
    """
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    return tmp_path


def _aged_dir(root: Path, name: str, *, age_seconds: float) -> Path:
    path = root / name
    path.mkdir()
    (path / "result.json").write_text("{}")
    when = time.time() - age_seconds
    os.utime(path, (when, when))
    return path


class TestSweepingLeftovers:
    def test_a_stale_workdir_is_removed(self, temp_root: Path) -> None:
        """The case a `finally` cannot cover.

        Every benchmark removes its own directory on the way out. A SIGKILL — a crash, an
        abrupt `systemctl restart` — skips that, and nothing else on the box knows the
        directory was ours. Same principle as reaping an orphaned engine, one level down.
        """
        stale = _aged_dir(temp_root, "vllmbench-bench-abc123", age_seconds=STALE_AFTER_SECONDS * 2)

        removed = sweep_stale_workdirs()

        assert removed == [stale]
        assert not stale.exists()

    def test_a_recent_workdir_is_left_alone(self, temp_root: Path) -> None:
        """Two agent processes can briefly overlap during a restart.

        Deleting a directory the outgoing one is still writing into would corrupt a
        benchmark that was about to succeed — turning a tidy-up into the data loss it
        exists to prevent.
        """
        fresh = _aged_dir(temp_root, "vllmbench-bench-live", age_seconds=5)

        assert sweep_stale_workdirs() == []
        assert fresh.exists()

    def test_it_only_touches_directories_it_could_have_created(self, temp_root: Path) -> None:
        """Matching by our own prefix is what makes this safe to run at startup.

        A pattern like `*bench*` would eventually match something belonging to whatever
        else runs on the GPU host, and deleting other people's files from a shared
        machine is not a recoverable mistake.
        """
        someone_else = _aged_dir(temp_root, "pytest-of-someone", age_seconds=99999)
        vllm_cache = _aged_dir(temp_root, "torchinductor_root", age_seconds=99999)

        sweep_stale_workdirs()

        assert someone_else.exists()
        assert vllm_cache.exists()

    def test_both_kinds_of_workdir_are_swept(self, temp_root: Path) -> None:
        """Benchmarks and server configs each get one, and both leak the same way."""
        made = [
            _aged_dir(temp_root, f"{prefix}xyz", age_seconds=STALE_AFTER_SECONDS * 2)
            for prefix in WORKDIR_PREFIXES
        ]

        assert sorted(sweep_stale_workdirs()) == sorted(made)

    def test_a_file_matching_the_prefix_is_ignored(self, temp_root: Path) -> None:
        stray = temp_root / "vllmbench-bench-not-a-directory"
        stray.write_text("x")
        os.utime(stray, (0, 0))

        assert sweep_stale_workdirs() == []
        assert stray.exists()


class TestHeadroom:
    def test_it_refuses_when_the_disk_is_too_full(self, tmp_path: Path) -> None:
        """Raising rather than warning.

        A warning here is a line in a log nobody reads until they are debugging the write
        error it predicted. Asked for more space than any real filesystem has, so this
        does not depend on the state of the machine running the test.
        """
        with pytest.raises(DiskFull) as exc:
            require_headroom(2**60, tmp_path)

        # The message has to say what to do, not just that something is wrong.
        assert "Free space on the GPU host" in str(exc.value)
        assert "GB free" in str(exc.value)

    def test_it_returns_what_it_found_when_there_is_room(self, tmp_path: Path) -> None:
        space = require_headroom(1, tmp_path)

        assert space.free_bytes > 0
        assert 0.0 < space.free_fraction <= 1.0

    def test_a_minimum_of_zero_never_refuses(self, tmp_path: Path) -> None:
        """Disabling the check is a choice an operator can make, not an accident."""
        assert require_headroom(0, tmp_path).free_bytes >= 0

    def test_it_reports_the_path_it_measured(self, tmp_path: Path) -> None:
        """Which filesystem is full is half the answer on a host with several mounts."""
        assert disk_space(tmp_path).path == tmp_path
