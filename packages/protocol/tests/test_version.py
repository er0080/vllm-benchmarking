"""The lockstep-versioning guarantee, asserted rather than assumed.

CLAUDE.md commits to the agent and control plane shipping as one version. That guarantee
is only worth anything if something enforces it, and the agent ships as a separately built
wheel installed into a user's vLLM environment — the one place drift is both easy and
invisible.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import vllmbench_agent
import vllmbench_api
import vllmbench_db
import vllmbench_mockagent
import vllmbench_orchestrator
import vllmbench_protocol

ROOT = Path(__file__).resolve().parents[3]


def test_version_file_matches_protocol_package() -> None:
    assert vllmbench_protocol.__version__ == (ROOT / "VERSION").read_text().strip()


def test_every_package_reports_the_same_version() -> None:
    versions = {
        "protocol": vllmbench_protocol.__version__,
        "db": vllmbench_db.__version__,
        "api": vllmbench_api.__version__,
        "orchestrator": vllmbench_orchestrator.__version__,
        "agent": vllmbench_agent.__version__,
        "mockagent": vllmbench_mockagent.__version__,
    }
    assert len(set(versions.values())) == 1, versions


def test_check_versions_script_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_versions.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_protocol_version_is_a_positive_int() -> None:
    # Distinct from __version__ on purpose: this one gates whether two sides may talk
    # at all, and increments only on a breaking change to the agent's HTTP API.
    assert isinstance(vllmbench_protocol.PROTOCOL_VERSION, int)
    assert vllmbench_protocol.PROTOCOL_VERSION >= 1


def test_the_agent_pins_protocol_exactly() -> None:
    """The published wheel must not be able to resolve this name from an index.

    ``vllmbench-protocol`` is on no package index. An unpinned requirement inside a wheel
    people download is an instruction to go and fetch that name from PyPI, where it does
    not exist — and an unregistered name in a published install instruction belongs to
    whoever registers it first. The documented install passes both wheel files so the
    lookup never happens; this pin is what stands behind that when someone installs only
    one of them, and it also makes a half-finished upgrade fail at install rather than at
    connect.
    """
    import tomllib

    data = tomllib.loads((ROOT / "packages/agent/pyproject.toml").read_text())
    pins = [
        dep
        for dep in data["project"]["dependencies"]
        if dep.partition("==")[0].strip() == "vllmbench-protocol"
    ]
    assert pins == [f"vllmbench-protocol=={(ROOT / 'VERSION').read_text().strip()}"], (
        f"expected one exact pin, found {pins!r}"
    )
