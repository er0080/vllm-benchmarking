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


def test_every_compose_image_pins_the_same_release() -> None:
    """The five published images must move together, and never partly.

    Five services carry the tag independently, so a bump that misses one leaves a stack
    running four services from one release and one from another. Nothing errors: compose
    starts them all, health checks pass, and `migrate` — whichever version it happens to
    be — applies its own idea of head to the database the others then read.
    """
    import re

    text = (ROOT / "compose.yaml").read_text()
    pins = re.findall(
        r"image: ghcr\.io/[\w./-]+:\$\{VLLMBENCH_VERSION:-([^}]+)\}",
        text,
    )
    assert len(pins) == 5, f"expected five pinned images, found {len(pins)}: {pins}"
    assert len(set(pins)) == 1, f"compose.yaml pins disagree: {sorted(set(pins))}"


def test_the_compose_pin_is_not_wired_to_version() -> None:
    """Deliberately unlocked, and worth a test so nobody 'fixes' it into lockstep.

    Every other version string in this repository is held equal to VERSION. This one must
    not be. VERSION names what is being built next; the pin names a tag that already
    exists on GHCR, and between a release bump landing on main and its tag being pushed
    there is nothing published under VERSION's name. Wiring them together would break
    `docker compose up` on main — and CI's documented-install job with it — for the
    duration of every release.

    So this asserts only the property that actually matters: the pin is a literal, not a
    reference to VERSION. It is allowed to equal VERSION (it does, right after a release)
    and allowed to lag it.
    """
    text = (ROOT / "compose.yaml").read_text()
    assert "${VERSION" not in text
    assert "$(cat VERSION)" not in text
