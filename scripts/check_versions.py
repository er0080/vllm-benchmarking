#!/usr/bin/env python3
"""Enforce lockstep versioning across the workspace.

The agent and the control plane ship as one version (CLAUDE.md, "Versioning"), so every
package's ``version`` must equal the root ``VERSION`` file, and so must
``vllmbench_protocol.__version__``.

This is checked rather than derived because the agent is installed separately, into a
user's vLLM environment, from a wheel built independently of the control-plane images. A
drifting version there is not a cosmetic problem: it is how a GPU host ends up running an
agent that disagrees with the orchestrator about what a field means.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def expected_version() -> str:
    return (ROOT / "VERSION").read_text().strip()


def pyproject_versions() -> dict[Path, str]:
    found: dict[Path, str] = {}
    for path in [ROOT / "pyproject.toml", *sorted(ROOT.glob("packages/*/pyproject.toml"))]:
        data = tomllib.loads(path.read_text())
        version = data.get("project", {}).get("version")
        if version is not None:
            found[path.relative_to(ROOT)] = version
    return found


def protocol_dunder_version() -> tuple[Path, str] | None:
    path = ROOT / "packages/protocol/src/vllmbench_protocol/version.py"
    match = re.search(r'^__version__ = "([^"]+)"', path.read_text(), re.MULTILINE)
    if match is None:
        return None
    return path.relative_to(ROOT), match.group(1)


def main() -> int:
    expected = expected_version()
    mismatches: list[str] = []

    for path, version in pyproject_versions().items():
        if version != expected:
            mismatches.append(f"  {path}: {version}")

    dunder = protocol_dunder_version()
    if dunder is None:
        mismatches.append("  packages/protocol/.../version.py: __version__ not found")
    elif dunder[1] != expected:
        mismatches.append(f"  {dunder[0]}: __version__ = {dunder[1]}")

    if mismatches:
        print(f"Version mismatch. VERSION says {expected!r}, but:", file=sys.stderr)
        print("\n".join(mismatches), file=sys.stderr)
        return 1

    print(f"All package versions match VERSION ({expected}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
