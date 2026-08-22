"""Whether this Python environment satisfies its own declared constraints.

The agent is installed *into vLLM's virtualenv* — that is the documented deployment, and
it is what keeps the agent able to invoke the vLLM binaries it orchestrates. So the
agent's dependency resolution and vLLM's constraints meet in one environment with nothing
arbitrating between them, and they can diverge without anything saying so:

    $ uv pip check
    The package `vllm` requires `fastapi[standard]>=0.133.0,<0.137.0`,
    but `0.141.1` is installed

That happened on the first real GPU host. vLLM kept working, which is luck rather than
design — and the machine it is luck on is the system under test, whose behaviour this
project exists to hold still.

**This reports; it never blocks.** The same reasoning as the vLLM version policy in
CLAUDE.md: refusing to start would put this control plane in the business of deciding what
is wrong with someone's virtualenv, and a false positive would take down a working GPU host
remotely. A measurement taken on an inconsistent environment is not necessarily wrong — it
is unattributable, which is a thing to record rather than a thing to prevent.

Version conflicts only. A *missing* distribution is a different problem with different
causes — optional extras, vendored packages, distro-managed installs — and reporting those
too would bury the one finding that matters in noise nobody acts on. False positives are
the expensive failure here: an operator who learns to ignore this has lost the signal.
"""

from __future__ import annotations

import logging
from importlib import metadata
from typing import TYPE_CHECKING, Any

from vllmbench_protocol import EnvironmentCheck, EnvironmentStatus

if TYPE_CHECKING:
    from packaging.requirements import Requirement

log = logging.getLogger(__name__)

#: Nothing is reported beyond this many conflicts. A wholly broken environment produces
#: pages of them and the first few are enough to act on — this is a status line, not a
#: dependency resolver's diagnostics.
MAX_CONFLICTS = 20


def _packaging() -> Any:
    """``packaging``, if this environment has it, without depending on it.

    Deliberately not declared in the agent's dependencies. This module exists because
    installing the agent into vLLM's virtualenv can disturb that virtualenv, and adding a
    package in order to detect added packages would be a poor joke — the protocol package
    carries the same rule in its own docstring, for the same reason.

    In practice it is always present: it is a build-system staple and vLLM itself requires
    it. When it is not, the check reports UNAVAILABLE, which is why that state exists.
    """
    import packaging.requirements
    import packaging.utils
    import packaging.version

    return packaging


def _installed(pkg: Any) -> dict[str, tuple[str, str]]:
    """Canonical name to (reported name, version) for everything installed.

    Duplicates are possible — two dist-info directories for the same project, which is
    itself a broken environment — and the first wins rather than raising, because a
    reporting function that dies on a broken environment reports nothing about the
    environment it was asked to describe.
    """
    found: dict[str, tuple[str, str]] = {}
    for dist in metadata.distributions():
        try:
            name = dist.metadata["Name"]
            version = dist.version
        except Exception:  # noqa: S112 - a malformed dist-info must not stop a sweep
            continue
        if not name or version is None:
            continue
        found.setdefault(pkg.utils.canonicalize_name(name), (name, version))
    return found


def _conflicts_for(
    pkg: Any,
    dist_name: str,
    dist_version: str,
    requires: list[str],
    installed: dict[str, tuple[str, str]],
) -> list[str]:
    conflicts: list[str] = []
    for raw in requires:
        requirement: Requirement
        try:
            requirement = pkg.requirements.Requirement(raw)
        except Exception:  # noqa: S112 - unparseable metadata is not this host's problem
            continue

        # Requirements gated on an extra are skipped: whether that extra is installed is
        # not recorded anywhere, so evaluating them would invent conflicts for optional
        # dependencies nobody asked for. Under-reporting is the correct direction.
        if requirement.marker is not None:
            try:
                if not requirement.marker.evaluate({"extra": ""}):
                    continue
            except Exception:  # noqa: S112 - an unevaluable marker is not a conflict
                continue

        target = installed.get(pkg.utils.canonicalize_name(requirement.name))
        if target is None:
            # Missing, not conflicting. See the module docstring.
            continue

        installed_name, installed_version = target
        if not requirement.specifier:
            continue
        try:
            parsed = pkg.version.Version(installed_version)
        except pkg.version.InvalidVersion:
            continue
        # prereleases=True so that an installed release candidate is judged against the
        # specifier rather than silently excluded — on this project's own packages, which
        # ship as 1.0.0rcN, the default would skip exactly the versions in play.
        if requirement.specifier.contains(parsed, prereleases=True):
            continue

        conflicts.append(
            f"{dist_name} {dist_version} requires {requirement}, "
            f"but {installed_name} {installed_version} is installed"
        )
    return conflicts


def probe_environment() -> EnvironmentCheck:
    """Check every installed distribution's requirements against what is installed.

    The same question ``pip check`` answers, asked from inside the process so that it
    works whether or not ``pip`` or ``uv`` is on the PATH — which on a systemd-run agent
    in someone else's virtualenv is not a safe assumption.
    """
    try:
        pkg = _packaging()
    except ImportError as exc:
        return EnvironmentCheck(
            status=EnvironmentStatus.UNAVAILABLE, detail=f"packaging is not installed ({exc})"
        )

    try:
        installed = _installed(pkg)
    except Exception as exc:  # reporting failure is a state here, not a crash
        log.warning("could not read installed distributions: %s", exc)
        return EnvironmentCheck(status=EnvironmentStatus.UNAVAILABLE, detail=str(exc))

    conflicts: list[str] = []
    for dist in metadata.distributions():
        if len(conflicts) >= MAX_CONFLICTS:
            break
        try:
            name = dist.metadata["Name"]
            requires = list(dist.requires or [])
        except Exception:  # noqa: S112 - same reasoning as _installed
            continue
        if not name or not requires:
            continue
        conflicts.extend(_conflicts_for(pkg, name, dist.version, requires, installed))

    truncated = len(conflicts) > MAX_CONFLICTS
    conflicts = conflicts[:MAX_CONFLICTS]

    if not conflicts:
        return EnvironmentCheck(status=EnvironmentStatus.OK, distributions=len(installed))
    return EnvironmentCheck(
        status=EnvironmentStatus.CONFLICTS,
        conflicts=conflicts,
        distributions=len(installed),
        detail="more conflicts than are listed" if truncated else None,
    )


def log_environment(check: EnvironmentCheck) -> None:
    """Say it at startup, where an operator installing the agent will see it.

    At WARNING rather than ERROR: nothing is broken yet, and the agent is about to run
    normally. The point is that the moment somebody can act on this is the moment they
    finish the install, not the moment a sweep produces a number nobody trusts.
    """
    if check.status is EnvironmentStatus.OK:
        log.info("environment consistent across %d installed distributions", check.distributions)
        return
    if check.status is EnvironmentStatus.UNAVAILABLE:
        log.warning("could not check environment consistency: %s", check.detail)
        return
    log.warning(
        "this environment does not satisfy its own declared constraints (%d %s). "
        "vLLM may work anyway; runs measured here are recorded as taken on an "
        "inconsistent environment.",
        len(check.conflicts),
        "conflict" if len(check.conflicts) == 1 else "conflicts",
    )
    for conflict in check.conflicts:
        log.warning("  %s", conflict)
