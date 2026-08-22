"""Detecting a virtualenv that no longer satisfies its own declared constraints.

The case this exists for is real and was found on a real host: the agent is installed into
vLLM's environment, uv resolved `fastapi` to a version above vLLM's ceiling, and nothing
said so. vLLM kept working. The next such pair might not, and it would break the machine
whose behaviour the whole project is trying to hold still.

Most of these tests assert what the check *declines* to report. That is the substance of
it. A conflict report nobody trusts is worthless, so under-reporting is the correct
direction on every ambiguous case — and each of those cases needs to stay decided.
"""

from __future__ import annotations

import packaging.requirements
import packaging.utils
import packaging.version

from vllmbench_agent.environment import MAX_CONFLICTS, _conflicts_for, probe_environment
from vllmbench_protocol import EnvironmentStatus

PKG = packaging


def conflicts(requires: list[str], installed: dict[str, str], requirer: str = "vllm") -> list[str]:
    return _conflicts_for(
        PKG,
        requirer,
        "0.25.1",
        requires,
        {packaging.utils.canonicalize_name(k): (k, v) for k, v in installed.items()},
    )


class TestTheCaseThisWasBuiltFor:
    def test_a_version_above_a_declared_ceiling_is_a_conflict(self) -> None:
        """Verbatim from `uv pip check` on the first real GPU host."""
        found = conflicts(["fastapi[standard]>=0.133.0,<0.137.0"], {"fastapi": "0.141.1"})
        assert len(found) == 1
        # All three facts, because "there is a conflict" is not actionable and
        # "vllm needs older fastapi than you have" is.
        assert "vllm 0.25.1" in found[0]
        assert "fastapi" in found[0]
        assert "0.141.1" in found[0]

    def test_a_version_inside_the_ceiling_is_not(self) -> None:
        assert conflicts(["fastapi[standard]>=0.133.0,<0.137.0"], {"fastapi": "0.136.2"}) == []


class TestWhatItDeclinesToReport:
    def test_a_missing_distribution_is_not_a_conflict(self) -> None:
        """A different problem with different causes, and the noisier one by far.

        Optional extras and distro-managed packages make "missing" common in a working
        environment. Reporting them would bury the finding that matters.
        """
        assert conflicts(["fastapi>=0.133.0"], {}) == []

    def test_requirements_gated_on_an_extra_are_skipped(self) -> None:
        """Whether the extra was installed is recorded nowhere, so this cannot be judged."""
        assert conflicts(['pytest>=8.0; extra == "test"'], {"pytest": "1.0"}) == []

    def test_a_marker_that_excludes_this_interpreter_is_skipped(self) -> None:
        assert conflicts(['tomli>=2.0; python_version < "3.0"'], {"tomli": "0.1"}) == []

    def test_an_unparseable_requirement_does_not_raise(self) -> None:
        assert conflicts(["this is not a requirement !!"], {"fastapi": "0.141.1"}) == []

    def test_an_unparseable_installed_version_is_skipped(self) -> None:
        """Distro-patched versions exist and are not this check's business to adjudicate."""
        assert conflicts(["fastapi>=0.133.0"], {"fastapi": "0.141.1-ubuntu3.14"}) == []

    def test_a_requirement_with_no_specifier_is_skipped(self) -> None:
        assert conflicts(["fastapi"], {"fastapi": "0.1.0"}) == []


class TestPrereleases:
    """This project ships as 1.0.0rcN, so excluding prereleases would skip its own packages."""

    def test_a_release_candidate_satisfies_a_matching_pin(self) -> None:
        assert conflicts(["vllmbench-protocol==1.0.0rc3"], {"vllmbench-protocol": "1.0.0rc3"}) == []

    def test_a_release_candidate_can_still_conflict(self) -> None:
        found = conflicts(["vllmbench-protocol==1.0.0rc3"], {"vllmbench-protocol": "1.0.0rc1"})
        assert len(found) == 1

    def test_names_are_matched_canonically(self) -> None:
        """`vllmbench_protocol` and `vllmbench-protocol` are the same distribution."""
        found = conflicts(["vllmbench_protocol<1.0"], {"vllmbench-protocol": "1.0.0rc3"})
        assert len(found) == 1


class TestProbingThisEnvironment:
    def test_it_returns_a_decided_state(self) -> None:
        check = probe_environment()
        assert check.status in (EnvironmentStatus.OK, EnvironmentStatus.CONFLICTS)
        # Never NOT_REPORTED: that state belongs to the control plane, describing an agent
        # that said nothing. An agent that ran the check has, by definition, reported.
        assert check.status is not EnvironmentStatus.NOT_REPORTED

    def test_it_says_how_much_it_looked_at(self) -> None:
        """So an empty conflict list can be told apart from an environment it could not read."""
        check = probe_environment()
        assert check.distributions is not None
        assert check.distributions > 0

    def test_conflicts_are_bounded(self) -> None:
        check = probe_environment()
        assert len(check.conflicts) <= MAX_CONFLICTS
