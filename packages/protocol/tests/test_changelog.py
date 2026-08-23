"""The changelog generator's parser, which is the part that can lose a change silently.

`pytest` only collects under `packages/`, so repo-level tooling is tested from here — the
same place `scripts/check_versions.py` is exercised from.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_spec = importlib.util.spec_from_file_location(
    "generate_changelog", ROOT / "scripts" / "generate_changelog.py"
)
assert _spec is not None and _spec.loader is not None
changelog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(changelog)


@pytest.mark.parametrize(
    ("subject", "kind", "scope", "text", "pr"),
    [
        ("feat: a thing", "feat", None, "a thing", None),
        ("fix(agent): another thing (#12)", "fix", "agent", "another thing", "12"),
        ("feat(api)!: a breaking thing (#3)", "feat", "api", "a breaking thing", "3"),
        # A subject that itself ends in parentheses must not lose them to the PR pattern.
        ("docs: why (and when) it matters", "docs", None, "why (and when) it matters", None),
    ],
)
def test_it_parses_conventional_subjects(
    subject: str, kind: str, scope: str | None, text: str, pr: str | None
) -> None:
    match = changelog.SUBJECT.match(subject)
    assert match is not None, subject
    assert match["type"] == kind
    assert match["scope"] == scope
    assert match["subject"] == text
    assert match["pr"] == pr


def test_a_malformed_subject_is_an_error_not_a_skipped_line() -> None:
    """The failure mode of a lenient parser is a change that never appears.

    A dropped line is indistinguishable from a release that did not contain the change,
    so this exits rather than continuing.
    """
    with pytest.raises(SystemExit) as excinfo:
        changelog.render_section("1.2.3", "2026-01-01", [("abc123def456", "no type here")])
    assert "not a Conventional Commit subject" in str(excinfo.value)


def test_release_chores_do_not_appear_under_their_own_heading() -> None:
    lines = changelog.render_section(
        "1.2.3", "2026-01-01", [("a" * 40, "chore: 1.2.3"), ("b" * 40, "feat: a thing")]
    )
    body = "\n".join(lines)
    assert "a thing" in body
    assert "chore" not in body
    assert "1.2.3" in lines[0]


def test_a_section_with_nothing_worth_listing_is_omitted_entirely() -> None:
    # Otherwise a release of pure housekeeping renders as a heading with no content,
    # which reads as a generator bug rather than as an accurate empty list.
    assert changelog.render_section("1.2.3", "2026-01-01", [("c" * 40, "chore: bump")]) == []


def test_breaking_changes_are_listed_first_regardless_of_type() -> None:
    lines = changelog.render_section(
        "2.0.0",
        "2026-01-01",
        [("d" * 40, "docs: a doc"), ("e" * 40, "fix(db)!: a breaking fix")],
    )
    body = "\n".join(lines)
    assert body.index("Breaking changes") < body.index("Documentation")
