#!/usr/bin/env python3
"""Derive CHANGELOG.md from the Conventional Commit history.

Squash merge makes the pull request title the permanent commit subject, so the history
already *is* the changelog — grouped by release rather than by date. Deriving the file
keeps one source of truth; a hand-maintained one is a second place for the same fact to be
wrong, and the two only ever disagree quietly.

Two decisions worth knowing before editing this:

**Subjects are reproduced verbatim.** They are written in this repository as statements
about behaviour — "the UI stopped working whenever the api container moved" — and that is
what makes a generated list readable. Reformatting them into an imperative house style
would throw away the only thing that makes the output worth reading.

**A subject that does not parse is an error, not a skipped line.** The failure mode of a
lenient parser is a change that silently never appears, which is indistinguishable from a
release that did not contain it.

The section for commits after the last tag is titled from ``VERSION`` when that version is
not yet tagged, and "Unreleased" when it is. That is what makes the output stable across
the moment a tag is pushed: a release chore PR bumps ``VERSION``, this file gains that
section, and tagging afterwards produces byte-identical output.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SUBJECT = re.compile(
    r"^(?P<type>[a-z]+)"
    r"(?:\((?P<scope>[^)]+)\))?"
    r"(?P<breaking>!)?"
    r": (?P<subject>.+?)"
    r"(?: \(#(?P<pr>\d+)\))?$"
)

# Order is the order sections appear. `chore` is deliberately absent: a release chore
# restates the heading it would sit under, and the rest is housekeeping nobody reads a
# changelog for.
SECTIONS = [
    ("feat", "Features"),
    ("fix", "Fixes"),
    ("perf", "Performance"),
    ("refactor", "Refactoring"),
    ("docs", "Documentation"),
    ("test", "Tests"),
    ("ci", "CI"),
    ("build", "Build"),
]
SKIP = {"chore", "style"}

REPO = "https://github.com/er0080/vllm-benchmarking"


def git(*args: str) -> str:
    # argv is a fixed list built here, never a shell string, and every caller passes
    # literals or a tag name that came out of `git tag` itself.
    return subprocess.run(  # noqa: S603 - fixed argv, never a shell string
        ["git", *args],  # noqa: S607 - git off PATH, as elsewhere in this repository
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def tags() -> list[str]:
    """Tags oldest first. Sorted by version, not by creation date.

    Creation date orders by when someone happened to push, which is the same thing right
    up until a patch release is cut from an older branch.
    """
    out = git("tag", "--list", "v*", "--sort=v:refname")
    return [t for t in out.splitlines() if t]


def commits(spec: str) -> list[tuple[str, str]]:
    out = git("log", "--no-merges", "--format=%H%x1f%s", spec)
    pairs = []
    for line in out.splitlines():
        if not line:
            continue
        sha, _, subject = line.partition("\x1f")
        pairs.append((sha, subject))
    return pairs


def section_date(spec: str) -> str:
    """The date of the newest commit in a span.

    Deliberately not the tag's own date. A section must render identically whether its
    commits are reached through a tag or as the not-yet-tagged tail, and a tag carries a
    creation date of its own that would differ by however long passed between merging a
    release bump and pushing the tag.
    """
    return git("log", "-1", "--format=%ad", "--date=short", spec)


def render_section(title: str, date: str, entries: list[tuple[str, str]]) -> list[str]:
    grouped: dict[str, list[str]] = {}
    for sha, subject in entries:
        match = SUBJECT.match(subject)
        if match is None:
            sys.exit(
                f"{sha[:12]} is not a Conventional Commit subject: {subject!r}\n"
                "Every subject must parse, because a lenient parser drops changes silently."
            )
        kind = match["type"]
        if kind in SKIP and not match["breaking"]:
            continue

        scope = f"**{match['scope']}:** " if match["scope"] else ""
        text = f"{scope}{match['subject']}"
        if match["pr"]:
            text += f" ([#{match['pr']}]({REPO}/pull/{match['pr']}))"
        if match["breaking"]:
            grouped.setdefault("BREAKING", []).append(text)
            continue
        grouped.setdefault(kind, []).append(text)

    if not grouped:
        return []

    lines = [f"## {title}" + (f" — {date}" if date else ""), ""]
    if "BREAKING" in grouped:
        lines += ["### Breaking changes", ""]
        lines += [f"- {item}" for item in grouped["BREAKING"]]
        lines.append("")
    for kind, heading in SECTIONS:
        if kind not in grouped:
            continue
        lines += [f"### {heading}", ""]
        lines += [f"- {item}" for item in grouped[kind]]
        lines.append("")
    return lines


def build() -> str:
    version = (ROOT / "VERSION").read_text().strip()
    known = tags()

    lines = [
        "# Changelog",
        "",
        "Generated from the Conventional Commit history by",
        "`scripts/generate_changelog.py`. Do not edit by hand — run `make changelog`.",
        "",
        "Squash merge makes each pull request title the permanent commit subject, so this",
        "file is that history grouped by release. The reasoning behind a change lives in",
        f"its pull request body; [ROADMAP.md]({REPO}/blob/main/ROADMAP.md) narrates what",
        "each release candidate was for and what it caught.",
        "",
    ]

    # Newest first, which is the order anyone reads a changelog in.
    #
    # Commits after the last tag appear only when VERSION names a release that does not
    # exist yet — that is, on a release chore branch, where they are about to become that
    # release. There is deliberately no "Unreleased" section: a branch's own commits have
    # no pull request number until the moment they are squashed onto main, so any section
    # containing them is guaranteed to differ from what main produces minutes later. The
    # file would be stale on main the instant it was merged, and `--check` would pass on
    # the branch while failing on the next one.
    #
    # The release chore commit itself is the exception that makes this work: it is a
    # `chore:` and therefore filtered, so its own missing number never matters, and every
    # other commit in the range was merged with its number already.
    if f"v{version}" not in known:
        pending = commits(f"{known[-1]}..HEAD" if known else "HEAD")
        lines += render_section(version, section_date("HEAD"), pending)

    for newer, older in zip(reversed(known), [*reversed(known[:-1]), None], strict=True):
        span = f"{older}..{newer}" if older else newer
        lines += render_section(newer.lstrip("v"), section_date(newer), commits(span))

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    text = build()
    target = ROOT / "CHANGELOG.md"

    if "--check" in sys.argv:
        if not target.is_file():
            print("CHANGELOG.md does not exist; run `make changelog`", file=sys.stderr)
            return 1
        if target.read_text() != text:
            print(
                "CHANGELOG.md is out of date with the commit history.\n"
                "Run `make changelog` and commit the result.",
                file=sys.stderr,
            )
            return 1
        print("CHANGELOG.md matches the commit history.")
        return 0

    target.write_text(text)
    print(f"wrote {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
