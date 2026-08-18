"""Catching the credentials nobody got round to changing.

`.env.example` exists so that a first install is one copy away from running, and the cost
of that convenience is that it ships values which are not secrets. `change-me` is nine
characters, so it clears every length check in this system, and a stack holding it starts
cleanly and reports itself healthy.

The failure this prevents is not a crash. It is a deployment running for months on a
shared secret that is written down in a public repository, with nothing having ever said
so. That is the same reasoning the MCP surface already applies to its own token: a
credential that authenticates nothing should be loud about it.

Warned rather than refused, deliberately. Refusing would stop a laptop evaluation with the
mock agent, where there is no secret to protect and no reason to demand one — and a person
who is told plainly can decide. The values are never logged, only the names that hold
them: complaining about a credential is a poor reason to put it in a log file.
"""

from __future__ import annotations

import logging

#: Values `.env.example` ships, plus the ones people reach for when replacing them.
#: Compared case-insensitively after trimming.
PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "change-me",
        "changeme",
        "change_me",
        "changethis",
        "please-change",
        "secret",
        "password",
        "token",
        "vllmbench",
        "xxx",
        "todo",
    }
)


def is_placeholder(value: str | None) -> bool:
    return value is not None and value.strip().lower() in PLACEHOLDERS


def warn_about_placeholders(log: logging.Logger, /, **secrets: str | None) -> list[str]:
    """Complain about every secret still holding an example value, and return their names.

    Keyword names are the environment variables as an operator knows them, because the
    next step is editing one of those and nothing else identifies which.
    """
    found = [name for name, value in secrets.items() if is_placeholder(value)]
    for name in found:
        log.warning(
            "%s is still an example value from .env.example. Anyone with the repository "
            "has it. Generate one with: python3 -c 'import secrets; "
            "print(secrets.token_urlsafe(32))'",
            name,
        )
    return found
