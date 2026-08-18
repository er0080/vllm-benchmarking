"""Logging that can be queried, and that cannot leak a secret.

Lives in ``protocol`` because it is the one package every service already depends on,
including the agent — and it costs the GPU host nothing, being pure standard library.

Two problems, and they want different solutions.

**Structure.** A sweep is hours of interleaved output from three services. Answering
"what did this run do" from a text log means grep and hope; the run id is in some lines
and not others, and nothing correlates the orchestrator's view with the agent's. So
records carry bound context — run, sweep, host — set once by a context manager and
attached to everything logged inside it, including by libraries that have never heard of
this project. Then the question is `jq 'select(.run_id == "...")'`.

**Secrets.** Redaction here is by *value*, never by pattern. Pattern matching — looking
for things shaped like a token — is the design that feels thorough and fails silently:
it cannot know what this deployment's token looks like, so it either misses real ones or
mangles innocent text. We hold our own secrets in settings; registering them is exact,
and the redaction then applies to text nobody wrote deliberately, which is where leaks
actually come from. A Postgres connection failure quotes the DSN, password included, and
nobody wrote that log line.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

#: Substituted for any registered secret. Greppable on purpose: seeing this in a log is
#: the redaction working, and its absence where one was expected is worth noticing.
REDACTED = "[redacted]"

#: Shortest value that may be registered as a secret.
#:
#: Not a policy about password strength — a floor on what is safe to *substitute*.
#: Replacing every occurrence of a three-character value would corrupt unrelated log
#: text, and a log mangled beyond reading is its own kind of failure. It matches
#: `AgentSettings.token`'s own minimum, so a real token always qualifies.
MIN_SECRET_LENGTH = 8

log = logging.getLogger(__name__)

# Bound context, per task. A contextvar rather than a thread-local: everything here is
# asyncio, and two concurrently-executing runs on one thread must not see each other's
# fields.
# `None` rather than `{}` as the default: a mutable default on a ContextVar is one
# shared object across every context, and a single in-place mutation anywhere would leak
# fields between unrelated runs. Nothing here mutates in place, but the default that
# cannot be misused is the one to have.
_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "vllmbench_log_context", default=None
)


def _fields() -> dict[str, Any]:
    return _context.get() or {}


_secrets: set[str] = set()

# Keys `logging` puts on every record. Anything else a caller attached via `extra=` is
# ours and belongs in the output, so the formatter diffs against this rather than keeping
# its own list of what to emit.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
    # uvicorn attaches an ANSI-coloured copy of its own message to every record. It is a
    # duplicate of `message` with escape codes in it, and emitting it would put terminal
    # control sequences inside a JSON string.
    "color_message",
}


def redact(*values: str | None) -> None:
    """Register secrets to strip from every log line.

    Idempotent and additive, so settings loaded at different points in startup can each
    register what they hold without coordinating.

    Values below :data:`MIN_SECRET_LENGTH` are refused *loudly*. Silently ignoring them
    would be the worst outcome available: the caller believes something is protected and
    it is not.
    """
    for value in values:
        if not value:
            continue
        if len(value) < MIN_SECRET_LENGTH:
            # The value itself is never named here, for the obvious reason.
            log.warning(
                "refusing to redact a value shorter than %d characters; it would corrupt "
                "unrelated log text. This secret will NOT be masked.",
                MIN_SECRET_LENGTH,
            )
            continue
        _secrets.add(value)


def scrub(text: str) -> str:
    """Replace every registered secret in ``text``.

    Longest first, so a secret that contains another does not leave the shorter one's
    replacement embedded in a half-redacted string.
    """
    for secret in sorted(_secrets, key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    return text


@contextmanager
def bound(**fields: Any) -> Iterator[None]:
    """Attach fields to everything logged inside this block.

    Nests: an inner block adds to the outer one rather than replacing it, so binding a
    sweep and then a run inside it yields records carrying both.
    """
    token = _context.set({**_fields(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def context() -> dict[str, Any]:
    """The currently bound fields. Mostly for tests and for building error responses."""
    return dict(_fields())


class ContextFilter(logging.Filter):
    """Copies bound fields onto each record.

    A filter rather than an adapter so it reaches records this project did not create —
    uvicorn's access lines and SQLAlchemy's warnings land inside the same bound block and
    should carry the same run id.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _fields().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line.

    Scrubbing happens here, on the fully rendered text, rather than in a filter. A filter
    sees the record before ``%``-formatting and before the traceback is rendered, so it
    would have to reimplement both to inspect what actually gets written — and the
    traceback is exactly where an unplanned secret shows up.
    """

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return scrub(json.dumps(payload, default=str))


class TextFormatter(logging.Formatter):
    """The human-readable form, for a terminal. Same scrubbing, same bound fields."""

    def __init__(self, service: str) -> None:
        super().__init__("%(asctime)s %(levelname)-5s [%(name)s] %(message)s")
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED and not key.startswith("_")
        }
        if extras:
            rendered += " " + " ".join(f"{k}={v}" for k, v in sorted(extras.items()))
        return scrub(rendered)


def configure_logging(
    service: str,
    *,
    level: str = "INFO",
    fmt: str | None = None,
    secrets: Iterable[str | None] = (),
) -> None:
    """Install the root handler every service logs through.

    ``fmt`` is ``json`` or ``text``; unset means read ``VLLMBENCH_LOG_FORMAT``, and
    failing that decide by whether stderr is a terminal. That default is the one that
    needs no thought in either place it matters: a container writes JSON for whatever
    collects it, and a developer's terminal stays readable.

    The root logger is replaced rather than added to, so uvicorn's and SQLAlchemy's
    records go through the same formatter — and therefore the same scrubbing. A second
    handler installed elsewhere would be a path around the redaction, which is the one
    thing this must not have.
    """
    redact(*secrets)

    chosen = (
        fmt or os.environ.get("VLLMBENCH_LOG_FORMAT") or ("text" if sys.stderr.isatty() else "json")
    )
    formatter: logging.Formatter = (
        TextFormatter(service) if chosen == "text" else JsonFormatter(service)
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    handler.addFilter(ContextFilter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    # uvicorn installs handlers on its own loggers with propagate=False, which puts its
    # access log outside the root handler — and therefore outside the redaction. Under
    # `uvicorn.run(..., log_config=None)` it never does this, but the compose services
    # start uvicorn from the command line where there is no such option, so the loggers
    # are unhooked here instead. Doing it after uvicorn's own setup is what makes it
    # work in both cases: this runs from the app's lifespan, which is later.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
