# ADR 0005 — Structured logs, and redaction by value rather than by pattern

- **Status:** accepted
- **Date:** 2026-08-17
- **Milestone:** 0.9.0 — Hardening
- **Issue:** none — issues were optional at this milestone; see CLAUDE.md

## Context

A sweep is hours of interleaved output from three services on two hosts. Until now each
formatted its own text lines, and answering "what did this run do" meant grep and hope:
the run id appears in some messages and not others, nothing correlates the orchestrator's
view of a run with the agent's, and libraries — SQLAlchemy, httpx, uvicorn — contribute
lines with no idea what run is in flight.

Separately, CLAUDE.md has always said "never log a token", and until now that was a rule
enforced by care at each call site. Care at each call site does not survive contact with
the failures that matter, because the leaks that happen are in text nobody wrote. A
Postgres connection failure quotes the whole DSN back — password included — from inside
the driver, and that string reaches a log line through `_wait_for_database`, through the
health check, through any `log.exception` that happens to wrap a connection error.

## Decision

### Redaction is by registered value, never by pattern

The tempting design is a regex: strip anything shaped like a bearer token, anything after
`password=`, anything that looks high-entropy. It reads as thorough and fails silently in
both directions. It cannot know what *this* deployment's token looks like, so it misses
real ones; and it mangles innocent text that happens to match, which produces a log that
is unreadable in exactly the incident where it is needed.

We hold our own secrets — they are in settings, at startup, by definition. Registering
those exact values is precise, and it applies to text nobody wrote deliberately, which is
where the leak actually is.

The consequence is a rule with teeth: **a secret added to settings must be added to that
service's `configure_logging` call in the same change.** A test asserts each service
registers what it holds, because the failure mode is otherwise entirely silent — no error,
no failing test, and a credential in a log file the first time something goes wrong near
it.

### Scrubbing happens in the formatter, not in a filter

A `logging.Filter` runs before `%`-formatting and before the traceback is rendered. To
inspect what actually gets written it would have to reimplement both — and the traceback
is precisely where an unplanned secret appears. The formatter sees the final string.

### There is exactly one handler

`configure_logging` replaces the root handler rather than adding to it, and unhooks
uvicorn's own loggers (which it installs with `propagate=False`, putting its access log
outside ours). A second handler anywhere would be a path around the redaction, and the
guarantee only means something if there is one way out.

### Values below eight characters are refused, loudly

Not a password-strength policy — a floor on what is safe to *substitute*. Replacing every
occurrence of a three-character value would corrupt unrelated log text. Eight matches
`AgentSettings.token`'s own minimum, so a real token always qualifies. Refusing silently
would be the worst option available: the caller then believes something is masked and it
is not. The warning never names the value it declined to protect, which would put it in
the log by way of complaining that it is not in the log.

### Context is bound, not threaded

`bound(run_id=..., sweep_id=...)` sets a contextvar; a filter copies those fields onto
every record emitted inside the block. A contextvar rather than a thread-local because
everything here is asyncio, and a filter rather than a `LoggerAdapter` because the fields
must reach records this project did not create — SQLAlchemy's warning during a run should
carry the run id, and SQLAlchemy has never heard of us.

### JSON off a TTY, text on one

Unset `VLLMBENCH_LOG_FORMAT` decides by `sys.stderr.isatty()`. That default needs no
thought in either place it matters: a container writes JSON for whatever collects it, and
a developer's terminal stays readable.

## Alternatives rejected

**A logging library — structlog, loguru.** Both are good and neither earns its place here.
The agent installs into a user's vLLM environment, where every dependency is a dependency
on the system under test; `protocol` is deliberately kept to the standard library. What we
need is a formatter, a filter and a contextvar, which is roughly a hundred lines.

**Redacting at the call sites that log credentials.** That is where we were, and it
protects only the lines someone thought about. The DSN-in-a-driver-exception case is not
one of those lines.

**A generated correlation id per HTTP request.** The API binds method and path instead.
This control plane serves a browser and a handful of agents, not a fleet behind a load
balancer, and "which endpoint was slow" is the question that gets asked. A correlation id
is the right answer the day there is more than one hop; adding it then is a two-line
change.

## Consequences

- A run's whole trail across three services is `jq 'select(.run_id == "...")'`.
- Adding a credential to settings now has a second obligation, and a test that enforces it.
- Log output changes shape in containers. Nothing consumes it yet, which is the cheapest
  moment for that to happen.
