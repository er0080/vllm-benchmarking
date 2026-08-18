# ADR 0004 — Failures are classified, and what we allow ourselves to infer

- **Status:** accepted
- **Date:** 2026-08-17
- **Milestone:** 0.9.0 — Hardening
- **Issue:** none — issues were optional at this milestone; see CLAUDE.md
- **Supersedes:** nothing

> Accepted. The decisions below are binding; changing one requires a superseding ADR, not
> an edit to this file. Rejected alternatives are retained deliberately — they are the
> record of why the project is not doing the obvious other thing.

---

## Context

Through 0.8.0 a failed run recorded `error`: free text, usually a wall of vLLM traceback.
That is the right thing to keep and the wrong thing to have only. It answers "what
happened to this run" and cannot answer "what happened to this sweep" — eleven failed
points give eleven walls of text and no way to see that nine of them are the same cause.
Grouping free text is guesswork. Grouping a column is a `GROUP BY`.

0.9.0's bar is that every failure mode found during 0.2–0.8 has a *defined* behaviour and
a test. Defined behaviour means more than "does not hang": it means the system says which
failure this was, in a vocabulary that is stable enough to count.

The risk in doing this is obvious and is the reason it needed a decision rather than a
patch. A classifier that guesses produces confident wrong labels, and a confident wrong
label is worse than no label — it sends the reader to fix the thing it names. This
repository exists to avoid exactly that class of error in measurements; it should not
introduce it in diagnostics.

## Decision

### 1. Structural evidence first, text last

Most kinds need no interpretation at all. The control plane knows, from its own position,
whether the connection failed, whether the token was refused, whether the protocol
handshake mismatched, and which phase of the run it was in when something threw. The phase
in particular is evidence that exists only at the call site and cannot be recovered from
the message afterwards — so it is fixed there, by wrapping each phase in a context manager
that supplies the default kind for failures raised inside it.

Only one distinction genuinely requires reading text: *why* an engine refused to start. Out
of memory and rejected-configuration have opposite responses — one is a sweep point that
asked for more than the card has, the other is a config that is simply wrong — and vLLM's
own output is the only witness.

### 2. Text patterns come from captured payloads, never documentation

The same rule CLAUDE.md already applies to `--save-result` and `/metrics`, for the same
reason. Three failures were provoked from the pinned `vllm/vllm-openai-cpu` image and
checked in as fixtures:

| Fixture | What produced it |
| --- | --- |
| `vllm_serve_no_kv_cache_memory_v0.25.1.log` | `VLLM_CPU_KVCACHE_SPACE=0` with `--max-model-len 2048` |
| `vllm_serve_max_model_len_rejected_v0.25.1.log` | `--max-model-len 999999` on a 2048-position model |
| `vllm_serve_unrecognized_argument_v0.25.1.log` | an argument that does not exist |

A test asserts each pattern still matches the payload it was read off, so an upstream
rewording fails a test rather than silently reclassifying every future failure as the
general kind.

One pattern — torch's `CUDA out of memory` — has no fixture, because the CPU backend
cannot produce it. It is marked as such in the source. Its consequence if wrong is a
mislabelled OOM, not a lost run.

The memory fixture is also worth keeping for a second reason: it reproduces the
"See root cause above" shape exactly. Its last twenty-five lines say only that engine core
initialization failed. A classifier reading a tail would learn nothing from it, which is
why the agent's separately-collected root causes are fed to the classifier alongside the
tail — and why there is a test asserting the tail alone yields `None`.

### 3. Unrecognized stays unrecognized

`classify_engine_output` returns `None` when nothing matches, and the caller supplies the
phase-appropriate general kind. There is deliberately no heuristic tier. The floor for
anything the orchestrator cannot place at all is `internal`, kept distinct from every
plausible neighbour precisely so that a rising `internal` count reads as a defect report
instead of inflating a real category.

### 4. The kind never replaces the message

Both are always written. The kind is a lens for counting; it necessarily discards the
detail an operator acts on — "engine ran out of memory" does not say how much it wanted.
A database check constraint requires a kind on any failed run, so no future code path can
fail without one.

That constraint is added `NOT VALID`. It binds every new and updated row and leaves history
alone: runs that failed before the column existed keep a NULL, which is the honest value.
Backfilling them would mean guessing a kind from old free text, which is the one thing this
ADR is written to prevent.

### 5. The agent classifies where it holds better evidence, over a header

The agent sees the whole vLLM log rather than the tail that fits in a response, and it
alone knows whether its own readiness or benchmark deadline expired — from the control
plane, a timeout and a crash are both "the agent said 409". So `ServerError` and
`BenchError` carry a kind set at their raise site, and the agent reports it in an
`X-Vllmbench-Failure-Kind` response header.

A header rather than a new wire field, and this is the load-bearing part of the decision.
The wire models use `extra="forbid"` and are guarded by `PROTOCOL_VERSION`, so a new field
means a protocol bump, which means the control plane refuses to run against every GPU host
until each is redeployed. Restructuring the `{"detail": ...}` error body would change what
an older control plane displays. A header is invisible to a client that does not read it
and simply absent from an agent that does not send it — so old and new on either side of
the boundary keep working, and a control plane talking to a stale agent is merely a little
less precise, falling back to the text.

Consequently nothing downstream *requires* the header, and a kind this build has never
heard of is ignored rather than trusted, falling back to the text. Both directions of
version skew are tested.

### 6. Reconnection is not retry

An agent that is briefly unreachable is retried, with bounded backoff, on exactly one call:
the host-facts probe that opens every run. Nothing is re-measured — that call happens before
any engine is started, so there is no result a second attempt could paper over. Once the
benchmark is running the window is gone: a lost connection then means the measurement is
lost, and the honest answer is a failed run.

Refused tokens and protocol mismatches are not retried at all. They will not fix themselves,
so retrying only delays the report by the length of the window.

The orchestrator additionally stands down for 30 seconds after a run fails with an
unreachable host. Without that, a host rebooting mid-sweep loses the whole remaining queue
in seconds — every one of those failures real and correctly recorded, but all of them the
same failure. This changes only how fast work is pulled; it never re-runs anything and
never rescues a run that has already failed.

## Alternatives rejected

**A native Postgres enum for `failure_kind`.** An unfamiliar kind — from a newer agent, or
a later build talking to this database — would fail the insert instead of being recorded.
Turning "I do not recognise this failure" into "the failure is lost" is the one outcome
worse than filing it under the wrong heading. Stored as text, as `synthetic_source` already
is, for the same reason.

**Requeueing a run whose host was unreachable.** Tempting, because such a run produced no
measurement and re-running it hides nothing. Rejected because it needs an attempt counter
to avoid livelocking against a host that is permanently down, and an attempt counter on a
run is the beginning of retry semantics that ROADMAP 0.4.0 deliberately declined. The
loop-level backoff gets most of the benefit — a sweep survives a reboot's worth of queue —
without a run ever executing more than once.

**Classifying `run.error` after the fact, in a query or a view.** This is the design that
looks cheapest and is worst. It would have to work on text that has already lost the phase,
the deadline and the untruncated log, and it would silently reinterpret history every time
a pattern changed. Classification happens at the moment of failure, from the evidence that
exists then, and the answer is stored.

## Consequences

- Failure counts by kind become a query, which is what makes a sweep's failures triageable
  without opening eleven runs.
- The mock agent gains failure injection, per app instance, so every classified path is
  testable without a GPU — an engine cannot run out of memory on a laptop. CLAUDE.md already
  names failure injection as part of what the mock is for.
- Two `FailureKind` enums exist: one in `protocol` (which must stay installable into a
  user's vLLM environment and cannot drag the database package onto a GPU host) and one in
  `db` (which owns what can be stored). A tier-1 test asserts they are identical, because
  the duplication is otherwise a slow-motion bug — a kind added on one side only would fall
  back to a default and never appear in a query result, with no error anywhere.
- Timeout budgets move from hard-coded wire defaults onto `gpu_host` (defaults) and `sweep`
  (overrides, null meaning "no opinion" so raising a host's limit raises the sweeps that
  never had one). The values are unchanged; only their reachability is new.
