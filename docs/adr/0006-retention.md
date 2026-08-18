# ADR 0006 — Retention deletes telemetry, and nothing else

- **Status:** accepted
- **Date:** 2026-08-18
- **Milestone:** 0.9.0 — Hardening
- **Issue:** none — issues were optional at this milestone; see CLAUDE.md
- **Supersedes:** nothing

> Accepted. The decisions below are binding; changing one requires a superseding ADR, not
> an edit to this file. Rejected alternatives are retained deliberately — they are the
> record of why the project is not doing the obvious other thing.

---

## Context

Nothing in this system had a bound. Every run adds rows to `engine_sample` and
`gpu_sample` and they are never removed, so the only question was how fast — and nobody
had measured it.

Measured, on the dev database holding the first real sweeps:

| | |
| --- | --- |
| `gpu_sample` | 225 bytes/row, up to 240 rows per run on a two-GPU host |
| `engine_sample` | 218 bytes/row, up to 125 rows per run |
| `run` + `run_summary` | ~2 kB, once |
| `raw_result` | ~1.1 kB, once |

Telemetry is about **97% of what a run occupies**, and it scales with duration and device
count rather than with run count. A one-hour run on eight GPUs is roughly 7 MB; a month of
continuous benchmarking is a few gigabytes; a busy year is tens of gigabytes.

That is not an emergency. It is unbounded, which is a different problem: the person who
eventually needs a bound should not have to invent a policy under pressure, on a disk
that is already full, with a sweep failing to write results.

## Decision

### Telemetry is the only thing retention deletes

It is diagnostic. It explains *why* a configuration won, and that question has a shelf
life in a way the measurement does not. Everything else is either the result or the means
of recomputing it.

`vllmbench_db.retention.PROTECTED` names what may never be touched, with the reason for
each. It is consulted at runtime, not merely documented: `prune_telemetry` refuses to run
if the tables it prunes ever intersect it. A test catches that in CI; the runtime check
catches it in the deployment where someone shipped past the test.

**`raw_result` is protected**, despite being the obvious second candidate at ~1.1 kB per
run. CLAUDE.md: "If we later discover we flattened something wrong, the raw record lets us
recompute. Never discard the original." A kilobyte against the ability to fix a mistake
without re-running weeks of GPU time is not a trade worth making.

### Deletion does not contradict "time-series tables are append-only"

Append-only means never *rewritten in place*, and that is what protects the measurement: a
sample edited after the fact makes a chart lie about what was observed. A sample removed
under a stated horizon, recorded per run, cannot — the chart says "pruned" instead of
drawing something false. The database trigger enforces exactly this distinction: it
rejects UPDATE on those tables and says nothing about DELETE.

### A pruned run is recorded as pruned

`run_telemetry_pruned` is a leaf table carrying the run, when it was pruned, and the
horizon in force.

Without it, a run detail page with an empty timeline has two readings — the policy removed
it, or sampling has been failing silently — and they call for opposite responses. One is
the system working. The other is diagnostic data being lost, and it would look exactly the
same until someone spent an afternoon debugging a sampler that was fine. The API carries
it on the telemetry response and the UI says which.

A separate table rather than a column on `run` for two independent reasons: a terminal run
is immutable and a trigger enforces it, and this is data *about* a run rather than a
correction to what it measured.

### The default is off

`VLLMBENCH_TELEMETRY_RETENTION_DAYS=0` keeps everything.

A benchmarking framework that silently discarded diagnostic data after ninety days —
because ninety sounded reasonable — would be making a decision that belongs to whoever
owns the results. Bounding growth is the point; choosing the bound is not ours. What the
framework owes is the ability to see the number (`GET /api/storage`, measured from this
database rather than from the constants above) and to act on it without inventing
anything.

### The deleting route is harder to invoke than the creating ones

`POST /api/storage/prune` is a dry run unless `confirm=true`. The default answer to "what
would this remove" is a number, not a removal. It is the only route in the API whose
purpose is to destroy data and it should not be reachable by a slip.

It is deliberately **not** exposed over MCP. CLAUDE.md's MCP rule is that no tool mutates
or deletes a run; deleting its telemetry is close enough to that line that an agent should
not be on the other side of it unattended.

### Retention runs from the orchestrator's idle path

Once a day, and only when there is no queued work. A deletion competing with a benchmark
for the same database is exactly the kind of interference that surfaces later as an
unexplained latency outlier in somebody's results. A failure in the pass is logged and
otherwise ignored — the queue is the job; reclaiming disk is housekeeping.

### The GPU host's disk is the agent's problem

Two additions there, both following rules the agent already had.

Working directories are removed in a `finally`, which a SIGKILL skips. They are now swept
at startup on exactly the principle that governs the process reaper: the agent cleans up
after its previous lifetime because nothing else on the box knows those were ours. Only
directories matching our own prefixes, and only ones older than an hour — two agent
processes can briefly overlap during a restart, and deleting a directory the outgoing one
is still writing into would turn a tidy-up into the data loss it exists to prevent.

Free space is checked before starting an engine or a benchmark, and a shortfall is a
refusal rather than a warning. A disk with no room turns a forty-minute benchmark into a
write error at the end of it; a `statvfs` turns that into an immediate, actionable no. It
reports as `host_disk_full` — a kind of its own, because it is not a property of the
configuration at all: the same config on the same card works once space is freed, so
filing it under `engine_load_failed` would send the reader to tune something that is fine.

## Alternatives rejected

**A default horizon of 90 or 180 days.** Silently deleting the data that answers "why did
this configuration win" is not a default a tool gets to choose for its user.

**Aggregating old telemetry instead of deleting it** — downsampling a year-old series to
one point a minute. Attractive, and wrong here for the reason `gpu_sample` is keyed per
device: aggregation destroys the imbalance signal that makes a tensor-parallel run
diagnosable, and a stored average is indistinguishable from an observation. If it is worth
keeping, keep it; if it is not, say it was removed.

**Deleting whole old runs.** They are the results. A benchmarking tool that forgets its
own measurements has failed at its only job.

**A cron container.** One more service, one more thing to fail silently, for work that
takes seconds a day. The orchestrator is already a long-running process with a database
connection and an idle path.

## Consequences

- Growth is now a number an operator can read (`GET /api/storage`), measured from their own
  database rather than from a constant that would be wrong on a different host.
- Docker's json-file driver is capped at 10 MB × 3 per service. It is unbounded by default,
  and a control plane that filled the host's disk with its own logs would take the database
  down with it — first symptom, a sweep failing to write results.
- `PROTECTED` is now the place a future change to what may be deleted has to argue with.
