# ADR 0007 — The schema is stable from here

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

Migrations have been forward-only since 0.1.0, but "forward-only" was a rule about *how*
to change the schema, not a statement that it had stopped changing. Through 0.8.0 it
changed a lot, and it should have: each milestone discovered something the previous one
could not have known — that `synthetic_source` had to be text rather than an enum, that
`kv_cache_usage_fraction` is a fraction and not a percent, that `sweep_seq` has to be
explicit because timestamp ties are not an ordering.

0.9.0 is where that stops. The remaining work is verification and documentation against a
fixed target, and 0.10.0 is where a change becomes a candidate for release rather than a
step toward one.

Declaring stability is worth an ADR because it is a promise, and a promise nobody wrote
down is one nobody can hold you to.

## Decision

### What "stable" means

Not "will never change". It means:

- **No column that holds a measurement is renamed, retyped or removed.** Anything in
  `run`, `run_summary`, `engine_sample` or `gpu_sample` that a chart or an export reads is
  fixed. A rename is add-column, backfill, drop-later — and pre-1.0 the drop-later is
  simply not scheduled.
- **Additions stay additive.** A new column is nullable, or NOT NULL with a server-side
  default. Never a NOT NULL with no default against a table that holds results.
- **Constraints tightened on existing tables arrive `NOT VALID`,** binding new rows and
  leaving history its honest values. `ck_run_failed_run_names_its_failure` is the worked
  example: backfilling it would have meant guessing a failure kind from old free text,
  which is exactly the inference the classification is written to avoid.
- **Semantics do not change under a stable name.** A column that meant a fraction keeps
  meaning a fraction. This is the one that actually corrupts results, because nothing
  errors: last month's rows and this month's rows both parse, and the chart quietly draws
  two different quantities as one series.

### What enforces it

Three things, none of which is anyone's memory.

`alembic check` (already in CI) fails when a model has drifted from its migrations. It
catches the schema change nobody wrote a migration for.

**Migrations now run against seeded data, not just an empty database.** This is the new
part, and it is the one that matters: an empty-database test cannot detect a NOT NULL
column with no default, a type change that fails on real values, or a constraint no
historical row satisfies. Every one of those passes against nothing and fails against a
month of results. CI now checks out the deployed schema, migrates a database with *that*
code, seeds it with a run, its summary and per-device telemetry, applies the branch's
migrations on top, and asserts every seeded **value** — not row count — survived. A
count-based check would miss a migration that rewrote a throughput figure or dropped one
device's telemetry while keeping its peer, and those are precisely the silent corruptions
this repository exists to prevent.

The baseline is `main` rather than a tag. For a pull request, main *is* the schema that is
already deployed, and it requires no tags to exist — which matters, because none do yet.
The step reads identically against a milestone tag when tags are cut; only the ref
changes.

The seed is plain SQL with explicit column lists, deliberately not the ORM: it runs
against the old schema, where the branch's models do not exist. The explicit list is also
the artifact — it is the set of columns whose survival is being asserted, in a file where
a reviewer sees it change.

### What this does not freeze

Tables that hold no measurements — `saved_view`, `mcp_write_audit`, `config_justification`,
`run_telemetry_pruned` — are workspace, not results. They can change more freely, and the
distinction is the same one `retention.PROTECTED` draws for a different reason: some rows
are the measurement, and the rest support it.

## Consequences

- The seeded-migration job is a hard gate. A migration that cannot run against a real run
  and its telemetry does not merge.
- Renaming a measurement column is now a multi-release operation, deliberately. That cost
  is the point: it is paid by whoever wants the rename, not by whoever finds a chart
  drawing two different quantities as one series.
- `scripts/seed_previous_schema.sql` becomes the place where "which columns must survive"
  is argued about, and `scripts/verify_seeded_data.sql` where "with what values" is.
