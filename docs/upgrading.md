# Upgrading

Two things move independently and must not drift: the control-plane stack on the control
host, and the agent on each GPU host. They version in lockstep and the protocol handshake
refuses a mismatch, naming both versions — a stale agent fails at connect time rather than
producing subtly wrong data mid-sweep. That refusal is the system working. This document is
how to avoid triggering it.

## When an agent upgrade is mandatory

Every release moves `__version__`, which is provenance. Only some move
`PROTOCOL_VERSION`, which is compatibility — and those are the ones where a stale agent is
refused rather than merely out of date. Upgrading the control plane past one of these
without upgrading the agents takes every GPU host offline until you do.

| Release | Protocol | What moved it |
| --- | --- | --- |
| 1.0.0rc1 – rc3 | 5 | — |
| **1.0.0rc4** | **6** | The agent reports whether its virtualenv satisfies its own constraints. |
| **1.0.0rc5** | **7** | Runs record what the engine resolved for speculative decoding, and what data they were measured against. |
| 1.0.0rc6 | 7 | — publishing only; **no agent upgrade required** |
| 1.0.0 | 7 | — **no agent upgrade required**; an rc6 agent serves a 1.0.0 control plane |

rc6 is the first candidate since rc3 that leaves the protocol alone. A GPU host running
the rc5 agent keeps working against an rc6 control plane, and its runs record
`agent_version: 1.0.0rc5` — which is the versioning design behaving as intended rather
than a gap: `__version__` is provenance and states what actually produced a measurement,
while `PROTOCOL_VERSION` is compatibility and is the only thing entitled to refuse. Upgrade
the agents when convenient; nothing is blocked until the protocol moves again.

Additive fields on the wire are still a protocol bump, because the wire schema forbids
unknown keys: a newer agent talking to an older control plane has its whole payload
rejected on the extra key. Being told the versions differ is better than losing a
forty-minute benchmark to a mismatch nobody named.

---

## The order

1. **Let in-flight work finish, or cancel it.** A run is claimed by the orchestrator and
   executed to completion; restarting the stack mid-run abandons it. Sweeps survive an API
   restart by design, but an upgrade changes the code underneath them.
2. **Back up the database.** Migrations are forward-only. There is no `downgrade`, and the
   only way back past one is a restore.

   ```bash
   docker compose exec -T postgres pg_dump -U vllmbench -Fc vllmbench > vllmbench-$(date +%F).dump
   ```

3. **Upgrade the control plane** by moving the image tag. The `migrate` service runs to
   completion before the API and orchestrator start, so this applies the schema and
   brings the services up in one command.

   ```bash
   git pull                        # picks up the new default pin in compose.yaml
   docker compose pull
   docker compose up -d --wait
   ```

   **Moving that tag is performing a migration.** `migrate` runs from the same pinned
   image as everything else, so the release you pin decides which schema your database is
   brought to — and forward-only means step 2's backup is the only way back. This is the
   one command in this document that changes data rather than code.

   If `.env` sets `VLLMBENCH_VERSION`, that wins over the default in `compose.yaml` and
   `git pull` will not move it. Edit it there instead, which is also how you pin a
   deployment to a release and leave it there.

   To upgrade a source checkout you are working on rather than a pinned release, use
   `make up`, which builds from `compose.build.yaml`.

4. **Upgrade every agent**, with `--reinstall-package` for each of `vllmbench-agent` and
   `vllmbench-protocol`. Without a reinstall flag `uv` does nothing when the version is
   unchanged; with the bare `--reinstall` it reinstalls the whole resolution, including
   the vLLM environment's own packages. See
   [agent-installation.md](agent-installation.md#6-upgrade).

5. **Verify both ends.**

   ```bash
   curl -s localhost:8000/api/health            # status ok, schema.ok true
   curl -X POST localhost:8000/api/hosts/<id>/refresh   # re-reads host facts, re-handshakes
   ```

## What a migration does to existing results

Nothing. Since 0.9.0 the schema is stable in a specific sense (ADR 0007): no column holding
a measurement is renamed, retyped or removed; additions are nullable or carry a server-side
default; constraints tightened on existing tables arrive `NOT VALID`, so history keeps its
honest values; and semantics never change under a stable name.

CI enforces it rather than trusting it. Every pull request migrates a database seeded with
a run, its summary and its per-device telemetry, then asserts every seeded **value**
survived — not row counts, which would miss a migration that rewrote a throughput figure or
dropped one device's telemetry while keeping its peer.

Verified again on the way to this release: a deployment two revisions behind, holding 83
runs, 56 summaries and 7,569 GPU samples, was upgraded with `docker compose up -d --build`
— the build-from-source path, which is what the documented upgrade was at the time. Both
migrations applied and all three counts were unchanged. The mechanism did not change when
the stack moved to pinned images: `migrate` still runs to completion before anything else
starts, and still applies the same Alembic revisions. What changed is which artifact it
runs from.

## Rolling back

**Code rolls back. Schema does not.**

Pinning an older tag leaves the database at the newer revision. The API
notices — it compares the applied revision against the head its build ships and reports
`status: degraded` with both revisions on `/api/health` — because the failure it prevents is
otherwise diagnosed by reading a Postgres type error backwards. That state is not supported,
even when it appears to work.

To actually go back past a migration, restore the dump you took in step 2. Pin the old
release first, so that when the stack comes back up its `migrate` service has nothing to
apply:

```bash
export VLLMBENCH_VERSION=<old-version>   # or set it in .env, which survives the shell
docker compose down
docker volume rm vllmbench_pgdata
docker compose up -d postgres
docker compose exec -T postgres pg_restore -U vllmbench -d vllmbench --clean --if-exists < your.dump
docker compose up -d
```

A `pg_dump -Fc` carries the `alembic_version` table with it, so the restored database comes
back at the revision it was taken at, not at the current head. Verified on the way to this
release: a dump of 83 runs and 7,569 GPU samples restored to exactly those counts and to
its own recorded revision.

Anything measured after the dump is gone. That is the cost of the forward-only rule, and it
is why the backup is step 2 rather than an afterthought.

## vLLM on the GPU host

The agent does not manage vLLM, and upgrading vLLM is a separate decision with separate
consequences.

`VLLM_REFERENCE_VERSION` at the repo root pins the version CI tests against. A GPU host
running something else is **recorded and warned about, never blocked** — benchmarking one
vLLM version against another is a legitimate, arguably headline, use of this tool. Blocking
would remove a feature.

But it changes what you are measuring. `vllm_version` is first-class provenance: charts
group by it and never silently overlay results across versions. After a vLLM upgrade, your
existing results and your new ones are two populations, and the framework will treat them
that way. If you want them comparable, re-measure the baseline.

An upgrade can also move the upstream contract. That is what tier 2 exists for: it re-derives
the `--save-result` field names from a live server, so a rename fails a test instead of
corrupting a sweep. If you upgrade vLLM ahead of `VLLM_REFERENCE_VERSION`, you are ahead of
that check.

## When it goes wrong

| Symptom | Cause |
| --- | --- |
| `/api/health` reports `degraded`, schema `applied` ≠ `expected` | The migrate service did not run, or the code was rolled back under a newer schema. |
| Writes fail with a Postgres type error | The same thing, undiagnosed. Check `/api/health` first. |
| A host refuses to connect, naming two protocol versions | Agent and control plane are on different releases. Upgrade the agent. |
| The agent still reports the old version after an upgrade | `uv pip install` without `--reinstall` is a no-op when the version string has not changed. |
| A host's vLLM version changed and old runs vanished from a chart | They did not; they are in a different provenance group. Change the comparison set. |
