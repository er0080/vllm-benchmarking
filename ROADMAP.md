# Roadmap to 1.0.0

Milestones are ordered so that each one ends with something demonstrably working, not with
a layer that only pays off later. The first real proof point is 0.2.0 — one config, one
workload, one number, in the database, on screen.

Versions are pre-1.0 and make no compatibility promises. The database schema settled at
0.9.0 and is forward-only from there (ADR 0007).

---

## 0.1.0 — Foundations

Scaffolding only. Nothing measures anything yet.

- [x] Repository structure, `uv` workspaces, `ruff` and TypeScript strict configs
- [x] `compose.yaml` — postgres, api, orchestrator, web, migrate
- [x] Initial schema and Alembic setup: `gpu_host`, `model`, `server_config`, `workload`,
      `sweep`, `run`, `run_summary`, `engine_sample`, `gpu_sample`
- [x] Agent package skeleton with `/health` and `/host-info` (per-GPU model, VRAM, device
      index, driver, CUDA, vLLM version)
- [x] Token auth and protocol-version handshake between control plane and agent
- [x] Control plane can register a GPU host and display its facts
- [x] **Mock agent** behind `--profile dev` — synthetic results and telemetry, so the
      control plane and UI are developable with no GPU host present
- [x] **Fake `vllm` shim** for agent lifecycle tests — deferred at first, then built,
      because the revisit condition this item named was met. The real CPU backend runs in
      CI in ~40s and validates the *actual* upstream contract, so it owns that job; the
      shim exists for the opposite one, producing on cue the failures a working server
      cannot be asked for — dying mid-load, hanging without ever becoming ready, ignoring
      SIGTERM, exiting zero without writing a result. It lives at
      `packages/agent/tests/fixtures/fake_vllm`.
- [x] `pre-commit` with ruff, pyright, tsc
- [x] CI tier 1 (lint, types, unit) and tier 2 (postgres, mock agent, real vLLM on the CPU
      backend, image builds)
- [x] Branch protection on `main` requiring green — every job CI runs gates the merge
- [x] `VLLM_REFERENCE_VERSION` pinned at the repo root
- [x] Verified on macOS/Colima `aarch64` locally, including the `vllm-openai-cpu`
      container; native Linux `x86_64` is covered by CI on ubuntu-latest
- [x] Colima resource minimums documented in the README prerequisites

**Done when:** `docker compose up` succeeds on both Linux and macOS/Colima, the UI shows
live facts read from a real GPU host through the agent, and the full CI suite passes
without any GPU.

**Status:** complete except for the real-GPU half of the done-when clause. Verified
end to end against the mock agent; the paths that need actual hardware are itemised in
[docs/hardware-verification.md](docs/hardware-verification.md) rather than assumed
working.

---

## 0.2.0 — Single run, end to end

The vertical slice. This is the milestone that proves the architecture.

- [x] Agent: launch `vllm serve --config <yaml>`, poll `/health` to true readiness,
      capture logs, tear down cleanly
- [x] Agent: run `vllm bench serve --save-result`, return parsed JSON
- [x] Agent: reap orphaned vLLM processes on startup
- [x] Control plane: flatten benchmark JSON into `run_summary`, retain raw `jsonb`
- [x] UI: define one server config and one workload, trigger a run, watch it progress,
      view the result

**Done when:** a single benchmark run is triggered from the browser and its TTFT, TPOT,
ITL, and throughput figures land in Postgres and render on a run detail page.

**Status:** complete, and verified end to end on real hardware — 2× RTX 3090, vLLM
0.25.1. A production 27B FP8 config at TP=2 ran verbatim: 100 requests, 0 failed, TTFT,
TPOT, ITL and per-GPU throughput in Postgres. Device attribution confirmed to come from
NVML rather than the config, VRAM confirmed to return to baseline, and orphan reaping
confirmed against a real four-process engine.

The GPU found five bugs that presented as healthy software, all of them differences
between the environment vLLM is installed in and the environment the agent handed it.
See [docs/hardware-verification.md](docs/hardware-verification.md).

---

## 0.3.0 — Telemetry

Turns "which config won" into "why it won."

- [x] Agent: sample vLLM `/metrics` during runs — KV cache utilization, prefix cache
      counters, running and waiting queue depth, preemptions
- [x] Agent: sample NVML **per device** during runs — SM utilization, memory, power,
      temperature, clocks — for every GPU participating in the run
- [x] Bounded, low-overhead sampling loops with configurable interval
- [x] Run detail page: telemetry timeline aligned to the benchmark window, with per-device
      series rather than a host-level average

**Done when:** a saturated run visibly shows KV cache pressure and a growing waiting queue
on the run detail timeline, and a tensor-parallel run shows each participating GPU as its
own series.

**Status:** complete, and demonstrated on real hardware. A deliberately saturated run
(`max-num-seqs: 32`, 256 concurrent) on 2× RTX 3090 produced a timeline whose numbers
cross-check exactly: running pinned at 32, waiting at 224 — precisely 256 − 32 — and KV
cache at 5%, which says that configuration was *admission*-limited rather than
memory-limited. Those have opposite fixes, and the summary row alone cannot tell them
apart.

The `/metrics` parser was written against payloads captured from a live engine, one idle
and one under 128-way concurrency. That is what surfaced the prefix collision between
`num_requests_waiting` and `num_requests_waiting_by_reason`, which would have recorded a
queue depth of zero while 64 requests were queued.

---

## 0.4.0 — Sweeps

- [x] Sweep authoring: matrix of server configs × workloads, with replicate count
- [x] `tensor_parallel_size` as a first-class sweep axis, with validation against the
      host's available device count
- [x] Orchestrator state machine: queue, execute, resume, cancel
- [x] Cache reset between runs (`/reset_*_cache`)
- [x] Server restart between server-config changes; reuse across workload-only changes
- [x] Live sweep progress in the UI, with mid-sweep cancellation
- [x] Sweep survives an API restart

**Done when:** a multi-hour sweep runs unattended, survives a control-plane restart, and
can be cancelled cleanly without leaving orphaned processes or VRAM held.

**Status:** complete. A sweep is materialized as runs at authoring time, so the plan is a
fact in the database rather than a belief held by a process — which is what makes progress
countable, resume free, and the whole thing independent of any one service staying up.

Verified against the mock: an 8-run matrix predicted 2 model loads and took exactly 2,
reusing the loaded engine for the other 6. Cancelling a 15-run sweep mid-flight cancelled
the 12 queued runs immediately, stopped the one benchmarking, and left the engine
`stopped` — recorded as cancelled rather than failed, because a sweep that was stopped on
purpose should not read as broken.

Retry is deliberately not implemented. A failed run is evidence, and silently re-running
it would hide a reproducible failure behind an eventual success; re-running is a new run,
linked by sweep, which is what `replicate_idx` and immutability already provide.

---

## 0.5.0 — Analysis

The payoff milestone. Load the `dataviz` skill before building these.

- [x] **Pareto frontier** — per-user output tok/s against per-GPU total tok/s, normalized
      by device count so tensor-parallel configurations are honestly comparable
- [x] Tensor-parallel scaling view: aggregate throughput and per-GPU efficiency against
      TP size, answering whether TP=N earns its extra devices. Curves are keyed by config
      family — the config text with its TP line normalized away — so a curve is never
      assembled from configurations that differ in anything else
- [x] Per-device utilization comparison to expose imbalance within a TP group — grouped
      bars, one per device, per run rather than per point, so a single execution that
      split badly is not averaged away by its replicates
- [x] Latency-versus-concurrency curves, median and p99 drawn together so the gap between
      them — the queueing signal a median alone hides — is the thing you see first
- [x] Throughput saturation curves against request rate or concurrency, sharing the load
      axis with the latency curves
- [x] Side-by-side comparison with a diff of the exact config text. The one view allowed
      to cross a comparability boundary — charts must never silently overlay two vLLM
      versions, but a side-by-side the reader named both sides of is where that
      comparison is the subject, and every difference is listed back
- [x] Replicate spread rendered, not averaged away — median with a min/max cross, and
      each point states whether its band measures back-to-back repeatability, run-to-run
      variance, or drift between separate sittings
- [x] Provenance guards that refuse to silently overlay incomparable runs — partitioning
      happens in the API, so a view is never handed two vLLM versions in one series
- [x] Filtering — population, host, sweep and tensor-parallel size, held above the views
      rather than inside each, so narrowing on one tab still means the same thing on the
      next
- [x] Saved views — a stored *query* (which chart, which runs, which axes), never a set
      of run ids, so reopening one includes everything measured since

**Done when:** a tuning decision can be made from the UI alone, without exporting to a
notebook.

---

## 0.6.0 — Agent interface (MCP)

Per [ADR 0001](docs/adr/0001-mcp-server-interface.md). Placed here because the valuable
analysis tools depend on 0.5.0 and the control tools on 0.4.0.

- [x] Streamable HTTP MCP server mounted at `/mcp` on `api`, behind `VLLMBENCH_MCP_ENABLED`
- [x] Bearer token auth — an unset token refuses every caller rather than accepting all,
      so enabling MCP without configuring one yields an inert surface, not an open one
- [x] Config **validation engine** — checks a candidate YAML against the target vLLM
      version's accepted arguments and returns structured, actionable errors. Pulled
      forward from 0.7.0: `validate_config` needs it, and 0.7.0's YAML editor is a UI over
      this engine rather than a separate implementation. The argument catalogue is
      *captured from a real parser*, never authored, and tier 2 re-derives it from the
      live container so an upstream rename fails a test instead of a sweep. No agent
      endpoint was needed: the host already reports its vLLM version, and a version with
      no capture falls back to the reference and says so.
- [x] Read tools — `list_hosts`, `list_configs` / `get_config`, `list_workloads`,
      `list_sweeps` / `get_sweep`, `query_runs`, `get_run`, `get_run_telemetry`,
      `compare_runs`, `get_pareto`, `server_info`. Each calls the same function the HTTP
      route calls, so the two surfaces cannot drift
- [x] `validate_config` — a read tool, so it stays available on a control plane with
      writes switched off, which is where an agent most needs to check a config it
      cannot create
- [x] Write tools, enabled by default and separately switchable — `create_config`,
      `create_workload`, `create_sweep`, `cancel_sweep`. No tool mutates or deletes a run,
      and a test asserts none is ever added
- [x] MCP resources: `vllmbench://config/{hash}`, `vllmbench://sweep/{id}/report` — both
      immutable, so a client may cache them by URI: a config's hash *is* its text, and a
      finished sweep's runs cannot change
- [x] Context economy: pagination with a hard maximum, summary fields by default,
      server-side telemetry downsampling — thinned per device in the query, because
      striding the interleaved series drops whole GPUs while looking complete
- [x] `initiated_by` provenance on sweeps and runs (`ui` / `mcp` / `api`) plus client
      identity — declared by the caller, since HTTP cannot tell a browser from curl and
      guessing would put a confident wrong answer in a provenance column
- [x] `get_sweep` returns `estimated_remaining`, extrapolated from completed points
- [x] Guardrails: one active sweep per host enforced in the domain layer (two sweeps
      interleave engine restarts, so neither measures what it planned), bounded matrix
      size, exact run count and engine-load count returned at authoring
- [x] Structured duration estimate — benchmark time and engine-load time separated,
      because a plan with three config changes left costs minutes more than one with
      three runs of the same config, and no number at all when nothing has finished
- [x] Audit log of write calls, including the refused ones — a refusal writes nothing to
      any other table, so without it the only record is the agent's own context

**Done when:** an agent connected over MCP can author a config, define and start a sweep,
poll it to completion, and read back a per-GPU normalized Pareto frontier — without
touching the UI.

---

## 0.7.0 — Configuration management

- [x] YAML editor surfacing the validation engine from 0.6.0 — one implementation, two
      interfaces. A plain textarea over the exact bytes, not a form of typed fields: a
      form would hold opinions about vLLM's option set that rot every release, and would
      silently drop any setting it did not know about
- [x] Content-addressed config storage and lineage (which config was derived from which).
      Lineage records how a config *first* came to exist — content addressing means
      re-submitting existing text returns the existing row, and letting a later submission
      reparent it would make the chain depend on who submitted last
- [x] Export a config for direct use with `vllm serve --config` — byte for byte, with no
      provenance header added. A header would not change what vLLM does but *would* change
      the hash, and then the file in production is not the file that was measured
- [x] Import an existing YAML, checked on arrival and flagged when it is byte-identical to
      a config that already exists
- [x] Annotate a config with the result that justified it. The run must be one that
      actually used the config: a citation that can point at the wrong evidence is worse
      than no citation

**Done when:** the winning configuration from a sweep can be exported and run in
production unchanged.

---

## 0.8.0 — Interop

- [x] Importer for upstream `vllm bench sweep serve` output directories — the files
      carry no provenance whatsoever, so the operator declares it and every imported run
      is marked permanently (ADR 0003). Written against a captured directory, checked in
      as a fixture
- [x] CSV and JSON export of any result set — same filters as the charts, so the file
      matches the screen, and every row carries its own provenance and population because
      a file is read by people who cannot see the filters that produced it
- [x] Shareable run and sweep reports — the same markdown the MCP resource serves, so
      the two cannot come to disagree about what a sweep measured
- [x] Standard work documentation: repeatable benchmark procedure in
      [docs/benchmark-procedure.md](docs/benchmark-procedure.md), written from the first
      real sweep

**Done when:** results produced outside this framework can be loaded into it, and results
produced inside it can be handed to someone who does not run it.

---

## 0.9.0 — Hardening

Still inside the relaxed-process window, deliberately: this milestone defines the
behavior of every failure mode and settles the schema, which is design work done by
discovering what breaks. Making each discovery cost an issue first is what that phase is
least able to afford. Rationale still lands in PR bodies and `docs/adr/` as it has all
along — see CLAUDE.md.

- [x] Schema stabilized; forward-only migrations from here (ADR 0007). Not "never
      changes" — no measurement column is renamed, retyped or removed; additions stay
      additive; constraints tightened on existing tables arrive `NOT VALID` so history
      keeps its honest values; and semantics never change under a stable name, which is
      the one that actually corrupts results because nothing errors
- [x] Failure handling: agent unreachable, vLLM OOM, model load failure, benchmark timeout.
      Each is recorded under a `failure_kind` rather than only as free text, so a sweep's
      failures are countable instead of eleven walls of traceback to read one at a time.
      Classified from structural evidence where possible — the phase a run was in is
      knowable at the call site and nowhere later — and from vLLM's own words only where
      nothing else can distinguish out-of-memory from a rejected config. Those patterns
      come from output captured off a real vLLM, and an unrecognized failure stays at the
      general kind rather than being guessed into a specific one (ADR 0004)
- [x] Disk and retention management — sized from measurement rather than guesswork.
      Telemetry is ~97% of what a run occupies and scales with duration and device count,
      so it is the only thing retention deletes; the measurement, its summary and the raw
      payload are protected by name, checked at runtime as well as by test, because
      `raw_result` is the only way to fix a flattening mistake without re-running weeks of
      GPU time. A pruned run is recorded as pruned — an empty timeline otherwise reads as
      a sampling bug rather than as the policy working. Off by default: bounding growth is
      the point, choosing the bound belongs to whoever owns the results. Agent-side, temp
      directories left by a SIGKILL are swept at startup and disk headroom is checked
      before an engine or a benchmark starts. Docker log rotation caps each service at
      30 MB
- [x] Agent restart and reconnection semantics — the host-facts probe that opens every run
      tolerates a briefly absent agent, with bounded backoff. Reconnection, not retry:
      nothing is re-measured, because that probe runs before any engine starts. Once the
      benchmark is running the window is gone, since a lost connection then means a lost
      measurement. The orchestrator also stands down after an unreachable host, so a
      reboot mid-sweep costs a pause rather than the whole remaining queue
- [x] Structured logging, no secrets in logs — JSON lines carrying bound context, so a
      run's whole trail across three services is a filter rather than a grep. Redaction is
      by registered *value*, never by pattern: pattern matching cannot know what a
      deployment's token looks like, and the leaks that actually happen are in text nobody
      wrote — a Postgres failure quotes the DSN back, password included, from inside the
      driver. Scrubbing runs in the formatter so it covers lazy `%s` arguments and
      tracebacks, and uvicorn is unhooked onto the same handler so its access log cannot
      route around it
- [x] Test coverage on the JSON-to-column flattening layer — and it found something. A
      benchmark where every request fails exits 0, writes a result file, and reports 0.00
      for every metric; on a latency axis 0 ms is the *best* value, so a run that measured
      nothing would have rendered as the fastest configuration ever tested and sat on the
      Pareto frontier. Captured from a real vLLM and now refused: zero completions is the
      absence of a measurement, not a measurement of zero. Partial failures are kept and
      warned about instead, since throughput divided by the whole duration understates
      them rather than inventing them. Also pinned: the flattener's output and
      `run_summary`'s columns correspond in both directions — the direction that fails
      silently is a column nothing ever fills, NULL on every row forever with no error
      anywhere
- [x] Migration CI: applied against an empty database and against one seeded with a run,
      its result and its per-device telemetry, then checked value by value. Row counts
      would miss a migration that rewrote a throughput figure or dropped one device's
      telemetry while keeping its peer. The baseline is `main` rather than a tag, since
      for a pull request main is the schema already deployed and no tags exist yet

**Done when:** every failure mode identified during 0.2–0.8 has a defined behavior and a
test.

**Status:** complete. The milestone did what it was placed inside the relaxed-process
window to do — it found things by breaking them.

The largest was in the flattening layer, and it was live: `vllm bench serve` exits 0 when
every request fails, writes a result file, and reports 0.00 for every metric. Nothing
upstream noticed, because nothing was wrong — the process succeeded, the file existed,
every required field was present. The zeros were the problem: on a latency axis 0 ms is
the best value there is, so a run that measured nothing would have appeared as the fastest
configuration ever tested. It is now refused, against a payload captured from a real vLLM
rather than an imagined one.

The rest of the milestone is the same shape. Failures were free text and are now countable
without becoming guesses (ADR 0004). Logs were per-service formats with redaction left to
care at each call site, and are now one queryable stream with redaction by registered value
(ADR 0005) — because the leaks that happen are in text nobody wrote, like a Postgres
failure quoting the DSN back from inside the driver. Growth was unbounded and unmeasured;
it is now measured, and the only thing retention may delete is telemetry (ADR 0006).
Migrations were tested against an empty database, which cannot detect the class of bug
forward-only exists to prevent, and now run against seeded data checked value by value
(ADR 0007).

---

## 0.10.0 — Release candidate

Process rigor steps up here: from this milestone every PR traces to a labeled, milestoned
issue, cross-linked with closing keywords. The schema is stable, the failure modes have
defined behavior, and what remains is verification against a fixed target — so a change is
now a candidate for release rather than a step toward one. See CLAUDE.md.

- [x] Issue-driven change management in effect — every PR in this milestone traces to a
      labeled, milestoned issue, cross-linked with closing keywords
- [x] Quick start verified from a clean control host and a clean GPU host — run from a
      fresh clone into an empty Docker state, and from a GPU host the agent had been
      uninstalled from, ending in a real benchmark
- [x] Agent installation guide — [docs/agent-installation.md](docs/agent-installation.md)
- [x] Tuning playbook: how to read the charts and what to change next —
      [docs/tuning-playbook.md](docs/tuning-playbook.md), written off the sweep already
      recorded rather than from general advice
- [x] Upgrade path documented — [docs/upgrading.md](docs/upgrading.md), verified by
      upgrading a deployment two revisions behind that held 83 real runs
- [x] Known limitations stated plainly — [docs/limitations.md](docs/limitations.md)

**Done when:** somebody who has never seen this repository can stand the stack up, take a
measurement, and know what it does not do.

**Status:** complete. Running the quick start against nothing is what made it a
verification rather than a proofread: it is the only condition under which a missing step
is visible. Two things were found that way — an install check whose success was silence,
and an upgrade instruction that was a no-op because `uv` sees an unchanged version string
and does nothing.

Reading the documentation as a set found a third, and it was not a documentation bug: the
integration suite defaulted to the database the compose stack keeps results in, and it
empties every table it is pointed at. `make test-integration` on a developer's laptop
destroyed their recorded runs. Fixed at the default.

### Release candidates

Tags cut against this milestone, and what each one found. A release candidate that finds
nothing was not tested.

- **v1.0.0rc1** — first tag. Everything through 0.10.0.
- **v1.0.0rc2** — the install, followed literally on a clean machine rather than from
  memory. The one that mattered: nginx resolved the api container's address once and
  cached it forever, so recreating that container left the UI returning 502 from a page
  that still rendered — and the documented upgrade does exactly that recreate. A first
  `up` works, which is why it reached a release candidate at all. Also: a required
  environment variable that nothing read, a stack that started clean holding `change-me`,
  an "unreachable" message that ended in a bare colon, and a quick start whose agent
  command blocked the terminal it then told you to use. CI now runs the documented install
  and moves the api container underneath it.
- **v1.0.0rc3** — the MCP surface, driven by an agent that had never read the codebase
  rather than reviewed by one that had. The one that mattered: `get_pareto` replaced an
  unrecognised metric key with the default and reported success, so an agent asking for a
  latency frontier got a throughput frontier with nothing saying so. `mean_ttft_ms` is the
  input that makes it real — not a typo, just not a key here, and the exact shape of a
  guess. Refused at the MCP boundary now; the HTTP endpoint keeps the fallback, because a
  person reads the axis label and an agent has none. Also 45 parameters with no
  descriptions, closed sets published as bare strings, and no read-only or destructive
  hints — all of which matter because for an MCP server the schema *is* the documentation,
  there being no README an agent reads first. Two readings of the surface were wrong in
  exactly the way a user's would have been: `resources/list` is empty because both
  resources are templated, and the working feature was reported missing. There is a guide
  now, and a tier-1 test that fails if the guide and the surface disagree.
- **v1.0.0rc4** — the two issues left open against this milestone, plus two the MCP surface
  turned up while being asked a question. **The first candidate to move
  `PROTOCOL_VERSION`, so every GPU host's agent must be upgraded alongside the control
  plane** — which makes it the first real exercise of a refusal the design has always
  promised. The repository had no licence, so the quick start was inviting people to do
  something exclusive copyright did not permit; it is Apache-2.0, matching vLLM. The agent
  could leave vLLM's own virtualenv outside vLLM's declared constraints with nothing saying
  so: it now checks, warns at startup, and records the answer as run provenance, never
  blocking, because refusing to start would adjudicate someone's virtualenv from another
  machine. And two MCP tools described themselves wrongly — `validate_config` advertised
  that it crossed the host boundary when it reads a database row, and `cancel_sweep`
  promised an interruption the orchestrator performs three seconds later, so an agent that
  cancelled and polled immediately had been told that meant failure.

- **v1.0.0rc5** — three defects found by *using* the framework rather than reading it, all
  of them at a seam between layers that individually worked and were individually tested.
  A sweep chose a serving configuration for a notebook coding assistant — four MTP depths,
  two workloads, 24 runs — and produced its answer (+75% per-user throughput on editing at
  depth 3) along with the discovery that the framework could not say which of those runs
  had been speculating. **The second candidate to move `PROTOCOL_VERSION`**, so every GPU
  host's agent must be upgraded alongside the control plane again.

  Cache resets between sweep points had never happened. `/reset_prefix_cache` sits behind
  `VLLM_SERVER_DEV_MODE`, the agent did not set it, and the supervisor read the resulting
  404 as a version that lacked the endpoint — so every sweep this project ever ran carried
  its prefix cache across every point, and said `reset caches: none available` while doing
  it. It cost nothing so far, and the database is what says so rather than hope: 8091
  prefix-cache queries across the MTP sweep and zero hits, because those prompts share
  nothing. A sweep over a repeated preamble would have had its later points inflated by its
  earlier ones and the *ordering* of the matrix would have picked the winner. Storing
  counters rather than a sampled hit rate is what made that answerable at all.

  Speculation is now provenance — method and depth, indexed, read from the engine's own
  `/server_info` rather than parsed out of config text, on the same rule invariant 8 sets
  for parallelism topology and for the same reason: a YAML saying `num_speculative_tokens:
  3` is not proof the engine drafted three tokens. Three states stay distinct, and the
  third is the one that matters: `"none"` is the engine saying it is not speculating, NULL
  is nobody having asked it.

  Inter-token latency turned out to be measuring a different quantity on either side of
  that boundary — the wait between *emissions*, not between tokens — rising 25.0 → 44.1 ms
  across the same arms whose throughput rose 39.6 → 69.3 tok/s. Both correct. Kept and
  relabelled *emission gap* rather than dropped, because chunked delivery is a real
  property of a streaming UI; a group mixing speculation settings now says so.

  `run.dataset_identity` had been NULL on every run this project ever produced, which
  invariant 6 has required since the first schema. The agent now hashes what it actually
  read, because `--dataset-path` names a file on a host the control plane cannot see.

  And tier 2 stopped taking the speculative field names on trust: it runs a second,
  speculating engine and re-derives them from it. That needed two things measured rather
  than assumed — ngram needs no draft model, so it runs on a CPU backend; and an engine
  that speculates but never drafts emits none of those fields at all, which is a fact worth
  knowing on its own, because it means a NULL acceptance rate is not evidence that
  speculation was off.

- **v1.0.0rc6** — the first candidate that publishes anything, and the first since rc3
  that leaves `PROTOCOL_VERSION` alone, so no GPU host has to be touched to accept it.
  Everything before this was built from source by whoever ran it, which meant architecture
  was a property of the builder and could not be got wrong; from here it is a property of
  the artifact, and getting it wrong is silent — an amd64-only image starts fine under
  Colima, answers every health check, and is merely unusably slow.

  Building it found that the artifact the roadmap describes could not have been installed.
  "Attaches the agent wheel", singular, but the agent requires `vllmbench-protocol`, which
  is published nowhere; a Release carrying one wheel attaches something nobody can install.
  Both go up now. The same fact turned out to be a supply-chain hazard rather than an
  inconvenience: both names are unregistered on PyPI, so an unpinned requirement inside a
  wheel handed to people is an instruction to fetch a name that belongs to whoever
  registers it first, landing a stranger's code in the vLLM virtualenv on a GPU host. The
  install names both files so the lookup never happens, the agent pins the exact version
  behind that, and `scripts/check_versions.py` holds the pin to `VERSION`.

  The publish itself then went clean on the first attempt, which is the unusual outcome
  here and worth stating plainly rather than quietly. All ten builds pushed by digest,
  `imagetools create` merged each pair, and every image reported both platforms. A Mac
  pulling `api:1.0.0rc6` resolved to `linux/arm64` off the manifest list unprompted and
  ran; the wheels downloaded from the Release installed into an empty virtualenv with
  nothing pointing at a checkout. `latest` did not move, in either sense — no GHCR
  `:latest` tag and no "Latest" release on the repository page — which is what a
  pre-release should do and is only knowable by asking.

  One expectation was wrong in the harmless direction. A GHCR package created by
  `GITHUB_TOKEN` was predicted to default to private, requiring a manual visibility flip
  per package before anyone outside could pull; all five came out public and the check
  passed unattended. The check stays regardless, and its job changed rather than ended: it
  was written as a reminder for a step someone had to remember, and it is now a regression
  guard on a setting that has no API, no test of its own, and no symptom visible to anyone
  holding credentials.

  So what this candidate found, it found during construction rather than at the tag — which
  is the honest reading of a green release run, not evidence that a release needs no
  candidate.

---

## 1.0.0 — Release

First milestone with published artifacts. Before this, everything is built from source and
the agent is installed from a git tag.

- [x] Release workflow: `v*` tag publishes multi-arch (`linux/amd64`, `linux/arm64`)
      control-plane images to GHCR and attaches the agent **wheels** to a GitHub Release —
      plural, because the agent requires `vllmbench-protocol` and a Release carrying one
      wheel attaches something nobody can install. Verified at `v1.0.0rc6`: five images,
      both platforms each, anonymously pullable; both wheels installing into an empty
      virtualenv on a machine with no source tree
- [x] Compose switches from local build to pinned published image tags — with a
      `compose.build.yaml` override so `make up` still runs a developer's own code, kept
      as a separate file because `image:` and `build:` on one service resolve differently
      depending on the local image cache. Verified on macOS/Colima against the real
      `v1.0.0rc6` images: arm64 pulled natively, schema unchanged, 69 existing runs intact
- [x] CHANGELOG generated from Conventional Commits — `scripts/generate_changelog.py`,
      checked in CI so the file cannot drift or be hand-edited, and read by the release
      workflow so the Release body and the changelog cannot disagree about what shipped

**Definition of done:**

1. A new user can go from an empty repository to a completed sweep with charted results by
   following the README alone.
2. A sweep of at least 24 points runs unattended to completion without manual intervention.
3. Every recorded run states its full provenance: vLLM version, agent version, GPU, driver,
   config hash, dataset identity, parallelism topology, and what initiated it.
4. No orphaned vLLM process or held VRAM after any normal or cancelled sweep.
5. The database schema is stable and migrations are forward-only.
6. A tuning decision can be made from the UI without external tooling.
7. The stack runs with equal fidelity on native Linux and on macOS with Colima.
8. A tensor-parallel sweep across TP sizes completes and charts per-GPU normalized
   throughput, with per-device telemetry available for every run in it.
9. An agent harness connected over MCP can complete a full tuning loop — author a config,
   run a sweep, read the results — without touching the UI.

**Released 2026-08-23.** All nine definition-of-done criteria met, and the last one that
needed a person rather than CI — equal fidelity on native Linux and macOS with Colima — was
confirmed against the published `v1.0.0rc6` images on both architectures rather than against
a local build.

The release candidates did what they were for. rc2 found an nginx cache that broke the UI on
the documented upgrade. rc3 found `get_pareto` silently substituting a metric an agent asked
for. rc4 found no licence and an agent that could not see its own environment drift. rc5
found that cache resets had never happened, that ITL was measuring emissions rather than
tokens, and that no run could say whether it had been speculating. rc6 found an agent wheel
that could not have been installed, and the dependency-confusion hazard behind it.

None of those were found by testing. They were found by using the thing, and then by
publishing it.

---

## 1.1.0 — Interconnect provenance

Opened by wanting to measure something the framework could not describe.

`ubuntu-llm`'s two 3090s sit on CPU root ports with peer-to-peer DMA refused by the driver,
and there is a community patch that enables it. Whether that patch is worth having is a
straightforward A/B question. Recording the answer was not: the patch is a rebuild of the
*same driver version*, so a run measured over a direct GPU-to-GPU path and one staging
through host memory agree on driver version, CUDA version, GPU model, vLLM version,
parallelism topology and device indices — every provenance field a run had. The deployment
already held 69 runs, all TP=2, all measured without P2P, and the boundary existed only in
somebody's memory of which week it was.

- [x] `peer_access` on the wire and on the run, observed over the devices a run actually
      used rather than the host's full complement, so a TP=1 control stays one series
      across a change only a multi-device run can feel (#117, protocol 8)
- [ ] Which driver *build* produced that state, for the case where two builds agree on
      peer access and differ in other ways (#119)
- [ ] The A/B itself, and whatever it says

The first of those found a live upstream typo — `nvidia-ml-py` 13.610.43 defines
`NVML_P2P_CAPS_INDEX_READ` as `(0,)` — which would have made the field uniformly blank
while looking implemented. Caught by capturing a payload from the real host before writing
the probe. The second was implemented as `srcversion` and removed before merge when a real
patched build reported a hash identical to the stock module's; what was tried is recorded
in #119 so it is not tried again.

---

## Explicitly out of scope for 1.0.0

Deferred, not rejected. Each is a post-1.0 candidate.

- Multi-node deployments of any kind, including tensor or pipeline parallelism spanning
  hosts. Single-host multi-GPU is **in scope** — see 0.3.0 and 0.5.0.
- Multiple concurrent GPU hosts, or a job queue across hosts
- Benchmarking from the control plane over the network
- Non-NVIDIA accelerators
- `vllm bench throughput` (offline) and `vllm bench mm-processor` — online serving is the
  focus
- Authentication beyond a shared agent token; multi-user access control
- Automated tuning or search strategies (Bayesian optimization, early stopping). The
  framework measures and displays; the human decides.
- Managing vLLM installation on the GPU host
- CI integration and regression gating
