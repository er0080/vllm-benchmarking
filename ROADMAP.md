# Roadmap to 1.0.0

Milestones are ordered so that each one ends with something demonstrably working, not with
a layer that only pays off later. The first real proof point is 0.2.0 — one config, one
workload, one number, in the database, on screen.

Versions are pre-1.0 and make no compatibility promises. The database schema may change
destructively before 0.9.0.

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
- [ ] ~~**Fake `vllm` shim** for agent lifecycle tests~~ — superseded. The real vLLM CPU
      backend runs in CI in ~40s and validates the *actual* upstream contract, which a
      shim can only ever validate our assumptions about. Revisit only if a failure mode
      is needed that real vLLM cannot be made to produce on cue (crash during model
      load, for instance).
- [x] `pre-commit` with ruff, pyright, tsc
- [x] CI tier 1 (lint, types, unit) and tier 2 (postgres, mock agent, real vLLM on the CPU
      backend, image builds)
- [x] Branch protection on `main` requiring green — all eight CI jobs required
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

- [ ] Importer for upstream `vllm bench sweep serve` output directories
- [x] CSV and JSON export of any result set — same filters as the charts, so the file
      matches the screen, and every row carries its own provenance and population because
      a file is read by people who cannot see the filters that produced it
- [x] Shareable run and sweep reports — the same markdown the MCP resource serves, so
      the two cannot come to disagree about what a sweep measured
- [ ] Standard work documentation: repeatable benchmark procedure in `docs/`

**Done when:** results produced outside this framework can be loaded into it, and results
produced inside it can be handed to someone who does not run it.

---

## 0.9.0 — Hardening

Process rigor steps up here too: from this milestone every PR traces to a labeled,
milestoned issue. Through 0.8.0 issues are optional, favoring iteration speed while the
architecture is still moving.

- [ ] Issue-driven change management in effect — see CLAUDE.md
- [ ] Schema stabilized; forward-only migrations from here
- [ ] Failure handling: agent unreachable, vLLM OOM, model load failure, benchmark timeout
- [ ] Disk and retention management for raw results and logs
- [ ] Agent restart and reconnection semantics
- [ ] Structured logging, no secrets in logs
- [ ] Test coverage on the JSON-to-column flattening layer
- [ ] Migration CI: applied against an empty database and one seeded at the previous tag

**Done when:** every failure mode identified during 0.2–0.8 has a defined behavior and a
test.

---

## 0.10.0 — Release candidate

- [ ] Quick start verified from a clean control host and a clean GPU host
- [ ] Agent installation guide
- [ ] Tuning playbook: how to read the charts and what to change next
- [ ] Upgrade path documented
- [ ] Known limitations stated plainly

---

## 1.0.0 — Release

First milestone with published artifacts. Before this, everything is built from source and
the agent is installed from a git tag.

- [ ] Release workflow: `v*` tag publishes multi-arch (`linux/amd64`, `linux/arm64`)
      control-plane images to GHCR and attaches the agent wheel to a GitHub Release
- [ ] Compose switches from local build to pinned published image tags
- [ ] CHANGELOG generated from Conventional Commits

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
