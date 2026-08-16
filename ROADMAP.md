# Roadmap to 1.0.0

Milestones are ordered so that each one ends with something demonstrably working, not with
a layer that only pays off later. The first real proof point is 0.2.0 — one config, one
workload, one number, in the database, on screen.

Versions are pre-1.0 and make no compatibility promises. The database schema may change
destructively before 0.9.0.

---

## 0.1.0 — Foundations

Scaffolding only. Nothing measures anything yet.

- [ ] Repository structure, `uv` workspaces, `ruff` and TypeScript strict configs
- [ ] `compose.yaml` — postgres, api, orchestrator, web, migrate
- [ ] Initial schema and Alembic setup: `gpu_host`, `model`, `server_config`, `workload`,
      `sweep`, `run`, `run_summary`, `engine_sample`, `gpu_sample`
- [ ] Agent package skeleton with `/health` and `/host-info` (per-GPU model, VRAM, device
      index, driver, CUDA, vLLM version)
- [ ] Token auth and protocol-version handshake between control plane and agent
- [ ] Control plane can register a GPU host and display its facts
- [ ] **Mock agent** behind `--profile dev` — synthetic results and telemetry, so the
      control plane and UI are developable with no GPU host present
- [ ] **Fake `vllm` shim** for agent lifecycle tests
- [ ] `pre-commit` with ruff, pyright, tsc
- [ ] CI tier 1 (lint, types, unit) and tier 2 (postgres, mock agent, real vLLM on the CPU
      backend); branch protection on `main` requiring green
- [ ] `VLLM_REFERENCE_VERSION` pinned at the repo root
- [ ] Verified on both native Linux and macOS/Colima, including the `vllm-openai-cpu`
      service container on `arm64` and `x86_64`
- [ ] Colima resource minimums documented in the README prerequisites

**Done when:** `docker compose up` succeeds on both Linux and macOS/Colima, the UI shows
live facts read from a real GPU host through the agent, and the full CI suite passes
without any GPU.

---

## 0.2.0 — Single run, end to end

The vertical slice. This is the milestone that proves the architecture.

- [ ] Agent: launch `vllm serve --config <yaml>`, poll `/health` to true readiness,
      capture logs, tear down cleanly
- [ ] Agent: run `vllm bench serve --save-result`, return parsed JSON
- [ ] Agent: reap orphaned vLLM processes on startup
- [ ] Control plane: flatten benchmark JSON into `run_summary`, retain raw `jsonb`
- [ ] UI: define one server config and one workload, trigger a run, watch it progress,
      view the result

**Done when:** a single benchmark run is triggered from the browser and its TTFT, TPOT,
ITL, and throughput figures land in Postgres and render on a run detail page.

---

## 0.3.0 — Telemetry

Turns "which config won" into "why it won."

- [ ] Agent: sample vLLM `/metrics` during runs — KV cache utilization, prefix cache hit
      rate, running and waiting queue depth, preemptions
- [ ] Agent: sample NVML **per device** during runs — SM utilization, memory, power,
      temperature, clocks — for every GPU participating in the run
- [ ] Bounded, low-overhead sampling loops with configurable interval
- [ ] Run detail page: telemetry timeline aligned to the benchmark window, with per-device
      series rather than a host-level average

**Done when:** a saturated run visibly shows KV cache pressure and a growing waiting queue
on the run detail timeline, and a tensor-parallel run shows each participating GPU as its
own series.

---

## 0.4.0 — Sweeps

- [ ] Sweep authoring: matrix of server configs × workloads, with replicate count
- [ ] `tensor_parallel_size` as a first-class sweep axis, with validation against the
      host's available device count
- [ ] Orchestrator state machine: queue, execute, retry, resume, cancel
- [ ] Cache reset between runs (`/reset_*_cache`)
- [ ] Server restart between server-config changes; reuse across workload-only changes
- [ ] Live sweep progress in the UI, with mid-sweep cancellation
- [ ] Sweep survives an API restart

**Done when:** a multi-hour sweep runs unattended, survives a control-plane restart, and
can be cancelled cleanly without leaving orphaned processes or VRAM held.

---

## 0.5.0 — Analysis

The payoff milestone. Load the `dataviz` skill before building these.

- [ ] **Pareto frontier** — per-user output tok/s against per-GPU total tok/s, normalized
      by device count so tensor-parallel configurations are honestly comparable
- [ ] Tensor-parallel scaling view: throughput and latency against TP size, with per-GPU
      efficiency, answering whether TP=N earns its extra devices
- [ ] Per-device utilization comparison to expose imbalance within a TP group
- [ ] Latency-versus-concurrency curves, p50 and p99
- [ ] Throughput saturation curves against request rate
- [ ] Side-by-side run and config comparison with a config diff
- [ ] Replicate spread rendered, not averaged away
- [ ] Filtering, and provenance guards that refuse to silently overlay incomparable runs
- [ ] Saved views

**Done when:** a tuning decision can be made from the UI alone, without exporting to a
notebook.

---

## 0.6.0 — Agent interface (MCP)

Per [ADR 0001](docs/adr/0001-mcp-server-interface.md). Placed here because the valuable
analysis tools depend on 0.5.0 and the control tools on 0.4.0.

- [ ] Streamable HTTP MCP server mounted at `/mcp` on `api`, targeting the 2026-07-28
      spec, behind `VLLMBENCH_MCP_ENABLED`
- [ ] Bearer token auth; default bind makes accidental exposure hard, and the docs state
      plainly that `/mcp` is LAN-only
- [ ] Config **validation engine** — checks a candidate YAML against the target vLLM
      version's accepted arguments and returns structured, actionable errors. Pulled
      forward from 0.7.0: `validate_config` needs it, and 0.7.0's YAML editor is a UI over
      this engine rather than a separate implementation.
- [ ] Read tools — `list_hosts`, `list_configs` / `get_config`, `validate_config`,
      `list_workloads`, `list_sweeps` / `get_sweep`, `query_runs`, `get_run`,
      `get_run_telemetry`, `compare_runs`, `get_pareto`
- [ ] Write tools, enabled by default — `create_config`, `create_workload`,
      `create_sweep`, `start_sweep`, `cancel_sweep`. No tool mutates or deletes a run.
- [ ] MCP resources: `vllmbench://config/{hash}`, `vllmbench://sweep/{id}/report`
- [ ] Context economy: pagination with a hard maximum, summary fields by default,
      server-side telemetry downsampling
- [ ] `initiated_by` provenance on sweeps and runs (`ui` / `mcp` / `api`) plus client
      identity where available
- [ ] `get_sweep` returns `estimated_remaining`, extrapolated from completed points
- [ ] Guardrails: one active sweep per host enforced in the domain layer, bounded matrix
      size, exact run count with a structured duration estimate, audit log of write calls

**Done when:** an agent connected over MCP can author a config, define and start a sweep,
poll it to completion, and read back a per-GPU normalized Pareto frontier — without
touching the UI.

---

## 0.7.0 — Configuration management

- [ ] YAML editor surfacing the validation engine from 0.6.0 — one implementation, two
      interfaces
- [ ] Content-addressed config storage and lineage (which config was derived from which)
- [ ] Export a config for direct use with `vllm serve --config`
- [ ] Import an existing YAML
- [ ] Annotate a config with the result that justified it

**Done when:** the winning configuration from a sweep can be exported and run in
production unchanged.

---

## 0.8.0 — Interop

- [ ] Importer for upstream `vllm bench sweep serve` output directories
- [ ] CSV and JSON export of any result set
- [ ] Shareable run and sweep reports
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
