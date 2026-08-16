# CLAUDE.md

Working agreements for AI-assisted development in this repository.
Read this before making changes. See [README.md](README.md) for what the project is and
[ROADMAP.md](ROADMAP.md) for what is in scope right now.

---

## The one thing to internalize

**This repository produces measurements.** A bug that makes a page render wrong is a bug.
A bug that makes a number wrong is worse, because the number gets recorded, charted,
compared against last month's number, and acted on. Correctness of recorded data outranks
features, speed, and elegance.

When in doubt about whether a change could alter what gets measured or how it is
attributed, treat it as high-risk and say so.

---

## Architecture invariants

These are decisions, not preferences. Changing one is an architectural change that needs
explicit discussion, not a refactor.

1. **Two hosts, one boundary.** The control plane never has GPU access and never assumes
   vLLM is reachable locally. All GPU-side work goes through the agent's HTTP API. Do not
   add a code path where the API or orchestrator shells out to `vllm`, `nvidia-smi`, or
   `torch`.

2. **The benchmark client runs on the GPU host, over loopback.** This keeps network RTT
   out of TTFT and ITL. Never add a mode that benchmarks a remote endpoint from the
   control plane without recording that fact on the run and preventing it from being
   charted alongside loopback runs.

3. **No Docker on the GPU host.** The agent is a `uv`-installable Python package that
   invokes the vLLM venv's own binaries as subprocesses. Do not introduce a container
   runtime dependency there.

4. **`vllm bench serve` is the atomic unit.** We orchestrate; we do not reimplement the
   benchmark. Do not hand-roll an HTTP load generator. If upstream's client lacks
   something we need, wrap it or record the gap in `docs/adr/`.

5. **Server configs are native vLLM YAML.** What is stored in the database is what is
   written to disk and passed to `vllm serve --config`. No intermediate schema that has to
   be translated. Validate; do not transform.

6. **Every run records its provenance.** vLLM version, agent version, GPU model, driver
   version, config hash, dataset identity, and `initiated_by` (`ui` / `mcp` / `api`, plus
   client identity where available) are captured on every run. A run that cannot state
   what produced it is not a valid result — and once more than one interface can start a
   sweep, "what produced it" includes which interface asked.

7. **Synthetic runs are quarantined.** Any run produced by the CPU backend, the fake
   `vllm` shim, or the mock agent is marked as such at the moment of creation and can
   never appear in a chart, a comparison, or an export alongside real measurements. This
   flag is set by the producer, not inferred later.

8. **Throughput is normalized by GPU count.** Single-host multi-GPU is in scope and tensor
   parallelism is a first-class sweep dimension, so raw aggregate throughput is not a
   comparable number: a TP=4 run trivially out-throughputs a TP=1 run while potentially
   being far worse per device. Every throughput figure carries both its aggregate and its
   per-GPU value, and comparison views default to per-GPU. Parallelism topology
   (`tensor_parallel_size`, `pipeline_parallel_size`, GPU count and device indices) is
   provenance under invariant 6 and is never inferred from the config text after the fact.

---

## Tech stack

| Layer | Choice | Notes |
| --- | --- | --- |
| Database | PostgreSQL | Results, configs, sweep definitions |
| Control API | FastAPI (Python) | JSON only, no server-rendered HTML |
| Orchestrator | Python, separate service | Long-running sweeps survive API restarts |
| Agent | FastAPI (Python), uv package | Deployed to GPU host, not containerized |
| Frontend | React + Vite + ECharts | Served by nginx |
| Migrations | Alembic | Forward-only, see below |
| Packaging | uv | Both control plane and agent |

---

## Data model conventions

- **Raw before derived.** Persist the full benchmark JSON verbatim in a `jsonb` column
  alongside the flattened, queryable columns. If we later discover we flattened something
  wrong, the raw record lets us recompute. Never discard the original.
- **Time-series tables are append-only.** `engine_sample`, `gpu_sample`, and
  `request_sample` are never updated in place.
- **`gpu_sample` is keyed per device**, on `(run_id, gpu_index, sampled_at)`. Never
  aggregate at write time — a host-level average destroys the imbalance signal that makes
  a tensor-parallel run diagnosable.
- **Configs are content-addressed.** A `server_config` is identified by the hash of its
  canonicalized YAML. Two runs claiming the same config must have byte-identical effective
  configuration.
- **Runs are immutable once terminal.** A run in `succeeded` or `failed` state is never
  mutated. Re-running produces a new run, linked by `sweep_id` and `replicate_idx`.
- **Migrations are forward-only.** No destructive migrations against tables holding
  results. Renames are add-column, backfill, drop-later.

---

## Working on the agent

The agent runs on the machine under test. Its resource footprint is part of its contract.

- Sampling loops must be bounded and cheap. Telemetry sampling should not be a measurable
  perturbation of the thing it measures.
- Subprocess lifecycle must be leak-free. Every `vllm serve` this agent starts, it kills —
  including on agent crash and restart. Orphaned processes holding VRAM are a serious
  failure mode; reap on startup.
- Wait for real readiness, not a sleep. Poll `/health` and confirm the model is loaded
  before starting a benchmark.
- Reset caches between runs. Upstream calls the `/reset_*_cache` endpoints between
  benchmark runs; prefix cache carryover across sweep points silently invalidates results.
- Stream logs and progress back to the control plane. A sweep that appears frozen for
  twenty minutes because a model is loading is a usability failure.

---

## Working on the frontend

- Charts are the product. Load the `dataviz` skill before writing chart code.
- The primary tuning view is the Pareto frontier: per-user output tok/s against per-GPU
  total tok/s. Build for that first.
- Never chart runs together that differ in provenance in ways that invalidate the
  comparison — different vLLM version, different GPU, different bench-client location.
  Group or warn; do not silently overlay.
- Show uncertainty. Sweeps default to multiple replicates; render the spread, not just the
  mean of three runs.

---

## Conventions

- Match the surrounding code's naming, comment density, and idiom.
- Python: type hints on public functions, `ruff` for lint and format.
- TypeScript: strict mode on.
- Secrets come from environment variables. Never commit a token, and never log one.
- Prefer editing existing files over creating new ones. Do not add documentation files
  that were not asked for.

---

## Development workflow

### Everything starts as an issue

All development, pre- and post-1.0.0, is driven by GitHub Issues. GitHub is the system of
record for change management, not a mirror of decisions made elsewhere.

- **No PR without an issue.** If work is worth doing, it is worth an issue first. Open one
  rather than skipping the step.
- **Label every issue** — at minimum a type (`enhancement`, `bug`, `documentation`) and
  the target milestone.
- **Milestones map to roadmap versions** (`0.1.0`, `0.2.0`, …). An issue with no milestone
  is unscheduled work, and that should be a deliberate state rather than an oversight.
- **Cross-link in both directions.** PR bodies use closing keywords (`Closes #12`) so the
  issue closes on merge. When a PR partially addresses an issue, say which part and link
  without a closing keyword.
- **Design discussion belongs in the issue**, not the PR. The issue is where a decision
  and its rejected alternatives are recorded; the PR is where the implementation of that
  decision is reviewed. Decisions that outlive the issue graduate to `docs/adr/`.

### Branching

- Short-lived feature branches off `main`, named `<type>/<issue-number>-<slug>` — for
  example `feat/14-sweep-orchestrator` or `docs/1-initial-feedback`.
- `main` is protected: PRs required, linear history, no force pushes.
- Pull requests are required. Self-merge is fine; merging without green CI is not.
- Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`). The
  scope is the component: `feat(agent):`, `fix(orchestrator):`.
- Squash merge. The PR title becomes the commit message, so it must be a valid
  Conventional Commit.

### Local development without a GPU

Most work does not need the GPU host, and should not require it. Two facilities exist so
that the control plane and frontend are fully developable on a laptop:

- **Mock agent** — implements the agent's HTTP contract and returns synthetic but
  realistic benchmark JSON and telemetry, with configurable latency and failure injection.
  Started with `docker compose --profile dev up`. It is also the integration-test fixture.
- **Fake `vllm` shim** — a stand-in binary accepting `serve --config` and
  `bench serve --save-result`, exposing `/health` and `/metrics`. Used for agent tests.

Both produce runs flagged synthetic per invariant 7. Keep them honest: when the real
agent's contract changes, the mock changes in the same PR.

### Pre-commit

`pre-commit` runs `ruff` (lint and format), `pyright`, and `tsc --noEmit`. Do not bypass
with `--no-verify`; if a hook is wrong, fix the hook.

### Versioning

Agent and control plane version in lockstep from a single source of truth. The agent
reports a protocol version on connect and the control plane refuses to run against a
mismatch, with an explicit message naming both versions. A stale agent must fail loudly at
connect time rather than produce subtly wrong data mid-sweep.

---

## Testing

Two tiers, both fully automated, neither requiring a GPU. There is deliberately **no
hardware validation tier**: the framework is the hardware test. Every real sweep exercises
the complete GPU path — server launch, NVML sampling, `/metrics` scraping, teardown — so a
separate validation script would only re-run that on the machine we are trying to keep
quiet. Failures on the GPU path are loud and immediate; the silent-corruption class is
what tier 2 exists to catch.

Do not add a CI runner, agent, or scheduled job to the GPU host. It is the system under
test, and anything running there during a sweep corrupts the measurement in flight.

### Tier 1 — Fast (every push, GitHub-hosted)

- Lint, format, type checks
- Unit tests, including the **JSON-to-column flattening layer**. This is the
  highest-consequence code in the repository; a mistake here corrupts every result
  silently and invisibly. It gets the most thorough tests in the codebase.
- Agent lifecycle tests against the fake `vllm` shim: readiness polling, subprocess
  reaping, timeout handling, crash-during-model-load, orphan cleanup on restart.

### Tier 2 — Integration (every PR, GitHub-hosted)

- Postgres service container, migrations applied from scratch
- Control plane against the mock agent, end to end
- **Real vLLM on the CPU backend** (`facebook/opt-125m`, `--load-format dummy`), which
  exposes the same OpenAI API and `/metrics` endpoint as the GPU build. This tier exists
  to validate the *actual* upstream contract rather than our assumptions about it — it is
  what catches a `--save-result` field rename in a new vLLM version before that rename
  reaches the database. This is the *only* automated check against real vLLM anywhere in
  the project, so treat a failure here as blocking rather than flaky.

  Use the official `vllm/vllm-openai-cpu` image as a service container rather than
  installing CPU wheels. It publishes both `latest-x86_64` and `latest-arm64` tags, which
  sidesteps the question of whether GitHub-hosted runners expose AVX512 and makes this
  tier runnable locally on an Apple Silicon Mac.

---

## vLLM version policy

`VLLM_REFERENCE_VERSION` at the repo root pins the version tier 2 tests against. Bumping
it is an explicit PR, never an incidental change.

Two mismatches exist and they are not the same thing:

| Mismatch | Behavior |
| --- | --- |
| **Agent protocol version** vs. control plane | **Refuse to run**, naming both versions. This is our own contract; a mismatch means wrong data. |
| **GPU host's vLLM version** vs. `VLLM_REFERENCE_VERSION` | **Warn, record, never block.** |

The second must never become a hard failure. Benchmarking one vLLM version against another
is a legitimate — arguably headline — use of this tool, and blocking on it would prevent a
feature. `vllm_version` is first-class provenance per invariant 6: charts group by it and
never silently overlay results across versions.

---

## Platform support

The stack must run with minimal friction on both traditional Linux and macOS with
Colima. The architecture already helps here — the control plane has no GPU dependency of
any kind — but these are real constraints, not aspirations:

- **Multi-arch.** Apple Silicon under Colima is `linux/arm64`. Pre-1.0 this is free
  because everything builds from source. From 1.0.0, published GHCR images must be
  `linux/amd64,linux/arm64`; a single-arch image forces emulation and makes the stack
  unusably slow on a Mac.
- **No `host.docker.internal` dependency.** The agent lives on a separate host reachable
  by LAN address, and the mock agent is a compose service reachable by service name.
  Neither needs Docker's host-gateway alias, whose behavior differs between Docker Desktop
  and Colima. Do not introduce a dependency on it.
- **Colima resource minimums** are a documented prerequisite. The defaults (2 CPU / 2 GB)
  cannot run Postgres alongside a vLLM CPU container.
- **File watching over virtiofs is unreliable.** Dev-mode Vite and uvicorn must use
  polling watchers, or hot reload silently stops working on macOS.
- **No absolute host paths** in compose files, and no assumptions about UID/GID mapping,
  which differs between Colima and native Linux.

Any change that works on only one of the two platforms is incomplete.

---

## CI/CD

GitHub Actions. Workflows mirror the test tiers:

- `ci.yml` — tiers 1 and 2, on push and PR. Required for merge.
- `build.yml` — builds control-plane images and the agent wheel on `main` to prove they
  build. **Does not publish before 1.0.0.**

### Releases

Pre-1.0 there are no published artifacts. Compose builds images locally; the agent is
installed on the GPU host with `uv pip install git+<repo>@<tag>`. Tags are cut for
milestones so a GPU host can be pinned to a known commit.

From 1.0.0, a `v*` tag triggers publishing control-plane images to GHCR — built for
`linux/amd64` and `linux/arm64` — and attaching the agent wheel to a GitHub Release. The
agent wheel itself is pure Python and arch-independent. Do not build the release workflow
before then; it is roadmap milestone 1.0.0, not scaffolding.

### Migrations in CI

Every PR applies migrations to an empty database and to a database seeded at the previous
tag. Forward-only means a migration that cannot run against real existing data is a
release-blocking bug, and CI is where that gets caught.

---

## When you are unsure

Ask about anything that would change what gets recorded, how a run is attributed, or the
host boundary. Make routine judgment calls on everything else without checking in.
