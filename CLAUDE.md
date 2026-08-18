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
| Logging | stdlib `logging`, JSON lines | `vllmbench_protocol.logging`; text on a TTY |
| Migrations | Alembic | Forward-only, see below |
| Packaging | uv | Both control plane and agent |

---

## Data model conventions

- **Retention deletes telemetry and nothing else.** `vllmbench_db.retention` names what
  it may never touch — runs, summaries, raw payloads, provenance — and refuses to run if
  that list ever intersects what it prunes. A pruned run is *recorded* as pruned, because
  an empty timeline otherwise reads as a sampling bug rather than as the policy working.
  Deleting samples does not contradict append-only: append-only means never rewritten in
  place, which is what stops a chart lying about what was observed.
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
- Disk is part of the footprint too, and it fails quietly. Working directories left by a
  SIGKILL are swept at startup on the same principle as reaping an orphaned engine, and
  headroom is checked before starting work — a disk with no room turns a forty-minute
  benchmark into a write error at the end of it.
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
  Redaction is not left to care at each call site: every service calls
  `configure_logging(...)` at startup with the values it holds, and the formatter strips
  them from the rendered line, traceback included. Add a secret to settings and it must be
  added there in the same change — a test asserts each service registers what it holds,
  because the failure mode is otherwise silent until something goes wrong near it.
- Prefer editing existing files over creating new ones. Do not add documentation files
  that were not asked for.

---

## Development workflow

### Issues: optional now, required from 0.10.0

Process rigor is staged deliberately. Development (**0.1.0 through 0.9.0**) favors rapid
iteration; release candidacy (**0.10.0 onward**) favors an auditable trail. The rules
change at that boundary, and only there.

**Through 0.9.0 — issues are optional.** Open one when it earns its keep:

- Design work with real alternatives to weigh
- External feedback or a bug report
- Work spanning several PRs that needs a place to be tracked

Otherwise go straight to a branch and a PR. A one-line issue that exists only to be
referenced by a PR is bureaucracy, not change management.

Hardening (0.9.0) is deliberately inside the relaxed window rather than the first
milestone outside it. That milestone defines the behavior of every failure mode in the
system and settles the schema — the work is still design, done by discovering what breaks,
and it is the phase most damaged by making each discovery cost an issue first. What
happens there is captured the same way everything else has been: in the PR body, and in
`docs/adr/` when a decision outlives the work.

**From 0.10.0 — every PR traces to an issue**, labeled with a type (`enhancement`, `bug`,
`documentation`) and a milestone, cross-linked with closing keywords (`Closes #12`). By
then the schema is stable, the failure modes have defined behavior, and the remaining work
is verification and documentation against a fixed target. From that point a change is a
candidate for release rather than a step toward one, and knowing why it landed matters more
than the speed of landing it.

### Where the "why" lives when there is no issue

This is the part that makes the relaxation safe rather than merely faster. Rationale does
not become optional just because issues do — it moves.

- **The PR body carries the reasoning.** What was tried, what was rejected, and what the
  reader would otherwise have to reconstruct. A PR whose body is a restatement of its diff
  has thrown away the only durable record of intent.
- **Binding decisions go to `docs/adr/`**, regardless of milestone or whether an issue
  existed. An ADR is not a heavier issue; it is the record of a decision that outlives the
  work, and it is required whenever one is made.

Squash merge means the PR body becomes the permanent commit record. It is the artifact
someone reads in a year, so write it for them.

### Branching

- Short-lived feature branches off `main`, named `<type>/<slug>`, or
  `<type>/<issue-number>-<slug>` when an issue exists — for example
  `feat/sweep-orchestrator` or `docs/1-initial-feedback`.
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

  Failure injection is `VLLMBENCH_MOCK_START_FAILURE` / `VLLMBENCH_MOCK_BENCH_FAILURE`,
  set to a `FailureKind` value. The mock then refuses with the same status code, message
  and `X-Vllmbench-Failure-Kind` header the real agent sends — the messages themselves
  lifted from captured vLLM output. This is the only way most failure paths are reachable
  without hardware: an engine cannot run out of memory on a laptop, and that is the
  failure a tuning sweep hits most.
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

  Use the official `vllm/vllm-openai-cpu` image, pinned to `VLLM_REFERENCE_VERSION`, and
  **run it on an arm64 runner** (`ubuntu-24.04-arm`, free for public repos).

  This is not arbitrary. The x86_64 image does not work on GitHub's x86 runners: they are
  AMD EPYC Zen 5, and the image picks oneDNN kernels that kill the worker during model
  warmup — silently under fp32, and with a matmul primitive error under bf16. It is not a
  missing CPU feature; those chips have AVX512 and bf16, and forcing a conservative
  `ONEDNN_MAX_CPU_ISA` did not help. Five hypotheses on that axis failed before changing
  arch fixed it in one attempt.

  The arm64 image is also what developers run locally on Apple Silicon, so CI and local
  development exercise the same artifact. Two flags are required rather than tuning:
  `--shm-size` (the engine core communicates over shared memory, and Docker's 64 MB
  default kills it) and `--cap-add=SYS_NICE` (NUMA binding syscalls are seccomp-gated on
  that capability, and denial surfaces only as a warning before the worker dies).

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

## Upstream contracts are verified, never assumed

**Do not write code against vLLM's documentation. Write it against a captured payload.**

This is not a general caution; it already cost real work. The published benchmarking docs
describe `--save-result` fields named `successful_requests`, `benchmark_duration_sec` and
`ttft_ms_p99`. vLLM 0.25.1 emits `completed`, `duration` and `p99_ttft_ms`. A flattening
layer written from the docs would have parsed cleanly, reported success, and written NULL
for every metric — the exact silent corruption this project exists to avoid.

So, whenever consuming something vLLM produces:

1. **Capture a real payload first**, from the CPU backend if no GPU is at hand, and check
   it into `tests/fixtures/`. A hand-written fixture encodes the author's belief about
   the format, which is the belief under test.
2. **Pin the field names in one place** — `vllmbench_protocol.bench_result` and
   `.metrics` — so there is a single thing to correct when upstream moves.
3. **Have tier 2 re-derive the contract from a live server**, so a vLLM upgrade that
   renames a field fails a test instead of corrupting a sweep.
4. **Fail loudly on a payload that does not match.** A summary row full of NULLs is
   indistinguishable from a benchmark that legitimately measured nothing.

The same reasoning applies to `/metrics`: there is no prefix-cache hit-rate gauge, only
`prefix_cache_queries_total` and `prefix_cache_hits_total`. Store counters, derive rates.
Counters can be differenced across any window; a rate sampled at an instant cannot be
recovered.

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

**From git, never from a local clone.** `vllmbench-protocol` is a workspace member, so
inside a checkout `uv pip install ./packages/agent` resolves it via `tool.uv.sources` and
installs it *editable* — leaving the GPU host's environment dependent on a source tree
that nobody thinks of as load-bearing. It works until the clone is deleted, then the
agent dies with `ModuleNotFoundError` on a machine with no source and no explanation.
This is verified, not trusted: `agent-install` in `ci.yml` installs from a throwaway
clone, deletes it, and fails if anything in the resulting environment still points at a
directory.

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
