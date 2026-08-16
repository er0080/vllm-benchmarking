# vLLM Benchmarking Framework

Standard work and tooling for measuring, recording, and tuning inference performance of
locally-hosted large language models served by [vLLM](https://docs.vllm.ai/).

This project wraps the [vLLM Benchmark CLI](https://docs.vllm.ai/en/v0.25.1/benchmarking/)
with the things it does not provide: durable result storage, run history, cross-sweep
comparison, configuration lineage, and a visual interface for parameter tuning.

> **Status:** pre-release. See [ROADMAP.md](ROADMAP.md) for the path to 1.0.0.

---

## What problem this solves

`vllm bench serve` produces a JSON blob and exits. `vllm bench sweep serve` produces a
directory of JSON blobs and PNGs. Neither remembers what you ran last week, neither lets
you ask "which `max_num_seqs` gave the best per-GPU throughput at p99 TTFT under 200ms,"
and neither tells you *why* a configuration lost.

This framework:

- **Designs** sweeps as a matrix of server configs × workloads, authored in a UI.
- **Runs** them against a real vLLM server it starts and stops per configuration.
- **Records** benchmark results, engine telemetry, and GPU telemetry to PostgreSQL.
- **Displays** them as Pareto frontiers, saturation curves, and per-run timelines.
- **Stores** server configurations as native vLLM YAML, directly usable with `vllm serve --config`.

---

## Architecture

The stack is split across two hosts by design. The control plane carries no GPU
dependency and imposes no measurable load on the system under test.

```
CONTROL HOST  (Docker Compose, no GPU)          GPU HOST  (uv/venv, no Docker)
┌───────────────────────────────────────┐       ┌─────────────────────────────────┐
│  web            React + Vite + ECharts│       │  vllmbench-agent   (FastAPI)    │
│  api            FastAPI, JSON only    │◄─────►│    ├─ vllm serve      subprocess│
│  orchestrator   sweep state machine   │ HTTP  │    ├─ vllm bench serve subprocess│
│  postgres       results + configs     │ +token│    ├─ /metrics scraper          │
│  migrate        one-shot schema init  │       │    └─ NVML sampler              │
└───────────────────────────────────────┘       └─────────────────────────────────┘
```

### Why this split

| Decision | Rationale |
| --- | --- |
| Control plane on a separate host | Keeps the system under test quiet. No database, web server, or browser competing for CPU during a measurement. |
| Benchmark client on the **GPU host**, over loopback | Network RTT and jitter never enter the TTFT/ITL numbers. Results stay directly comparable to published vLLM figures. |
| Agent is a uv package, not a container | The agent must invoke the venv's own `vllm` binary. Installing it into that environment means **no container runtime is required on the GPU host**. |
| Framework owns the sweep loop | `vllm bench serve` is the atomic unit; orchestration is ours. This buys live progress, mid-sweep cancellation, per-run streaming into Postgres, and telemetry interleaved with each run. |
| Configs stored as vLLM YAML | vLLM accepts `--config file.yaml` natively. No lossy translation layer between what you see and what runs. |

---

## What gets measured

### Benchmark results
Captured from `vllm bench serve --save-result`:

- Throughput — requests/sec, output tokens/sec, total tokens/sec
- **TTFT** — time to first token, mean / median / p99
- **TPOT** — time per output token, mean / median / p99
- **ITL** — inter-token latency, mean / median / p99
- Request counts, token counts, benchmark duration

### Engine telemetry
Sampled from the vLLM `/metrics` Prometheus endpoint for the duration of each run:

- KV cache utilization
- Prefix cache hit rate
- Running / waiting queue depth
- Preemption counts

### GPU telemetry
Sampled via NVML on the GPU host:

- SM utilization, memory used, power draw, temperature, clock speeds

The last two categories are the point. A p99 TTFT number tells you a configuration lost.
KV cache pressure and queue depth tell you *why*, which is what makes the next
configuration better instead of merely different.

---

## Prerequisites

### Control host
- Docker and Docker Compose v2 — native Linux or macOS with [Colima](https://github.com/abiosoft/colima)
- No GPU required

On macOS, Colima's defaults are too small to run the stack alongside the CPU-backend test
container. Start it with more headroom:

```bash
colima start --cpu 4 --memory 8 --disk 60
```

### GPU host
- NVIDIA GPU with a working driver
- Python environment managed by `uv` with vLLM installed
- Network reachability from the control host on the agent port

Single-GPU targets are the supported configuration. Multi-GPU and multi-host are out of
scope for 1.0.0.

---

## Quick start

> Not yet implemented — see [ROADMAP.md](ROADMAP.md) milestone 0.2.0.
> This section describes the intended interface.

**On the GPU host**, install and start the agent inside the vLLM environment:

```bash
uv pip install vllmbench-agent
vllmbench-agent serve --host 0.0.0.0 --port 9110 --token "$VLLMBENCH_TOKEN"
```

**On the control host**, bring up the stack:

```bash
cp .env.example .env      # set VLLMBENCH_AGENT_URL and VLLMBENCH_TOKEN
docker compose up -d
```

Open http://localhost:8080, register the GPU host, and author your first sweep.

---

## Repository layout

```
.
├── agent/          vllmbench-agent — uv package deployed to the GPU host
├── api/            FastAPI control-plane service
├── orchestrator/   sweep execution state machine
├── web/            React + Vite + ECharts frontend
├── db/             schema migrations
├── configs/        vLLM YAML server configurations
├── docs/           standard work, tuning playbook, ADRs
├── compose.yaml
├── CLAUDE.md       working agreements for AI-assisted development
└── ROADMAP.md      milestones to 1.0.0
```

---

## Documentation

- [ROADMAP.md](ROADMAP.md) — milestones, scope, and definition of done for 1.0.0
- [CLAUDE.md](CLAUDE.md) — architecture invariants and development conventions
- [vLLM Benchmark CLI](https://docs.vllm.ai/en/v0.25.1/benchmarking/cli/) — upstream reference
- [vLLM Parameter Sweeps](https://docs.vllm.ai/en/v0.25.1/benchmarking/sweeps/) — upstream sweep tooling
