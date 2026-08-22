# vLLM Benchmarking Framework

Standard work and tooling for measuring, recording, and tuning inference performance of
locally-hosted large language models served by [vLLM](https://docs.vllm.ai/).

This project wraps the [vLLM Benchmark CLI](https://docs.vllm.ai/en/v0.25.1/benchmarking/)
with the things it does not provide: durable result storage, run history, cross-sweep
comparison, configuration lineage, and a visual interface for parameter tuning.

> **Status:** 1.0.0rc3 — release candidate. See [ROADMAP.md](ROADMAP.md) for what remains.

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
- **Exposes** all of the above over MCP, so Claude Code and similar agent harnesses can
  drive a tuning loop directly. Off by default; see [docs/mcp.md](docs/mcp.md) to turn it
  on and connect a client, and [ADR 0001](docs/adr/0001-mcp-server-interface.md) for why it
  is shaped the way it is.

---

## Architecture

The stack is split across two hosts by design. The control plane carries no GPU
dependency and imposes no measurable load on the system under test.

```mermaid
flowchart LR
  subgraph control["Control Host — Docker Compose, no GPU"]
    direction TB
    web["web<br/>React + Vite + ECharts"]
    api["api<br/>FastAPI, JSON only"]
    orch["orchestrator<br/>sweep state machine"]
    migrate["migrate<br/>one-shot schema init"]
    pg[("postgres<br/>results + configs")]
    web --> api
    api --> pg
    api <--> orch
    orch --> pg
    migrate --> pg
  end

  subgraph gpuhost["GPU Host — uv/venv, no Docker, 1..N GPUs"]
    direction TB
    agent["vllmbench-agent<br/>FastAPI"]
    serve["vllm serve<br/>subprocess"]
    bench["vllm bench serve<br/>subprocess"]
    scraper["metrics scraper"]
    nvml["NVML sampler<br/>per GPU"]
    agent --> serve
    agent --> bench
    agent --> scraper
    agent --> nvml
    bench -.->|loopback| serve
    scraper -.->|"/metrics"| serve
  end

  orch <-->|"HTTP + token"| agent
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
Sampled via NVML for **every GPU** participating in the run, attributed per device:

- SM utilization, memory used, power draw, temperature, clock speeds

Per-device attribution is what makes tensor parallelism analyzable. A TP=4 run where one
device sits at 60% utilization while three sit at 95% is telling you something a
host-level average would hide entirely.

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
- One or more NVIDIA GPUs on a single host, with a working driver
- Python environment managed by `uv` with vLLM installed
- Network reachability from the control host on the agent port

Single-host multi-GPU is supported, and **tensor parallel size is a first-class sweep
dimension** — "is TP=2 on two GPUs better than two independent TP=1 servers" is a question
this framework exists to answer. Multi-node deployments are out of scope for 1.0.0.

---

## Quick start

Three steps: bring up the control plane, install the agent on the GPU host, take a
measurement. Verified end to end from a clean control host and a GPU host with no agent
installed.

### 1. Control plane

```bash
git clone https://github.com/er0080/vllm-benchmarking
cd vllm-benchmarking
cp .env.example .env
```

Edit `.env` and set one value:

- `VLLMBENCH_TOKEN` — the shared secret the control plane presents to the agent.
  Generate one with `python3 -c 'import secrets; print(secrets.token_urlsafe(32))'`.

Everything else has a working default. The GPU host's address is not set here — it belongs
to the host and is entered when you register it in step 3.

```bash
docker compose up -d
```

That builds the images, applies the schema, and starts Postgres, the API, the orchestrator
and the web UI. Budget about a minute, plus however long your connection takes to pull the
base images the first time.

```bash
curl -s localhost:8080/api/health   # {"status":"ok", ... "schema":{"ok":true, ...}}
```

Then open <http://localhost:8080>.

If it reports `degraded`, the message names which half is unhappy — the database or the
schema. Leaving `VLLMBENCH_TOKEN` at its example value is not an error, but both services
say so in their logs at startup, because a shared secret that ships in a public repository
is not a secret.

### 2. Agent, on the GPU host

Install it **into the vLLM environment**, from git:

```bash
source /path/to/vllm-env/.venv/bin/activate
uv pip install "git+https://github.com/er0080/vllm-benchmarking@v1.0.0rc3#subdirectory=packages/agent"
```

Give it the same token, in a file rather than on a command line where `ps` can read it,
and start it:

```bash
umask 077 && echo "VLLMBENCH_TOKEN=the-token-from-step-1" > ~/.vllmbench-agent.env
set -a; . ~/.vllmbench-agent.env; set +a; vllmbench-agent
```

**That runs in the foreground and stops when you close the terminal** — including when an
SSH session ends. Fine for a first measurement; leave it running and use a second terminal
for the check below. For anything beyond that, install it as a service:
[docs/agent-installation.md](docs/agent-installation.md) has a systemd unit.

From a second terminal on the GPU host:

```bash
curl -s localhost:9110/health
```

The port is 9110 and the control host has to be able to reach it — that is the one piece
of network configuration this system needs.

That guide also covers upgrading, and the environment facts that cost the most time —
chiefly that a service does not inherit the `HF_HOME` you set in `~/.bashrc`, so vLLM
re-downloads weights the host already has.

### 3. Your first measurement

In the UI:

1. **Hosts → Register.** Give it a name and the agent URL. The handshake reads the host's
   facts, and per-device GPU model, VRAM, driver, CUDA and vLLM version appear. Those
   become provenance on every run this host produces.
2. **Runs.** Pick the host. Config and workload can both be authored inline the first
   time — leave each dropdown on "New from…" and fill in the fields below it:

   ```yaml
   model: facebook/opt-125m
   max_model_len: 2048
   gpu_memory_utilization: 0.30
   tensor_parallel_size: 1
   ```

   A workload of 20 prompts at concurrency 4 is enough to prove the path. Start it.
3. Watch it move through `starting` and `benchmarking`, then open the run. TTFT, TPOT, ITL
   and per-GPU throughput are in Postgres, with the raw benchmark payload kept verbatim
   beside them and a per-second telemetry timeline underneath.

Once the model is cached on the GPU host this takes about ninety seconds end to end. The
very first time, add however long the weights take to download — the agent is running
`vllm serve`, and vLLM fetches from HuggingFace like it always does.

The config you just wrote is now content-addressed and reusable. The **Configs** tab is
where configs get validated before they cost GPU time, annotated with the run that
justifies them, and exported — the bytes are identical to what ran, so they go straight
into production.

From there: **Sweeps** authors a matrix of configs × workloads with replicates, and the
analysis tabs chart the results. What each chart means and what to change next is
[docs/tuning-playbook.md](docs/tuning-playbook.md).

### Without a GPU

The stack is fully developable on a laptop. `make dev` — or
`docker compose --profile dev up -d` — adds a mock agent that implements the agent's HTTP
contract and returns synthetic but realistic results and telemetry, with configurable
failure injection. Everything it produces is marked synthetic at the moment of creation and
can never be charted beside a real measurement.

### Stopping, starting, and what deletes results

```bash
docker compose stop      # stop, keep everything
docker compose up -d     # start again
docker compose down      # remove the containers; the database volume survives
docker compose down -v   # remove the volume too — this destroys every recorded run
```

Only the last one loses data, and it loses all of it. Results live in a Docker volume, so
they outlive the containers but not that flag. `make help` lists the rest of the targets,
including `make check` for everything CI runs.

---

## Documentation

- [ROADMAP.md](ROADMAP.md) — milestones, scope, and definition of done for 1.0.0
- [CLAUDE.md](CLAUDE.md) — architecture invariants and development conventions
- [docs/agent-installation.md](docs/agent-installation.md) — installing, running and upgrading the agent on a GPU host
- [docs/tuning-playbook.md](docs/tuning-playbook.md) — how to read each chart and what to change next
- [docs/upgrading.md](docs/upgrading.md) — moving a running deployment forward, and what cannot be rolled back
- [docs/limitations.md](docs/limitations.md) — what this release does not do
- [docs/adr/](docs/adr/) — accepted architecture decisions and the alternatives they rejected
- [docs/hardware-verification.md](docs/hardware-verification.md) — what real hardware found, and what it confirmed
- [vLLM Benchmark CLI](https://docs.vllm.ai/en/v0.25.1/benchmarking/cli/) — upstream reference
- [vLLM Parameter Sweeps](https://docs.vllm.ai/en/v0.25.1/benchmarking/sweeps/) — upstream sweep tooling
