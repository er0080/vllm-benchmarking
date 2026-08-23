# Known limitations

What this release does not do, stated plainly. A limitation discovered by a user is worse
than one they were told about.

[ROADMAP.md](../ROADMAP.md) has the out-of-scope list, which answers "what will this never
do". This answers "what will bite me next week".

---

## Scale and concurrency

**One run executes at a time, across the whole deployment.** The orchestrator claims a
single queued run, executes it to completion, and claims the next. This is deliberate — a
GPU host runs one engine, so a second concurrent run would contend for the same VRAM and
measure the contention — but the consequence is that **registering a second GPU host does
not double throughput.** The second host sits idle while the first works. Multiple
concurrent hosts and a cross-host queue are post-1.0.

**A sweep is serial and can take a long time.** Most of the wall clock is model loading,
not benchmarking: on the reference sweep, two engine loads accounted for 358 of 1180
seconds. The authoring response returns `engine_starts` for this reason; that number
predicts the duration far better than the run count does.

**Single-host multi-GPU only.** Tensor and pipeline parallelism spanning hosts is out of
scope for 1.0.0.

**NVIDIA only.** Telemetry is NVML.

## Security

**The control plane has no authentication.** The API and the UI are open to anything that
can reach the port. The shared token authenticates the control plane *to the agent*, and
nothing authenticates a browser to the control plane. There is no user model, no access
control, and no audit of who changed what — only of what MCP wrote.

This is a tool for a trusted LAN. **Do not expose it to the internet.** The same applies to
`/mcp`, which is bearer-token only and off by default.

**The agent runs `vllm serve` with configuration it is handed.** Anyone who can reach the
control plane can cause arbitrary vLLM arguments to execute on the GPU host, as the user
the agent runs as. Treat control-plane access as equivalent to shell access on the GPU
host.

## Measurement

**Results are only as good as a quiet machine.** Anything else running on the GPU host
during a sweep is inside the measurement. The framework cannot detect this, and a sweep run
against a busy host produces numbers that look exactly like real ones.

**Tail latencies are noisier than throughput.** Replicates of the reference sweep agreed to
within 1% on throughput and disagreed by a factor of two on p99 TTFT, because a p99 over
128 prompts is an estimate from roughly one request. Do not decide between
configurations on a p99 alone. See [tuning-playbook.md](tuning-playbook.md).

**Grouped replicates understate real variance.** They measure repeatability under
near-identical conditions, back to back on a warm host. Variance between separate sittings
is larger, and the spread is labelled with which it is.

**Comparisons across provenance are refused, not reconciled.** Different GPU, vLLM version
or benchmark-client location means different populations. The framework groups and warns
rather than overlaying — which is correct, and means some comparisons you might want are
simply not available.

**The emission gap is not a latency you can compare across speculation settings.** What
`vllm bench serve` calls inter-token latency is the wait between *emissions*, and a
speculative emission carries every token the target accepted. Measured across four MTP
depths on one model, it rose from 25.0 ms to 44.1 ms while per-user throughput rose from
39.6 to 69.3 tok/s — both correct, describing different things. Charts label it *Emission
gap* and warn when a group mixes speculation settings, but the number itself cannot be made
comparable. Compare speed with TPOT or per-user output rate.

**A speculating engine that never drafts is silent about it.** vLLM omits the
`spec_decode_*` fields from a benchmark result entirely when no drafts happened, rather than
reporting zeros — so a NULL acceptance rate does not mean speculation was off. That is why
`run.speculative_method` is read from the engine rather than inferred from the measurement.

**A dataset identity past 2 GiB is a sampled digest, not a full hash.** Large local corpora
are identified by their first and last 64 MiB plus their length, so an edit in the middle of
one will not change it. The identity says which it is — `sha256-head-tail:` rather than
`sha256:` — because reading tens of gigabytes before every benchmark is a real cost on the
machine under test.

**Imported upstream sweeps carry no provenance.** `vllm bench sweep serve` output has no
vLLM version, GPU model, driver, host or device count in it. You are asked to declare them,
and the import is refused without them rather than defaulted. Per-GPU throughput cannot be
derived at all without a device count.

## Operations

**No published images or wheels before 1.0.0.** Compose builds locally; the agent installs
from a git tag. First run takes a few minutes to build.

**No CHANGELOG yet.** It is generated from Conventional Commits at 1.0.0. Until then the
PR bodies are the record, and they are written to be read.

**Migrations are forward-only.** There is no downgrade. Going back past a migration means
restoring a dump, which loses anything measured since. See [upgrading.md](upgrading.md).

**Retention is off by default and the database grows.** Telemetry is roughly 97% of what a
run occupies and scales with duration and device count — about 7 MB per run-hour on eight
GPUs. Nothing is deleted unless you set a horizon. `GET /api/storage` reports the real
figure for your database.

**The agent does not manage vLLM.** Installing, upgrading and configuring vLLM on the GPU
host is yours. A version mismatch against `VLLM_REFERENCE_VERSION` is recorded and warned
about, never blocked.

**Installing the agent into the vLLM environment can move that environment's packages.**
The agent declares dependency floors and vLLM declares ceilings, and nothing arbitrates
between them in a shared venv: a resolution that satisfies the agent can leave vLLM outside
its own declared constraints.

This is now *detected* rather than fixed. The agent checks its own environment at startup
and on every handshake, warns in its log, and reports the result as run provenance — so a
measurement taken on an inconsistent environment says so instead of looking like any other
number. It never blocks: refusing to start would put this control plane in the business of
adjudicating someone's virtualenv, and a false positive would take down a working GPU host
remotely.

What it does not do is prevent the divergence. If that matters on your host, point
`VLLMBENCH_VLLM_BIN` at vLLM in a separate environment and install the agent into its own.

**Windows is not supported.** Linux, or macOS with Colima, on the control host; Linux on
the GPU host.

## Product

**No automated tuning.** No search strategy, no Bayesian optimization, no early stopping.
The framework measures and displays; the human decides. This is a scope decision rather
than a missing feature.

**`vllm bench throughput` and `mm-processor` are not wrapped.** Online serving only.

**The UI assumes one person at a time.** Concurrent editing has no conflict handling
because there are no users to be in conflict.
