# ADR 0001 — MCP server for agent-driven benchmarking

- **Status:** accepted
- **Issue:** #2
- **Accepted in:** #4
- **Graduated from proposal in:** #5
- **Milestone:** 0.6.0
- **Supersedes:** nothing

> Accepted. The decisions below are binding; changing one requires a superseding ADR, not
> an edit to this file. Rejected alternatives are retained deliberately — they are the
> record of why the project is not doing the obvious other thing.

---

## Summary

Expose the framework over the Model Context Protocol so Claude Code and comparable agent
harnesses can author configurations, launch sweeps, and analyze results through a
standards-based interface rather than screen-scraping a UI or hand-rolling HTTP calls.

Decided, in one line: **mount a Streamable HTTP MCP server at `/mcp` on the existing `api`
service, targeting the 2026-07-28 specification, authenticated by bearer token, shipping
read and write tools together.**

Scheduled as roadmap milestone **0.6.0**.

---

## Why this is a good fit

Parameter tuning is a search problem with a slow, expensive evaluation function. That is
close to the ideal shape for an agent loop: propose a configuration, measure it, read the
telemetry, propose a better one. The human bottleneck today is not judgment, it is the
tedium of authoring the next point in the matrix and waiting.

The framework already has the two things such a loop needs and a raw `vllm bench` CLI does
not: durable history, so an agent can see what has already been tried, and telemetry, so
it can reason about *why* a configuration lost rather than only *that* it lost.

---

## Protocol revision

Target the **2026-07-28** specification. Relevant properties:

- **Stateless protocol core.** Fits a containerized service behind a proxy with no session
  affinity.
- **Two standard transports** — stdio and Streamable HTTP. The legacy SSE transport is
  superseded and is not implemented.
- **`Mcp-Method` and `Mcp-Name` headers** mirror body fields so intermediaries can route
  and rate-limit without parsing JSON. Free observability: our own logs can report which
  tool was called without inspecting payloads.
- **Cacheable list results**, which matters for `list_configs` and similar, called
  repeatedly across a long agent session.
- **OAuth 2.1** as the authorization standard for remote servers.

---

## Decision 1 — Where the server lives

**Mounted on the existing `api` service at `/mcp`**, behind `VLLMBENCH_MCP_ENABLED` so the
entire surface can be switched off.

| Option | Assessment |
| --- | --- |
| **A. Mounted on `api` at `/mcp`** | **Chosen.** One deployment, one auth story, and — decisively — no way for the MCP surface to drift from the REST surface, because both call the same domain functions. |
| B. Separate `mcp` compose service | Rejected. Cleanest isolation and independently disableable, but adds a network hop, a second auth boundary, and duplicated DTOs. The isolation benefit is achievable in option A with a feature flag. |
| C. Local stdio server | Rejected. Contradicts the self-contained-stack goal and requires installation outside compose. Revisit only if a target client cannot do remote HTTP; Claude Code can. |

## Decision 2 — Authentication

**Bearer token**, reusing the `VLLMBENCH_TOKEN` pattern established for the agent.

| Option | Assessment |
| --- | --- |
| **Bearer token** | **Chosen for pre-1.0.** One environment variable, no authorization server to run. |
| OAuth 2.1 | Deferred, not rejected. Correct and standards-aligned — the spec's answer for remote servers — but requires an authorization server, discovery endpoints, and token lifecycle for a single-user LAN tool. Revisit at 1.0.0 or on the first request to reach the stack remotely. |

**This is a real constraint, not a footnote:** a bearer token is adequate on a trusted LAN
and inadequate on the public internet. The documentation must say plainly that `/mcp` is
not to be exposed beyond the local network, and the default bind must make accidental
exposure hard.

---

## Decision 3 — Input format: YAML or structured

This looked like one question and is actually two, because `create_config` and
`create_sweep` take fundamentally different things.

**A single server config is a vLLM artifact.** Invariant 5 settles it: native YAML,
validate don't transform. There is a second reason beyond the invariant — vLLM has
hundreds of flags that change between versions, so any structured schema enumerating them
begins rotting the day it is written and needs maintenance on every
`VLLM_REFERENCE_VERSION` bump. YAML passthrough never rots.

**A sweep matrix is not a vLLM artifact.** `{max_num_seqs: [32, 64, 128],
tensor_parallel_size: [1, 2]}` is not a config, it is a generator of configs, and the
concept is entirely ours. Expressing it as YAML would mean inventing a non-vLLM YAML
schema — precisely the conflation invariant 5 warns against. It would also forfeit the
single largest reliability lever available: **MCP tools declare JSON Schema for their
inputs.** A structured `axes` parameter constrains the model at the protocol level, for
free. A YAML string gets no such help.

| Tool | Input | Rationale |
| --- | --- | --- |
| `create_config` | raw YAML string | Lossless, invariant 5, immune to vLLM version drift |
| `create_sweep` | `base_config_id` + structured `axes` | Schema-constrained, and the matrix is our concept anyway |

### Why the override merge does not violate invariant 5

Each sweep point applies its axis overrides onto the base YAML and stores the **resulting
YAML**, which is content-addressed like any other config. The stored artifact is native
vLLM YAML that runs unmodified. That is generation, not translation: nothing is
round-tripped through an intermediate schema, and no fidelity is lost.

### A consequence: `validate_config` is a first-class tool

Models are unreliable at YAML — indentation, and type coercion such as `no` parsing as
boolean false — but they are good at fixing errors when told precisely what is wrong. A
free, safe, repeatable validation call gives them that loop. Validation must therefore be
available as a no-side-effect read tool, not only as a step buried inside `create_config`,
so the only way to check YAML is not to create something.

---

## Decision 4 — Cost estimation before a sweep runs

Estimates depend on **run history**, which begins accumulating at 0.2.0 — not on 0.5.0,
which delivers the analysis UI, a different thing.

Decomposing what an estimate actually is:

- **Rate-limited benchmark duration is arithmetic.** A run with `--num-prompts N
  --request-rate R` takes roughly N/R seconds regardless of hardware. No history needed.
- **Saturation runs are not.** `--request-rate inf --max-concurrency C` runs as fast as the
  configuration allows, which is the thing being measured. Genuinely unknown a priori.
- **Model load and server startup** need history, but are highly consistent per model and
  TP size on a given host. A few observations produce a good number.

Three rules follow.

**1. Exact counts always, from day one.** The genuinely useful pre-flight number is not
duration:

> This sweep is 96 configurations × 3 replicates = **288 runs**.

That is exact, needs zero history, and is what a human actually needs to decide whether to
approve. Duration is the refinement layered on top.

**2. The estimate that matters is the live one.** After a few points complete,
extrapolating from observed durations is accurate, needs no historical model, and is
available in every sweep regardless of accumulated history. `get_sweep` returns
`estimated_remaining` derived from completed points. This is where the engineering effort
belongs — not in a priori prediction.

**3. Never return a bare number.** A priori estimates are structured, carrying
`confidence`, `basis` (what data supported it, and how many samples), and a decomposition
into startup and benchmark components. A confidently wrong "about 20 minutes" on a sweep
that runs for nine hours is worse than an honest "unknown." Making the uncertainty
machine-readable lets the agent relay it instead of rounding it into a claim.

---

## Tool surface

Split by side effect, because the safety story depends on the split being visible rather
than implied.

### Read tools

| Tool | Returns |
| --- | --- |
| `list_hosts` | GPU hosts, device inventory, driver and vLLM versions |
| `list_configs` / `get_config` | vLLM YAML server configs, with lineage |
| `validate_config` | Structured, actionable errors for a candidate YAML. No side effects. |
| `list_workloads` | Workload definitions |
| `list_sweeps` / `get_sweep` | Definition, status, progress, points remaining, `estimated_remaining` |
| `query_runs` | Filtered, paginated run summaries |
| `get_run` | Full metrics for one run, with provenance |
| `get_run_telemetry` | Downsampled engine and per-device GPU series |
| `compare_runs` | Structured diff of configuration and metrics across runs |
| `get_pareto` | Frontier points for a sweep, per-GPU normalized |

### Write tools

Shipped in the first release alongside the read tools, and **enabled by default**. The
agent-driven tuning loop is the motivation for this work; shipping the write tools
defaulted off would satisfy the letter of that and not the intent.
`VLLMBENCH_MCP_WRITE_ENABLED` remains as an opt-out for deployments that want an
analysis-only surface.

| Tool | Effect |
| --- | --- |
| `create_config` | Validate and store a vLLM YAML config |
| `create_workload` | Store a workload definition |
| `create_sweep` | Define a matrix from a base config and structured axes. **Does not start it.** Returns exact run count and a structured duration estimate. |
| `start_sweep` | Begin execution. Returns immediately. |
| `cancel_sweep` | Stop a running sweep, tearing down cleanly |

**There is deliberately no tool that mutates or deletes a run, a result, or a telemetry
sample.** Runs are immutable once terminal, and that invariant should not have an
agent-shaped hole in it. Correcting bad data is a human operation against the database,
not a tool call.

### Resources

Configs and completed sweep reports are also exposed as MCP resources addressed by URI
(`vllmbench://config/{hash}`, `vllmbench://sweep/{id}/report`). Resources are the right
primitive for stable, cacheable, human-inspectable documents; tools are for actions.

---

## Design constraints specific to this framework

### Sweeps outlive tool calls

A sweep runs for hours. An MCP tool call is a request and a response. `start_sweep`
therefore returns as soon as the sweep is queued, and the agent polls `get_sweep`. No tool
call ever blocks on sweep completion — not with a long timeout, not with a streamed
progress hack. The separation between `create_sweep` and `start_sweep` exists for the same
reason it exists in the UI: the agent should be able to show a human what it is about to
spend two hours of GPU time on before it spends it.

### Context economy is a correctness concern

A single run has thousands of telemetry samples; a sweep has tens of runs. Returning that
raw into a model's context is expensive and actively unhelpful — it displaces the reasoning
the tokens were meant to fund.

Therefore, for every read tool:

- Paginate, with a bounded default page size and a hard maximum.
- Return summary fields by default; raw and per-sample data only on explicit opt-in.
- Downsample telemetry **server-side** to a requested resolution. Never return a full
  series and expect the client to thin it.
- Prefer returning a computed answer over returning the data to compute it from.
  `get_pareto` returning frontier points beats `query_runs` returning everything and hoping
  the model does the algebra correctly.

### Provenance extends to initiation

Invariant 6 requires every run to state what produced it. An agent-initiated sweep adds a
question the current provenance set cannot answer: *who started this, and why did it run
at 3am?* Sweeps and runs gain an `initiated_by` field recording the interface (`ui`,
`mcp`, `api`) and the MCP client identity where available.

### Existing invariants that constrain this

- **Invariant 7 (synthetic quarantine)** applies unchanged. A sweep an agent starts against
  the mock agent is flagged synthetic at creation, exactly as a UI-initiated one is. The
  MCP path must not become a way to launder synthetic results into real ones.
- **Invariant 8 (per-GPU normalization)** applies to every throughput figure a tool
  returns. An agent comparing raw aggregate throughput across tensor-parallel sizes will
  reach a confidently wrong conclusion, and unlike a human looking at a chart it has no
  axis label to warn it.

---

## Guardrails

Writes ship enabled, so the remaining guardrails carry more weight than they would have
under a read-only-first rollout. They are load-bearing, not decorative.

1. **One active sweep per GPU host**, enforced in the domain layer rather than the MCP
   layer, so the constraint holds regardless of which interface asks.
2. **Bounded matrix size.** Reject sweep definitions above a configurable run count. An
   agent that accidentally requests a 4,000-point Cartesian product gets an error, not a
   fortnight of GPU time.
3. **`create_sweep` reports exact cost** — run count always, duration estimate with
   explicit confidence — so an agent, and the human reading its output, sees the
   commitment before `start_sweep`.
4. **No tool mutates or deletes results.** Worth restating as a guardrail and not only as a
   surface decision: the worst realistic outcome of an agent misusing this interface is
   wasted GPU time, never lost measurements.
5. **Every write tool call is logged** with its arguments and initiating client.
6. **Writes can be disabled** via `VLLMBENCH_MCP_WRITE_ENABLED` for analysis-only
   deployments.

---

## Decision 5 — Milestone placement

**Milestone 0.6.0**, inserted after Analysis, with later milestones renumbered.

| Option | Assessment |
| --- | --- |
| **New milestone after 0.5.0** | **Chosen.** The valuable analysis tools depend on 0.5.0 and the control tools on 0.4.0. MCP is a distinct interface with its own surface and safety story; hiding it inside another milestone understates it. |
| Fold into Interop | Rejected. Thematically defensible, but it would dominate a milestone otherwise made of importers and exporters. |
| Defer to post-1.0 | Rejected. Contradicts the pre-1.0.0 requirement in #2. |

---

## Scope of this decision

This ADR binds the placement, transport, authentication, input formats, estimation
strategy, read/write split, and guardrails above. Exact tool names and argument shapes
will firm up during implementation and do not require a superseding ADR.
