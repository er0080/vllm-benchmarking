# Proposal 0001 — MCP server for agent-driven benchmarking

- **Status:** proposed
- **Issue:** #2
- **Target:** pre-1.0.0
- **Supersedes:** nothing
- **On acceptance:** graduates to `docs/adr/` and the roadmap is amended

---

## Summary

Expose the framework over the Model Context Protocol so Claude Code and comparable agent
harnesses can author configurations, launch sweeps, and analyze results through a
standards-based interface rather than screen-scraping a UI or hand-rolling HTTP calls.

Recommendation, in one line: **mount a Streamable HTTP MCP server at `/mcp` on the
existing `api` service, with a bearer token, a summary-first tool surface, and writes
disabled by default.**

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
  superseded and should not be implemented.
- **`Mcp-Method` and `Mcp-Name` headers** mirror body fields so intermediaries can route
  and rate-limit without parsing JSON. Free observability: our own logs can report which
  tool was called without inspecting payloads.
- **Cacheable list results**, which matters for `list_configs` and similar, called
  repeatedly across a long agent session.
- **OAuth 2.1** as the authorization standard for remote servers.

---

## Option 1 — Where the server lives

| Option | Description | Assessment |
| --- | --- | --- |
| **A. Mounted on `api` at `/mcp`** | MCP is a second protocol over the same domain layer, in the same process. | **Recommended.** One deployment, one auth story, and — decisively — no way for the MCP surface to drift from the REST surface, because both call the same functions. |
| B. Separate `mcp` compose service | Standalone service calling `api` over HTTP. | Cleanest isolation and independently disableable, but adds a network hop, a second auth boundary, and duplicated DTOs. The isolation benefit is achievable in option A with a feature flag. |
| C. Local stdio server | User runs a stdio binary via `uvx`; it talks to the API. | Contradicts the self-contained-stack goal and requires installation outside compose. Worth revisiting only if a target client cannot do remote HTTP; Claude Code can. |

Option A, with the whole MCP surface behind `VLLMBENCH_MCP_ENABLED` so it can be switched
off entirely.

## Option 2 — Authentication

The spec answer for remote servers is OAuth 2.1, with the server as a Resource Server
advertising an authorization server via `.well-known` endpoints. That is the right answer
for a multi-tenant public server and the wrong answer for this one, at least now.

| Option | Assessment |
| --- | --- |
| **Bearer token** | **Recommended pre-1.0.** Reuses the `VLLMBENCH_TOKEN` pattern already established for the agent. One environment variable, no authorization server to run. |
| OAuth 2.1 | Correct and standards-aligned, but requires an authorization server, discovery endpoints, and token lifecycle for a single-user LAN tool. Defer until there is a second user or an exposure requirement. |

**This is a real constraint, not a footnote:** a bearer token is adequate on a trusted LAN
and inadequate on the public internet. The documentation must say plainly that `/mcp` is
not to be exposed beyond the local network, and the default bind should make accidental
exposure hard. Revisit at 1.0.0 or on the first request to reach the stack remotely.

---

## Tool surface

Split by side effect, because the safety story depends on the split being visible rather
than implied.

### Read tools — always available

| Tool | Returns |
| --- | --- |
| `list_hosts` | GPU hosts, device inventory, driver and vLLM versions |
| `list_configs` / `get_config` | vLLM YAML server configs, with lineage |
| `list_workloads` | Workload definitions |
| `list_sweeps` / `get_sweep` | Sweep definitions, status, progress, points remaining |
| `query_runs` | Filtered, paginated run summaries |
| `get_run` | Full metrics for one run, with provenance |
| `get_run_telemetry` | Downsampled engine and per-device GPU series |
| `compare_runs` | Structured diff of configuration and metrics across runs |
| `get_pareto` | Frontier points for a sweep, per-GPU normalized |

### Write tools — disabled by default

| Tool | Effect |
| --- | --- |
| `create_config` | Validate and store a vLLM YAML config |
| `create_workload` | Store a workload definition |
| `create_sweep` | Define a matrix. **Does not start it.** Returns point count and duration estimate. |
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

1. **Writes are opt-in.** `VLLMBENCH_MCP_WRITE_ENABLED` defaults to false. Read-only is the
   safe and genuinely useful default — analysis is most of the value.
2. **One active sweep per GPU host**, enforced in the domain layer rather than the MCP
   layer, so the constraint holds regardless of which interface asks.
3. **`create_sweep` returns a cost estimate** — point count and estimated duration — so an
   agent, and the human reading its output, sees the commitment before `start_sweep`.
4. **Bounded matrix size.** Reject sweep definitions above a configurable point count. An
   agent that accidentally requests a 4,000-point Cartesian product should get an error,
   not a fortnight of GPU time.
5. **Every write tool call is logged** with its arguments and initiating client.

---

## Milestone placement

The valuable tools are the analysis ones, which depend on 0.5.0. The control tools depend
on 0.4.0. So the earliest coherent placement is after 0.5.0.

| Option | Assessment |
| --- | --- |
| **New milestone after 0.5.0**, renumbering later milestones | **Recommended.** MCP is a distinct interface with its own surface and safety story; hiding it inside another milestone understates it. |
| Fold into 0.7.0 Interop | Thematically defensible — it is interoperability — but it would dominate a milestone otherwise made of importers and exporters. |
| Defer to post-1.0 | Contradicts the issue's pre-1.0.0 requirement. |

Renumbering is deliberately left out of this PR. It is a mechanical edit best made once
the proposal is accepted, and doing it here would conflict with #3.

---

## Open questions

1. **Read-only or read-write at first ship?** The proposal defaults writes off. Shipping
   read-only first and adding writes once the read surface has been exercised is the
   lower-risk sequence, but it delays the agent-driven-tuning loop that motivates the
   issue.
2. **Should `create_sweep` accept raw YAML, or structured parameters?** Raw YAML honors
   invariant 5 and stays lossless. Structured parameters are easier for a model to get
   right and easier to validate. Possibly both, with YAML as the stored form.
3. **Is a duration estimate feasible before 0.5.0?** It needs historical run times per
   config class. Without history it is a guess, and a confidently wrong estimate may be
   worse than none.

---

## What acceptance means

Accepting this proposal means agreeing to the placement (option A), the transport and auth
choices, the read/write split, and the guardrails. It does not commit to the exact tool
names, which will firm up during implementation.
