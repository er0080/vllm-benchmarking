# The MCP interface

The control plane speaks [MCP](https://modelcontextprotocol.io) at `/mcp`, so an agent can
read measurements and author sweeps through the same code the UI uses. It is off by
default and behind a bearer token.

[ADR 0001](adr/0001-mcp-server-interface.md) records *why* this surface looks the way it
does. This document is how to turn it on and what it exposes.

The design constraint worth stating first, because it explains several decisions below:
**for an agent, the schema is the documentation.** Nothing reads a guide before calling a
tool. So every parameter carries a description, every closed set of values is published as
an enum, and an unrecognised value is refused rather than replaced with a default — a tool
that reports success while answering a different question is the one failure an agent
cannot detect.

---

## Turning it on

Four settings, all in `.env`:

| Variable | Default | What it does |
| --- | --- | --- |
| `VLLMBENCH_MCP_ENABLED` | `false` | Mounts the surface. Nothing is served at `/mcp` until this is true. |
| `VLLMBENCH_MCP_TOKEN` | *empty* | The bearer token clients present. **Empty refuses every request**, so an enabled-but-unconfigured server is inert rather than open. |
| `VLLMBENCH_MCP_WRITE_ENABLED` | `true` | Whether the five write tools work. False leaves them listed and refusing, and each refusal is recorded. |
| `VLLMBENCH_MCP_ALLOWED_HOSTS` | `[]` | `Host` headers to accept, as a JSON list. Empty disables DNS-rebinding protection, which is the default and is deliberate — see below. |

`VLLMBENCH_MCP_TOKEN` is **not** `VLLMBENCH_TOKEN`. That one is what the control plane
presents to the agent on the GPU host; this one is what an MCP client presents to the
control plane. They are different directions and should be different secrets.

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'   # generate one
docker compose up -d api                                        # apply
```

If you are upgrading a stack whose `.env` predates this feature, compare it against
`.env.example` rather than assuming — a `.env` copied before 0.6.0 has no
`VLLMBENCH_MCP_TOKEN` line at all, and an absent variable reads as an empty one, which
refuses every call.

---

## Pointing a client at it

Claude Code, scoped to the current project so the token stays out of the repository:

```bash
claude mcp add --transport http --scope local vllmbench \
  http://localhost:8000/mcp \
  --header "Authorization: Bearer $VLLMBENCH_MCP_TOKEN"
```

Any other client, as configuration:

```json
{
  "mcpServers": {
    "vllmbench": {
      "type": "http",
      "url": "http://localhost:8000/mcp",
      "headers": { "Authorization": "Bearer <VLLMBENCH_MCP_TOKEN>" }
    }
  }
}
```

Do not use a project scope, `.mcp.json`, or anything else that gets committed: it would
put the token in git.

**A client added mid-session is not visible to that session.** Servers are read at
startup, so restart the client before concluding the URL is wrong.

Use `http://<host>:8000/mcp` with no trailing slash. Both work — the mount answers the
bare path directly rather than redirecting to it, because MCP clients do not follow
redirects and the failure would surface as `Unexpected content type` rather than as a 307.

### Reaching it from another machine

The token is the access control. Rebinding protection is off by default because there is
no host value the stack could guess: the control plane is reached by LAN address, and a
non-empty allowlist that does not contain the address a client actually uses rejects every
request with `Invalid Host header`. If you set it, set it to what the client sends,
including the port:

```bash
VLLMBENCH_MCP_ALLOWED_HOSTS=["10.0.0.5:8000"]
```

That protection defends a *browser* tricked into calling a local service. It is not what
guards this surface, and enabling it is not a substitute for keeping `/mcp` off the public
internet.

### Checking it works

```bash
curl -s -X POST http://localhost:8000/mcp \
  -H "Authorization: Bearer $VLLMBENCH_MCP_TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

Nineteen tools, or `401` if the token is wrong. The same request without the header must
also be `401`; if it is not, `VLLMBENCH_MCP_ENABLED` is false and you are looking at the
API's 404.

---

## The tools

Nineteen, split by what they touch. Every one is annotated with `readOnlyHint` /
`destructiveHint` / `idempotentHint`, so a harness can decide what to auto-approve without
reading this table.

### Inventory

| Tool | |
| --- | --- |
| `list_hosts` | GPU hosts this control plane knows about, and what each last reported. |
| `list_configs` | Server configurations, newest first. YAML omitted — fetch it deliberately. |
| `get_config` | One configuration's exact YAML. |
| `get_config_lineage` | Where a configuration came from, and what came from it. |
| `list_workloads` | Benchmark workloads — the traffic each run was measured under. |
| `list_sweeps` | Sweeps, newest first, with run counts by status. |
| `get_sweep` | One sweep, including how far through it is. |
| `server_info` | Version, protocol version, whether writes are on, and what this surface will not do. |

### Measurements

| Tool | |
| --- | --- |
| `query_runs` | Recent runs with their headline metrics and the provenance behind them. |
| `get_run` | One run in full, including every flattened metric. |
| `get_pareto` | Measurement points, partitioned into sets that may honestly be compared. |
| `compare_runs` | Two measurement points side by side, with a diff of their configurations. |
| `get_run_telemetry` | Engine and per-device series for one run, thinned by stride. |
| `validate_config` | Check a configuration against a host's vLLM before spending GPU time on it. |

### Writes

Gated by `VLLMBENCH_MCP_WRITE_ENABLED` and audited individually.

| Tool | |
| --- | --- |
| `create_config` | Store a vLLM server configuration, exactly as given. |
| `annotate_config` | Record why a configuration is worth keeping. Never edits the YAML. |
| `create_workload` | Define the traffic a run is measured under. |
| `create_sweep` | Author a sweep. Every run is created immediately, in execution order. |
| `cancel_sweep` | Stop a sweep. Finished runs keep their results. |

List pages default to 25 rows and are capped at 100 however large a `limit` is passed. An
agent asking for everything is usually defaulting rather than choosing, and a list that
grows without bound is how a long session spends its context on its least interesting
data.

---

## Resources

Two, both **templated**:

| URI template | Type | |
| --- | --- | --- |
| `vllmbench://config/{config_hash}` | `text/yaml` | The configuration text itself, byte for byte, with no JSON envelope — hand it straight to a host. |
| `vllmbench://sweep/{sweep_id}/report` | `text/markdown` | What one sweep measured, partitioned the same way every chart is. |

Templated resources are advertised under **`resources/templates/list`**, not
`resources/list`. `resources/list` is empty here and that is correct, not a bug — it lists
concrete resources, and there are none. This has already misled one reader into reporting
the feature as unimplemented, which is why `server_info` names both templates.

---

## What the surface guarantees

These are the repository's invariants as they appear to a caller, not extra rules for
agents:

- **One population per question.** Every tool returning measurements takes `source`, which
  is `real` or `synthetic`. There is no value meaning both. Synthetic runs — the mock agent
  and the CPU backend — are quarantined at creation and never appear beside real ones.
- **Comparable sets, not one list.** `get_pareto` returns groups, each one host, GPU model,
  vLLM version and bench-client location. Points from different groups are not a series,
  which is why they do not arrive as one.
- **Per-GPU as well as aggregate.** A tensor-parallel run out-throughputs a narrower one
  trivially while possibly being worse per device, so both figures travel together and
  comparisons default to per-GPU.
- **Runs are immutable once terminal.** No tool mutates or deletes a run. `cancel_sweep`
  moves *queued* runs to cancelled; it does not touch a measurement.
- **Every write says it came from here.** Rows created through MCP record
  `initiated_by="mcp"` plus the client's identity, so a result can always state which
  interface asked for it.
- **Unknown values are refused.** An unrecognised `source` or Pareto axis is an error
  naming the valid ones, never a silent substitution. The HTTP endpoint underneath does
  substitute for an unrecognised axis, deliberately — a person reads the axis label and
  sees it immediately, and a bookmarked URL should still draw a chart.

### The audit log

Every write call appends a row: tool, arguments as they arrived, outcome
(`succeeded` / `refused` / `failed`), and what it acted on. Refusals are recorded too,
including those caused by writes being switched off — an agent bouncing repeatedly off a
read-only control plane is something an operator should be able to see.

Nothing is redacted, because nothing reaching these tools is a secret: the bearer token is
consumed by the transport and never appears as a tool argument.

---

## Security posture

A shared bearer token over HTTP, on a LAN, behind a feature flag. That is the whole of it.

Adequate for one operator, one control plane and one token in an environment variable. Not
adequate on the public internet, and not intended to be: there is no per-client identity,
no scoping beyond the read/write switch, and no transport encryption unless you put one in
front of it. OAuth 2.1 is the standard for remote MCP servers and is post-1.0 work here.

Do not expose `/mcp` beyond your network.
