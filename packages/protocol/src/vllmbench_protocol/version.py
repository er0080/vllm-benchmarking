"""Version and protocol-compatibility constants.

Two different numbers live here and they answer different questions.

``__version__`` is the release version of the whole project. Every package in the
workspace carries the same value — the agent and the control plane are versioned in
lockstep — and ``scripts/check_versions.py`` enforces that against the root ``VERSION``
file.

``PROTOCOL_VERSION`` is the compatibility number for the agent's HTTP API. It increments
only when a change would break an older counterpart, which is far less often than
``__version__`` changes. The control plane refuses to talk to an agent reporting a
different ``PROTOCOL_VERSION`` — see CLAUDE.md, "Versioning". That refusal is deliberate:
a stale agent must fail loudly at connect time rather than produce subtly wrong data
halfway through a sweep.

Note the contrast with the *vLLM* version, which is handled in the opposite way. A vLLM
version mismatch between a GPU host and ``VLLM_REFERENCE_VERSION`` is recorded and warned
about but never blocks, because benchmarking one vLLM version against another is a
supported use of this tool.
"""

from __future__ import annotations

__all__ = ["PROTOCOL_VERSION", "__version__"]

__version__ = "1.0.0"

# 3: BenchRequest carries `served_model_name` separately from `model`. An agent still on
#    2 receives only `model` and passes it as `--model`, which is what vLLM loads the
#    *tokenizer* from — so an alias there silently tokenizes with the wrong tokenizer and
#    records wrong input-token counts. That is a data-corruption difference, not a
#    feature difference, which is exactly what this number exists to refuse.
# 4: BenchResponse carries telemetry, and BenchRequest can set the sampling interval.
#    An agent on 3 returns no samples at all, which a control plane on 4 would record as
#    a run that was genuinely never sampled rather than one whose agent cannot sample.
# 5: the agent can stop a benchmark in flight (POST /bench/cancel). Without it a
#    cancelled sweep keeps burning GPU time until the current point finishes, and the
#    only alternative — killing the engine underneath a running client — leaves that
#    client thrashing against a dead socket.
# 6: HostInfo carries `environment`, the agent's report on whether its own virtualenv
#    satisfies the constraints everything installed there declares. Additive on the wire
#    and still a bump, because `_Wire` forbids unknown fields: a 6 agent talking to a 5
#    control plane would have its whole host-info payload rejected on an extra key, which
#    is a worse failure than the mismatch being named. It is also a provenance change —
#    invariant 6 — and a run that cannot say whether it was measured on a coherent
#    environment is missing something a reader would want.
# 7: three provenance facts a run could not previously state about itself, all of them
#    only knowable on the GPU host (invariant 1), so all of them arriving on the wire.
#    `BenchResponse` and `ServerStatus` carry `speculative_method` and
#    `speculative_tokens`, read from the engine's own `/server_info` rather than parsed
#    back out of the config YAML — invariant 8's rule for parallelism topology, applied
#    to speculation for the same reason: the YAML can say `num_speculative_tokens: 3`
#    while the engine runs without a drafter. `BenchResponse` also carries
#    `dataset_identity`, which invariant 6 has required since the first schema and which
#    was NULL on every run this project had produced.
#
#    Additive again, and a bump again, for the reason 6 was: `_Wire` forbids unknown
#    fields, so a 7 agent talking to a 6 control plane has its whole benchmark result
#    rejected on an extra key. Losing a forty-minute run to a version mismatch that was
#    never named is exactly what this number exists to prevent.
PROTOCOL_VERSION = 7
