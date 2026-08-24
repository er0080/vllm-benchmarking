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

__version__ = "1.1.0"

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
# 8: HostInfo carries a host-wide `peer_access`, and BenchResponse carries one scoped to
#    the devices the benchmark actually used. Together they answer a question no run could
#    previously answer about itself: which interconnect was underneath it.
#
#    This one is worth stating plainly, because it is the case invariant 6 was written
#    for and the schema had no room for it. Enabling peer-to-peer DMA on consumer GPUs
#    is done by replacing the kernel module with a patched build of the *same version*,
#    so the driver reports 610.43.02 either way. Every provenance field a run carries —
#    driver, CUDA, GPU model, vLLM version, parallelism topology, device indices — is
#    byte-identical across that change, while what a tensor-parallel run measures is not.
#    Two populations, one series, nothing to group or warn on.
#
#    Scoped to the run's own devices rather than the host's, because a single-device run
#    has no peer access to report and saying "unsupported" would split a TP=1 control
#    across a boundary it cannot be on either side of. The host-wide value stays on
#    HostInfo, where an operator checking their setup will look for it.
#
#    Additive, and a bump, for the reason 6 and 7 were: `_Wire` forbids unknown fields.
# 9: `ServerStatus` and `BenchResponse` carry `engine_env` — the subset of the engine
#    subprocess's launch environment that can change what it measures.
#
#    8 recorded the interconnect a run was measured over. This records the settings that
#    decide what the engine does with it, which turn out to be the same class of gap and
#    a wider one. `NCCL_P2P_LEVEL=SYS` moved per-GPU throughput 13% on this project's own
#    GPU host, and `VLLM_CUSTOM_ALLREDUCE_PUSH` selects an entirely different all-reduce
#    kernel; neither appears in the config YAML, so neither reaches the config hash. Runs
#    differing in either are byte-identical in every column the schema had — the same
#    "two populations, one series" that 8 was written for.
#
#    Prefix-matched rather than enumerated, because the variable that forced this did not
#    exist when the list would have been written. See `ENGINE_ENV_PREFIXES`.
#
#    Observed, not reconstructed: the recorded mapping is filtered from the very dict
#    handed to `subprocess.Popen`, not rebuilt from the agent's own environment
#    afterwards. Invariant 8's rule — provenance is never inferred after the fact — with
#    the agent's environment being exactly the thing that could have drifted in between.
#
#    Secret-looking names keep their key and lose their value. Whether `VLLM_API_KEY` was
#    set changes how the engine behaves and is provenance; what it was is a secret, and
#    this payload is persisted to a database and returned by a JSON API.
#
#    Additive, and a bump, for the reason 6, 7 and 8 were: `_Wire` forbids unknown fields.
PROTOCOL_VERSION = 9
