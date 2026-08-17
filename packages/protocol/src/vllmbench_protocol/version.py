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

__version__ = "0.1.0"

# 3: BenchRequest carries `served_model_name` separately from `model`. An agent still on
#    2 receives only `model` and passes it as `--model`, which is what vLLM loads the
#    *tokenizer* from — so an alias there silently tokenizes with the wrong tokenizer and
#    records wrong input-token counts. That is a data-corruption difference, not a
#    feature difference, which is exactly what this number exists to refuse.
# 4: BenchResponse carries telemetry, and BenchRequest can set the sampling interval.
#    An agent on 3 returns no samples at all, which a control plane on 4 would record as
#    a run that was genuinely never sampled rather than one whose agent cannot sample.
PROTOCOL_VERSION = 4
