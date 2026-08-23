"""What went wrong with a run, in a form that can be counted.

``run.error`` has always carried the full story, and it stays the record of record. What
it cannot do is answer a question across many runs: a sweep with eleven failed points
gives an operator eleven walls of vLLM traceback and no way to see that nine of them are
one cause. Grouping free text is guesswork; grouping a column is a ``GROUP BY``.

So each failure also records a *kind*. Two rules keep the kind honest:

**The kind never replaces the message.** It is a lens for counting, not a summary. The
full text is always written alongside it, because the kind necessarily throws away the
detail an operator needs to act — "the engine ran out of memory" does not say how much
it wanted.

**Structural evidence beats text.** Most kinds are known from the control plane's own
position: it knows whether the connection failed, whether the token was refused, which
phase of the run it was in, and whether it was its own deadline that expired. Only one
distinction cannot be made that way — *why* an engine refused to start — because the
only witness is vLLM's own output. That one is matched against patterns taken from
payloads captured from a real vLLM (``tests/fixtures/vllm_serve_*.log``), never from
documentation, and an unrecognized failure stays at the general kind rather than being
guessed into a specific one. A wrong specific kind is worse than an honest general one:
it sends the reader to fix the thing it names.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum

from vllmbench_protocol.errors import (
    AgentAuthError,
    AgentError,
    AgentUnreachable,
    ProtocolMismatch,
)

log = logging.getLogger(__name__)


#: Response header by which an agent names the kind of failure it is reporting.
#:
#: A header rather than a field in the error body, and deliberately so. The agent holds
#: evidence the control plane never sees — the whole vLLM log rather than a tail, and its
#: own knowledge of which deadline expired — so it is the better classifier. But the
#: alternatives both cost something: a new wire model means a protocol version bump and a
#: redeployment to every GPU host, and restructuring the `{"detail": ...}` error body
#: would change what an older control plane displays. A header is invisible to a client
#: that does not read it and absent for an agent that does not send it, so old and new on
#: either side of the boundary keep working unchanged.
FAILURE_KIND_HEADER = "X-Vllmbench-Failure-Kind"


class FailureKind(StrEnum):
    """Why a run did not produce a measurement.

    Ordered roughly by where in a run's life it can occur, which is also the order in
    which an operator would investigate: reach the host, start the engine, run the
    benchmark, read the result.
    """

    #: The agent did not answer. Network, host down, agent not running.
    AGENT_UNREACHABLE = "agent_unreachable"
    #: The agent answered and refused the token.
    AGENT_AUTH = "agent_auth"
    #: The agent speaks a different protocol version. Never proceeds — see errors.py.
    PROTOCOL_MISMATCH = "protocol_mismatch"

    #: The engine could not allocate what it needed. Usually KV cache, and usually the
    #: interaction of `gpu-memory-utilization`, `max-model-len` and `max-num-seqs` —
    #: which is to say, usually a point in the sweep matrix rather than a broken host.
    ENGINE_OUT_OF_MEMORY = "engine_out_of_memory"
    #: vLLM rejected the configuration itself: an argument it does not have, or a value
    #: outside what the model allows. Actionable without touching the hardware.
    ENGINE_CONFIG_REJECTED = "engine_config_rejected"
    #: The engine died during startup for some other reason.
    ENGINE_LOAD_FAILED = "engine_load_failed"
    #: The engine started and stayed alive but never began serving within its budget.
    ENGINE_NOT_READY = "engine_not_ready"

    #: The engine came up, served, and then died. Distinct from every kind above, which
    #: are all failures to *start*: this one has a configuration that demonstrably works
    #: for a while, so the fix is never "check the config" and the run before it may be a
    #: perfectly good measurement. Seen for real on MTP drafting depths of 4 and above,
    #: where vLLM 0.25.1 corrupts memory and takes the engine down mid-benchmark.
    ENGINE_CRASHED = "engine_crashed"

    #: The GPU host is out of disk. Named separately from every engine failure because
    #: it is not a property of the configuration at all — the same config on the same
    #: card works again once space is freed, so re-running is the right response, and
    #: filing it under `engine_load_failed` would send the reader to tune a config that
    #: is fine.
    HOST_DISK_FULL = "host_disk_full"

    #: The benchmark client exceeded the time allowed for it and was killed.
    BENCHMARK_TIMEOUT = "benchmark_timeout"
    #: `vllm bench serve` ran and failed.
    BENCHMARK_FAILED = "benchmark_failed"

    #: The benchmark produced output that does not match the contract. This means vLLM
    #: changed its `--save-result` schema, not that the configuration was bad, and the
    #: fix is in this repository rather than on the host.
    RESULT_SCHEMA_MISMATCH = "result_schema_mismatch"

    #: A bug here, or a state we did not anticipate. Distinct from every kind above
    #: precisely so that it stays countable: a rising `internal` count is a defect
    #: report, and folding it into a plausible neighbour would hide that.
    INTERNAL = "internal"


# vLLM's own words for the failures we can name, each read off a captured payload rather
# than a document. The fixture that produced each is named beside it, and a test asserts
# every pattern still matches its fixture — so an upstream rewording fails a test instead
# of silently reclassifying every future failure as `engine_load_failed`.
_ENGINE_PATTERNS: tuple[tuple[re.Pattern[str], FailureKind, str], ...] = (
    (
        # vllm_serve_no_kv_cache_memory_v0.25.1.log. The CPU backend's wording says
        # `gpu_memory_utilization` even on CPU, which is why the match is on the first
        # sentence rather than the advice that follows it.
        re.compile(r"No available memory for the cache blocks", re.I),
        FailureKind.ENGINE_OUT_OF_MEMORY,
        "vllm_serve_no_kv_cache_memory_v0.25.1.log",
    ),
    (
        # torch's allocator, on a real GPU. Not reproducible on the CPU backend, so this
        # one pattern has no fixture of its own — it is matched loosely and deliberately
        # last-resort, and its consequence if wrong is a mislabelled OOM, not a lost run.
        re.compile(r"(?:torch\.)?(?:cuda\.)?OutOfMemoryError|CUDA out of memory", re.I),
        FailureKind.ENGINE_OUT_OF_MEMORY,
        "",
    ),
    (
        # vllm_serve_unrecognized_argument_v0.25.1.log — argparse rejects it before vLLM
        # starts at all, exit code 2.
        re.compile(r"^vllm: error: (?:unrecognized arguments|argument )", re.M),
        FailureKind.ENGINE_CONFIG_REJECTED,
        "vllm_serve_unrecognized_argument_v0.25.1.log",
    ),
    (
        # vllm_serve_max_model_len_rejected_v0.25.1.log — vLLM validates its config
        # objects with pydantic, so a value the model cannot support arrives as a
        # ValidationError naming the config class.
        re.compile(r"ValidationError: \d+ validation error for \w*Config", re.M),
        FailureKind.ENGINE_CONFIG_REJECTED,
        "vllm_serve_max_model_len_rejected_v0.25.1.log",
    ),
    (
        # vllm_serve_mtp_illegal_memory_access_v0.25.1.log — the engine had been serving
        # for a minute before this. Matched last so that an out-of-memory death, which
        # also kills the engine and also mentions CUDA, keeps its more specific kind: the
        # fix for one is a smaller batch and for the other is a bug report.
        #
        # Two alternatives, because they are two views of the same event. CUDA's wording
        # is what the worker raises; `EngineDeadError` is what vLLM tells the *next*
        # request, and on a crashed engine every subsequent request gets only that.
        re.compile(r"illegal memory access|EngineDeadError", re.I),
        FailureKind.ENGINE_CRASHED,
        "vllm_serve_mtp_illegal_memory_access_v0.25.1.log",
    ),
)


def classify_engine_output(text: str) -> FailureKind | None:
    """Name the cause of a failed engine start from vLLM's own output.

    Returns ``None`` when nothing matches, which the caller must treat as "the general
    kind", not as "no failure". Guessing here would attach a confident label to an
    unfamiliar failure and send the reader to fix something that is not broken.

    Order matters: memory exhaustion is checked before configuration rejection, because
    a configuration that asks for more KV cache than exists produces both a memory
    message and a great deal of config context, and the memory message is the actionable
    one.
    """
    if not text:
        return None
    for pattern, kind, _fixture in _ENGINE_PATTERNS:
        if pattern.search(text):
            return kind
    return None


def classify_agent_error(exc: AgentError, *, default: FailureKind) -> FailureKind:
    """Name a failure at the control plane ↔ agent boundary.

    Evidence in descending order of quality:

    1. **The error type.** Unreachable, refused token and protocol mismatch are raised
       by the client's own transport and handshake layers, so they need no reading.
    2. **The agent's own verdict**, when it sent one. It saw the whole vLLM log rather
       than a tail, and it knows which of its deadlines expired — facts that do not
       survive the trip. An unfamiliar value is ignored rather than trusted, so a newer
       agent naming a kind this build has never heard of degrades to the text below
       instead of failing.
    3. **The text.** A bare :class:`AgentError` carries the agent's detail verbatim,
       which for a failed model load is vLLM's own output.

    Falls back to the caller's phase-appropriate default when none of them says
    anything.
    """
    if isinstance(exc, AgentUnreachable):
        return FailureKind.AGENT_UNREACHABLE
    if isinstance(exc, AgentAuthError):
        return FailureKind.AGENT_AUTH
    if isinstance(exc, ProtocolMismatch):
        return FailureKind.PROTOCOL_MISMATCH

    reported = getattr(exc, "reported_kind", None)
    if reported:
        try:
            return FailureKind(reported)
        except ValueError:
            log.warning("agent reported unknown failure kind %r; classifying from text", reported)

    return classify_engine_output(str(exc)) or default


#: Kinds where retrying the same run against the same host could plausibly succeed
#: without anything being changed — the host was momentarily away, not wrong.
#:
#: This is *not* a retry list. Retry is deliberately unimplemented (ROADMAP 0.4.0: a
#: failed run is evidence, and re-running it silently would hide a reproducible failure
#: behind an eventual success). It exists so the orchestrator can tell "this host is
#: currently unreachable, stop pulling work for it" from "this configuration does not
#: work, keep going" — a distinction about the *host*, not about the measurement.
TRANSIENT_KINDS: frozenset[FailureKind] = frozenset(
    {FailureKind.AGENT_UNREACHABLE},
)
