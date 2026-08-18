"""Classifying failures, against output captured from a real vLLM.

Every pattern in :mod:`vllmbench_protocol.failures` is checked against the payload it was
read off, for the same reason the flattening tests use a captured `--save-result` file: a
hand-written sample encodes the author's belief about the wording, which is the belief
under test. The three fixtures here were produced by running the pinned
``vllm/vllm-openai-cpu`` image until it refused to start, in three different ways.

The other half of these tests is about restraint. A classifier that guesses is worse than
one that shrugs, because a confident wrong kind sends the reader to fix something that is
not broken — so "unrecognized stays unrecognized" gets as much attention here as the
matches do.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vllmbench_protocol.errors import (
    AgentAuthError,
    AgentError,
    AgentUnreachable,
    ProtocolMismatch,
)
from vllmbench_protocol.failures import (
    _ENGINE_PATTERNS,
    FailureKind,
    classify_agent_error,
    classify_engine_output,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(errors="replace")


class TestPatternsMatchRealOutput:
    """The test that fails when upstream rewords a message.

    Without it, a rename in vLLM silently reclassifies every future memory failure as
    `engine_load_failed` — which is not wrong, exactly, but quietly removes the
    distinction the column exists to draw.
    """

    @pytest.mark.parametrize(
        ("fixture", "expected"),
        [
            (
                "vllm_serve_no_kv_cache_memory_v0.25.1.log",
                FailureKind.ENGINE_OUT_OF_MEMORY,
            ),
            (
                "vllm_serve_unrecognized_argument_v0.25.1.log",
                FailureKind.ENGINE_CONFIG_REJECTED,
            ),
            (
                "vllm_serve_max_model_len_rejected_v0.25.1.log",
                FailureKind.ENGINE_CONFIG_REJECTED,
            ),
        ],
    )
    def test_a_captured_failure_is_named_correctly(
        self, fixture: str, expected: FailureKind
    ) -> None:
        assert classify_engine_output(_fixture(fixture)) is expected

    def test_every_pattern_that_claims_a_fixture_still_matches_it(self) -> None:
        """No pattern may name a fixture it no longer matches.

        Catches the case where a pattern is edited to fix one payload and quietly stops
        matching the one it was written for.
        """
        for pattern, _kind, fixture in _ENGINE_PATTERNS:
            if not fixture:
                continue
            assert pattern.search(_fixture(fixture)), (
                f"{pattern.pattern} no longer matches {fixture}"
            )

    def test_the_memory_failure_is_found_under_its_traceback(self) -> None:
        """The reason the agent collects root causes separately, restated as an assertion.

        vLLM prints the worker's real exception, unwinds through a hundred lines of outer
        traceback, and signs off with "See root cause above". The last lines of this
        fixture say only that engine core initialization failed — a classifier reading a
        tail would learn nothing.
        """
        whole = _fixture("vllm_serve_no_kv_cache_memory_v0.25.1.log")
        tail = "\n".join(whole.splitlines()[-25:])

        assert classify_engine_output(whole) is FailureKind.ENGINE_OUT_OF_MEMORY
        assert "Engine core initialization failed" in tail
        assert classify_engine_output(tail) is None

    def test_a_gpu_out_of_memory_is_recognized(self) -> None:
        """torch's own wording, which the CPU backend cannot produce.

        The only pattern here without a fixture of its own, and flagged as such in the
        source. Two spellings because vLLM surfaces both depending on where the
        allocation failed.
        """
        for text in (
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
            "RuntimeError: CUDA out of memory.",
        ):
            assert classify_engine_output(text) is FailureKind.ENGINE_OUT_OF_MEMORY


class TestRestraint:
    def test_an_unfamiliar_failure_is_not_guessed_at(self) -> None:
        assert classify_engine_output("Segmentation fault (core dumped)") is None
        assert classify_engine_output("") is None

    def test_the_caller_supplies_the_fallback(self) -> None:
        """`None` means "you decide", and the caller knows the phase.

        A run that failed while starting an engine and a run that failed while
        benchmarking produce different defaults from the same unrecognized text, which is
        the point: the phase is evidence the text does not carry.
        """
        exc = AgentError("agent at http://h returned 409: Segmentation fault")
        assert (
            classify_agent_error(exc, default=FailureKind.ENGINE_LOAD_FAILED)
            is FailureKind.ENGINE_LOAD_FAILED
        )
        assert (
            classify_agent_error(exc, default=FailureKind.BENCHMARK_FAILED)
            is FailureKind.BENCHMARK_FAILED
        )


class TestBoundaryErrors:
    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (AgentUnreachable("http://h", "connect refused"), FailureKind.AGENT_UNREACHABLE),
            (AgentAuthError("http://h"), FailureKind.AGENT_AUTH),
            (ProtocolMismatch("http://h", 2, 5), FailureKind.PROTOCOL_MISMATCH),
        ],
    )
    def test_the_error_type_decides_where_it_can(
        self, exc: AgentError, expected: FailureKind
    ) -> None:
        """Structural evidence, needing no interpretation.

        These are raised by the client's own transport and handshake layers, so the kind
        is known without reading anything — and must not be overridden by whatever text
        happens to be in the message.
        """
        assert classify_agent_error(exc, default=FailureKind.INTERNAL) is expected


class TestAgentReportedKind:
    """The agent's own verdict, when it sent one.

    It saw the whole vLLM log rather than the tail that fits in a response, and it knows
    which of its own deadlines expired. Neither survives the trip, so where it has an
    opinion it outranks anything read out of the text here.
    """

    def test_it_outranks_the_text(self) -> None:
        exc = AgentError("agent at http://h returned 422: benchmark exceeded its 3600s timeout")
        exc.reported_kind = FailureKind.BENCHMARK_TIMEOUT.value

        assert (
            classify_agent_error(exc, default=FailureKind.BENCHMARK_FAILED)
            is FailureKind.BENCHMARK_TIMEOUT
        )

    def test_an_agent_that_says_nothing_falls_back_to_the_text(self) -> None:
        """Older agents do not send the header at all, and must keep working.

        This is what buys the header its keep over a new wire field: no protocol bump, no
        redeployment to every GPU host, and a control plane that is simply a little less
        precise against an agent that has not been updated.
        """
        exc = AgentError(
            "agent at http://h returned 409: ValueError: No available memory for the cache blocks"
        )
        assert exc.reported_kind is None
        assert (
            classify_agent_error(exc, default=FailureKind.ENGINE_LOAD_FAILED)
            is FailureKind.ENGINE_OUT_OF_MEMORY
        )

    def test_a_kind_this_build_has_never_heard_of_is_ignored(self) -> None:
        """Forward compatibility, in the direction that actually happens.

        A newer agent naming a kind this control plane does not know must degrade to
        reading the text, not raise — the alternative is a control plane that cannot
        record failures from a host it can otherwise talk to.
        """
        exc = AgentError(
            "agent at http://h returned 409: ValueError: No available memory for the cache blocks"
        )
        exc.reported_kind = "engine_ate_the_kv_cache"

        assert (
            classify_agent_error(exc, default=FailureKind.ENGINE_LOAD_FAILED)
            is FailureKind.ENGINE_OUT_OF_MEMORY
        )


class TestUnreachableSaysSomething:
    """The most likely failure of a first install, and it used to say nothing.

    `httpx.ConnectTimeout` stringifies to the empty string, which left the message ending
    in a bare colon — the one place a diagnosis would go, blank.
    """

    def test_a_silent_transport_error_still_names_its_kind(self) -> None:
        import httpx

        from vllmbench_protocol import AgentClient

        client = AgentClient("http://10.255.255.1:9110", "a-token-long-enough")
        exc = client._unreachable(httpx.ConnectTimeout(""))

        assert "is unreachable: ConnectTimeout" in str(exc)
        assert "is unreachable: ." not in str(exc)

    def test_it_says_what_to_check(self) -> None:
        from vllmbench_protocol import AgentUnreachable

        message = str(AgentUnreachable("http://host:9110", "Connection refused"))

        assert "Connection refused" in message
        assert "agent is running" in message
        assert "reach them" in message
