"""Flattening tests, against a payload captured from a real vLLM.

CLAUDE.md names this the highest-consequence code in the repository, so it gets the most
thorough tests. The fixture is not hand-written: it is the verbatim output of
``vllm bench serve --save-result`` on vLLM 0.25.1. Hand-written fixtures encode the
author's belief about the format, which is exactly the belief that was wrong.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vllmbench_protocol.bench_result import (
    REQUIRED_FIELDS,
    SUMMARY_FIELD_MAP,
    BenchResultError,
    EmptyBenchResult,
    flatten_bench_result,
)

FIXTURE = Path(__file__).parent / "fixtures" / "bench_serve_v0.25.1.json"

#: A payload from an engine that was speculating — Qwen3.8-27B-FP8 with MTP at depth 3 on
#: 2x RTX 3090. Captured, not written: the speculative block only appears when speculation
#: is on, so the ordinary fixture cannot prove anything about it.
SPECULATIVE_FIXTURE = Path(__file__).parent / "fixtures" / "bench_serve_speculative_v0.25.1.json"

#: Sources that exist only when the engine speculates. Everything else in the map must be
#: present in every payload.
SPECULATIVE_SOURCES = frozenset(k for k in SUMMARY_FIELD_MAP if k.startswith("spec_decode_"))


@pytest.fixture
def payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def speculative_payload() -> dict[str, Any]:
    return json.loads(SPECULATIVE_FIXTURE.read_text())


class TestFieldMapMatchesReality:
    def test_every_mapped_source_field_exists_in_a_real_payload(
        self, payload: dict[str, Any]
    ) -> None:
        """The test that would have caught the original mistake.

        The published docs describe ``successful_requests`` / ``benchmark_duration_sec``
        / ``ttft_ms_p99``. vLLM emits ``completed`` / ``duration`` / ``p99_ttft_ms``.
        Mapping the documented names would have written NULL for every metric while
        reporting success.
        """
        missing = set(SUMMARY_FIELD_MAP) - payload.keys()
        # Asserted as an equality rather than a subset. The speculative block is absent
        # from this payload legitimately, but anything *else* going missing is the failure
        # this test exists for, and a `<=` would swallow it.
        assert missing == set(SPECULATIVE_SOURCES), (
            f"mapped fields absent from a real payload: {sorted(missing - SPECULATIVE_SOURCES)}"
        )

    def test_the_speculative_block_exists_when_speculating(
        self, speculative_payload: dict[str, Any]
    ) -> None:
        """The other half: those fields are real, not invented from the docs."""
        missing = set(SUMMARY_FIELD_MAP) - speculative_payload.keys()
        assert not missing, f"mapped fields absent from a speculative payload: {sorted(missing)}"

    def test_required_fields_are_a_subset_of_the_map(self) -> None:
        assert REQUIRED_FIELDS <= set(SUMMARY_FIELD_MAP)

    def test_column_names_are_unique(self) -> None:
        # Two sources mapping to one column would silently drop one of them.
        columns = list(SUMMARY_FIELD_MAP.values())
        assert len(columns) == len(set(columns))


class TestFlattening:
    def test_maps_metrics_onto_columns(self, payload: dict[str, Any]) -> None:
        flat = flatten_bench_result(payload)
        assert flat["successful_requests"] == 8
        assert flat["failed_requests"] == 0
        assert flat["total_generated_tokens"] == 128
        assert flat["ttft_ms_p99"] == pytest.approx(734.4049174101428)
        assert flat["itl_ms_median"] == pytest.approx(10.492878499917424)

    def test_unmapped_fields_survive_in_extra(self, payload: dict[str, Any]) -> None:
        # Nothing is silently dropped: an unrecognized field is a field we have not
        # promoted yet, not a field that does not matter.
        flat = flatten_bench_result(payload)
        assert "rtfx" in flat["extra"]
        assert "request_goodput" in flat["extra"]

    def test_context_fields_are_not_duplicated_into_summary(self, payload: dict[str, Any]) -> None:
        # The run row already records provenance. Two copies eventually disagree.
        flat = flatten_bench_result(payload)
        assert "model_id" not in flat
        assert "model_id" not in flat["extra"]

    def test_nulls_are_preserved_rather_than_defaulted(self) -> None:
        payload = {
            "completed": 1,
            "duration": 1.0,
            "total_input_tokens": 1,
            "total_output_tokens": 1,
            "mean_ttft_ms": None,
        }
        flat = flatten_bench_result(payload)
        # Null means "not computed". Coercing it to 0.0 would claim a measured zero, and
        # a zero TTFT would silently win every comparison it appears in.
        assert flat["ttft_ms_mean"] is None


class TestPerGpuNormalization:
    """Invariant 8, computed once at write time so consumers cannot disagree."""

    def test_single_gpu_normalization_is_identity(self, payload: dict[str, Any]) -> None:
        flat = flatten_bench_result(payload, gpu_count=1)
        assert flat["output_token_throughput_per_gpu"] == flat["output_token_throughput_tok_sec"]

    def test_multi_gpu_divides_by_device_count(self, payload: dict[str, Any]) -> None:
        flat = flatten_bench_result(payload, gpu_count=4)
        assert flat["output_token_throughput_per_gpu"] == pytest.approx(
            payload["output_throughput"] / 4
        )
        assert flat["total_token_throughput_per_gpu"] == pytest.approx(
            payload["total_token_throughput"] / 4
        )

    def test_a_tp4_run_can_lose_to_tp1_per_device(self) -> None:
        """The reason invariant 8 exists, as an executable statement.

        On aggregate throughput the four-GPU run wins by 3x and looks obviously better.
        Per device it is 25% worse. Charting the aggregate would recommend the wrong
        configuration with complete confidence.
        """
        base = {"completed": 1, "duration": 1.0, "total_input_tokens": 1, "total_output_tokens": 1}
        tp1 = flatten_bench_result({**base, "output_throughput": 100.0}, gpu_count=1)
        tp4 = flatten_bench_result({**base, "output_throughput": 300.0}, gpu_count=4)

        assert tp4["output_token_throughput_tok_sec"] > tp1["output_token_throughput_tok_sec"]
        assert tp4["output_token_throughput_per_gpu"] < tp1["output_token_throughput_per_gpu"]

    def test_null_throughput_normalizes_to_null(self) -> None:
        base = {"completed": 1, "duration": 1.0, "total_input_tokens": 1, "total_output_tokens": 1}
        flat = flatten_bench_result(base, gpu_count=2)
        assert flat["output_token_throughput_per_gpu"] is None

    def test_gpu_count_below_one_is_rejected(self, payload: dict[str, Any]) -> None:
        with pytest.raises(BenchResultError, match="at least 1"):
            flatten_bench_result(payload, gpu_count=0)


class TestFailsLoudly:
    """A truncated or reorganized payload must raise, never produce a row of NULLs."""

    def test_missing_required_fields_raise(self) -> None:
        with pytest.raises(BenchResultError) as exc:
            flatten_bench_result({"completed": 5})
        message = str(exc.value)
        assert "duration" in message
        # The message must point at the likely cause, not just state a fact.
        assert "--save-result schema" in message

    def test_empty_payload_raises(self) -> None:
        with pytest.raises(BenchResultError):
            flatten_bench_result({})

    def test_non_object_payload_raises(self) -> None:
        with pytest.raises(BenchResultError, match="expected a JSON object"):
            flatten_bench_result([])  # type: ignore[arg-type]

    def test_docs_style_field_names_are_rejected(self) -> None:
        """A payload using the *documented* names must fail, not silently produce NULLs.

        This is the regression guard for the original bug: if someone later "fixes" the
        map back to the documented names, this fails.
        """
        documented = {
            "successful_requests": 8,
            "benchmark_duration_sec": 1.5,
            "ttft_ms_p99": 700.0,
        }
        with pytest.raises(BenchResultError):
            flatten_bench_result(documented)


class TestBenchmarksThatMeasuredNothing:
    """The silent-corruption case, captured from a real vLLM rather than imagined.

    `bench_serve_all_requests_failed_v0.25.1.json` was produced by pointing
    `vllm bench serve` at a model name the server does not serve. Every request 404s —
    and the client **exits 0**, writes a result file, and reports zeros.

    Nothing upstream of the flattening layer notices: the process succeeded, the file
    exists, every required field is present. And the zeros are not neutral. On a latency
    axis 0 ms is the best value there is, so a run that measured nothing would render as
    the fastest configuration ever tested and sit on the Pareto frontier.
    """

    @pytest.fixture
    def all_failed(self) -> dict[str, Any]:
        return json.loads(
            (
                Path(__file__).parent / "fixtures" / "bench_serve_all_requests_failed_v0.25.1.json"
            ).read_text()
        )

    def test_the_captured_payload_really_does_look_successful(
        self, all_failed: dict[str, Any]
    ) -> None:
        """Pinning what makes this dangerous, so the reason the check exists stays visible.

        If a future vLLM starts emitting nulls or a non-zero exit here, this test fails
        and whoever sees it can decide the guard is no longer needed — rather than
        deleting it because it looks like paranoia.
        """
        assert all_failed["completed"] == 0
        assert all_failed["failed"] == 8
        assert all_failed["mean_ttft_ms"] == 0.0
        assert all_failed["output_throughput"] == 0.0
        # Every field the contract requires is present. That is the whole problem.
        assert not (REQUIRED_FIELDS - all_failed.keys())

    def test_it_is_refused_rather_than_flattened(self, all_failed: dict[str, Any]) -> None:
        with pytest.raises(EmptyBenchResult) as exc:
            flatten_bench_result(all_failed)

        # The message has to name the likely cause. "0 completed" alone sends the reader
        # looking at the engine, and the usual answer is a served-model-name mismatch.
        assert "0 requests" in str(exc.value)
        assert "8 request(s) failed" in str(exc.value)

    def test_it_is_a_kind_of_BenchResultError(self, all_failed: dict[str, Any]) -> None:
        """Subclassed so existing handlers still catch it, and new ones can tell it apart.

        The two failures need different responses: a schema mismatch is work in this
        repository, an empty benchmark is work on the host.
        """
        with pytest.raises(BenchResultError):
            flatten_bench_result(all_failed)

    def test_a_partially_failed_benchmark_is_kept(self, payload: dict[str, Any]) -> None:
        """Those requests really did complete, and the figures describe them.

        Throughput is divided by the whole duration, so a partial failure *understates* a
        run rather than inventing one. `failed_requests` is flattened alongside so the
        caveat travels with the numbers.
        """
        partial = {**payload, "completed": 6, "failed": 2}

        flat = flatten_bench_result(partial)

        assert flat["successful_requests"] == 6
        assert flat["failed_requests"] == 2

    def test_a_missing_completed_count_is_still_a_schema_error(self) -> None:
        """Absent and zero are different failures, and only one of them is this one."""
        with pytest.raises(BenchResultError, match="missing required fields"):
            flatten_bench_result(
                {"duration": 1.0, "total_input_tokens": 1, "total_output_tokens": 1}
            )


class TestSpeculativeDecoding:
    """The metric that explains a speculative result, rather than restating it.

    Throughput and TPOT say speculation helped or hurt. Acceptance rate says why, and it is
    the figure that transfers to a different workload — which is why it belongs in a column
    and not only in the log text it used to live in.
    """

    def test_the_scalars_land_on_columns(self, speculative_payload: dict[str, Any]) -> None:
        flat = flatten_bench_result(speculative_payload)
        # Verbatim from the captured run: 2723 of 3780 drafted tokens kept.
        assert flat["spec_acceptance_rate"] == pytest.approx(72.037, rel=1e-4)
        assert flat["spec_acceptance_length"] == pytest.approx(3.1611, rel=1e-4)
        assert flat["spec_num_drafts"] == 1260
        assert flat["spec_draft_tokens"] == 3780
        assert flat["spec_accepted_tokens"] == 2723

    def test_the_rate_agrees_with_the_counts(self, speculative_payload: dict[str, Any]) -> None:
        """Counters and the rate derived from them must tell the same story.

        Counters can be differenced across any window; a rate sampled at an instant cannot
        be recovered. If upstream ever changes what the rate is a percentage *of*, this
        fails rather than quietly redefining the axis of every chart.
        """
        flat = flatten_bench_result(speculative_payload)
        derived = 100.0 * flat["spec_accepted_tokens"] / flat["spec_draft_tokens"]
        assert derived == pytest.approx(flat["spec_acceptance_rate"], rel=1e-6)

    def test_per_position_rates_stay_whole(self, speculative_payload: dict[str, Any]) -> None:
        """An array as long as the speculation depth is not a scalar and not comparable.

        Spreading it across fixed columns would invent a width the data does not have —
        depth 1 and depth 3 would disagree about what column three means.
        """
        flat = flatten_bench_result(speculative_payload)
        assert flat["extra"]["spec_decode_per_position_acceptance_rates"] == [
            0.85,
            pytest.approx(0.7167, rel=1e-3),
            pytest.approx(0.5944, rel=1e-3),
        ]
        assert not any(key.startswith("spec_decode_per_position") for key in flat)

    def test_a_non_speculative_run_leaves_them_absent(self, payload: dict[str, Any]) -> None:
        """Absent, not zero.

        NULL says the engine was not speculating. Zero would say it drafted and had
        everything rejected. Those are opposite findings, and defaulting to zero here would
        put the second one in the database every time somebody benchmarked without MTP.
        """
        flat = flatten_bench_result(payload)
        for column in ("spec_acceptance_rate", "spec_acceptance_length", "spec_num_drafts"):
            assert column not in flat or flat[column] is None

    def test_a_non_speculative_payload_still_flattens(self, payload: dict[str, Any]) -> None:
        """The block is optional, so its absence must not look like a broken result."""
        flat = flatten_bench_result(payload)
        assert flat["successful_requests"] == 8
