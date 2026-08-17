"""Parsing vLLM's ``/metrics``.

Tested against payloads captured from a live vLLM 0.25.1 on a dual RTX 3090 host, not
against hand-written samples. A hand-written fixture encodes the author's belief about
the format, which is the belief under test — and every trap below was invisible until a
real payload was in hand.

The loaded fixture was taken with 128 concurrent requests in flight. That matters: in the
idle payload every value we care about is ``0.0``, so a parser that returned zeros for
everything would pass against it perfectly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vllmbench_protocol.metrics import (
    ALL_METRICS,
    COUNTER_METRICS,
    GAUGE_METRICS,
    parse_metrics,
    prefix_cache_hit_rate,
)

FIXTURES = Path(__file__).parent / "fixtures"
IDLE = (FIXTURES / "metrics_vllm_0_25_1_idle.txt").read_text()
LOADED = (FIXTURES / "metrics_vllm_0_25_1_loaded.txt").read_text()


class TestAgainstRealPayloads:
    def test_every_metric_we_claim_is_present(self) -> None:
        """If vLLM drops or renames one of these, that must fail here.

        A missing metric silently becomes a NULL column, and a telemetry series full of
        NULLs looks the same as an engine that was genuinely idle.
        """
        parsed = parse_metrics(LOADED)
        missing = set(ALL_METRICS.values()) - set(parsed)
        assert not missing, f"vLLM no longer emits: {missing}"

    def test_loaded_values_are_the_real_ones(self) -> None:
        parsed = parse_metrics(LOADED)
        assert parsed["num_requests_running"] == 64.0
        assert parsed["num_requests_waiting"] == 64.0
        assert parsed["prefix_cache_queries_total"] == 339968.0
        assert parsed["prefix_cache_hits_total"] == 2480.0
        assert parsed["num_preemptions_total"] == 0.0

    def test_kv_cache_usage_is_a_fraction_not_a_percent(self) -> None:
        """vLLM calls it ``_perc`` and emits 0.112 for 11.2%.

        Storing it under a ``_pct`` name is how a chart ends up rendering "0.1%" while
        the cache is an eighth full. It is kept exactly as emitted and converted only at
        display, so the name says fraction.
        """
        parsed = parse_metrics(LOADED)
        value = parsed["kv_cache_usage_fraction"]
        assert 0.0 < value < 1.0
        assert value == pytest.approx(0.11223878211531546)

    def test_idle_payload_parses_to_zeros_not_absence(self) -> None:
        # Zero and missing are different claims, and only one of them means the engine
        # was idle.
        parsed = parse_metrics(IDLE)
        assert parsed["num_requests_running"] == 0.0
        assert set(ALL_METRICS.values()) <= set(parsed)


class TestNameCollisions:
    """The trap that makes prefix matching wrong.

    The real payload contains ``num_requests_waiting`` *and*
    ``num_requests_waiting_by_reason``, with the latter appearing afterwards. Any
    ``startswith`` parser records the last one it sees — a queue depth of 0 while 64
    requests are queued, which is exactly the number someone would be reading to
    diagnose why TTFT is bad.
    """

    def test_by_reason_does_not_overwrite_waiting(self) -> None:
        assert "vllm:num_requests_waiting_by_reason" in LOADED
        assert parse_metrics(LOADED)["num_requests_waiting"] == 64.0

    def test_prefix_match_would_have_been_wrong(self) -> None:
        # Pins the hazard itself: the naive implementation returns a different answer,
        # so this test fails loudly if anyone "simplifies" the parser back to it.
        naive: dict[str, float] = {}
        for line in LOADED.splitlines():
            for metric, column in ALL_METRICS.items():
                if line.startswith(metric):
                    naive[column] = float(line.split()[-1])
        assert naive["num_requests_waiting"] == 0.0
        assert parse_metrics(LOADED)["num_requests_waiting"] == 64.0

    def test_counter_created_series_is_ignored(self) -> None:
        # Prometheus emits `X_created` alongside each counter, holding a unix timestamp.
        # Matched loosely, that timestamp becomes the counter value.
        assert "prefix_cache_queries_created" in LOADED
        assert parse_metrics(LOADED)["prefix_cache_queries_total"] == 339968.0


class TestAggregation:
    """vLLM labels every series ``engine="N"``, so data parallelism means duplicates."""

    def test_counts_are_summed_across_engines(self) -> None:
        text = (
            'vllm:num_requests_running{engine="0",model_name="m"} 4.0\n'
            'vllm:num_requests_running{engine="1",model_name="m"} 6.0\n'
        )
        assert parse_metrics(text)["num_requests_running"] == 10.0

    def test_fractions_are_averaged_not_summed(self) -> None:
        # Summing two 90%-full caches would report 180% utilization.
        text = (
            'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.9\n'
            'vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.5\n'
        )
        assert parse_metrics(text)["kv_cache_usage_fraction"] == pytest.approx(0.7)


class TestMalformedInput:
    def test_comments_and_blank_lines_are_skipped(self) -> None:
        assert parse_metrics("# HELP x y\n# TYPE x gauge\n\n") == {}

    def test_unparseable_value_is_omitted_not_zeroed(self) -> None:
        # Recording 0.0 would be indistinguishable from a real idle reading.
        assert "num_requests_running" not in parse_metrics("vllm:num_requests_running nonsense\n")

    def test_nan_is_omitted(self) -> None:
        assert "num_requests_running" not in parse_metrics("vllm:num_requests_running NaN\n")

    def test_unlabelled_series_still_parses(self) -> None:
        assert parse_metrics("vllm:num_requests_running 3.0\n")["num_requests_running"] == 3.0

    def test_unknown_metrics_are_ignored(self) -> None:
        # vLLM adding a metric is not an error.
        assert parse_metrics('vllm:something_brand_new{a="b"} 1.0\n') == {}

    def test_empty_payload(self) -> None:
        assert parse_metrics("") == {}


class TestHitRate:
    def test_derived_from_counters(self) -> None:
        parsed = parse_metrics(LOADED)
        rate = prefix_cache_hit_rate(
            parsed["prefix_cache_queries_total"], parsed["prefix_cache_hits_total"]
        )
        assert rate == pytest.approx(2480.0 / 339968.0)

    def test_no_queries_is_none_not_zero(self) -> None:
        # 0% measured and "nothing asked yet" are different claims, and averaging the
        # second into a sweep would drag the number down.
        assert prefix_cache_hit_rate(0, 0) is None


def test_gauges_and_counters_are_disjoint() -> None:
    assert not set(GAUGE_METRICS) & set(COUNTER_METRICS)
    assert len(ALL_METRICS) == len(GAUGE_METRICS) + len(COUNTER_METRICS)
