"""Grouping, replicate aggregation, and the Pareto frontier.

These are the rules that decide what a chart is allowed to draw as one series and what
its error bars claim, so they are tested against the shapes that actually break them:
runs from two vLLM versions, replicates pulled from different sweeps, a point measured
once, and a frontier with ties on it.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest

from vllmbench_api.analysis import (
    METRICS_BY_KEY,
    PARETO_X,
    PARETO_Y,
    PER_USER_OUTPUT_TOK_S,
    PER_USER_OUTPUT_TOK_S_P99,
    Point,
    PointKey,
    RunRecord,
    RunSource,
    build_groups,
    build_point,
    comparability_key,
    derive_per_user_rates,
    group_warnings,
    pareto_frontier,
    spread_basis,
    summarize,
)
from vllmbench_db.enums import ReplicateOrder

HOST = uuid.uuid4()
SWEEP_A = uuid.uuid4()
SWEEP_B = uuid.uuid4()
EPOCH = dt.datetime(2026, 8, 17, 12, 0, tzinfo=dt.UTC)


def record(**overrides: object) -> RunRecord:
    """A real-shaped run; every test overrides only what it is about."""
    base: dict[str, object] = dict(
        run_id=uuid.uuid4(),
        finished_at=EPOCH,
        gpu_host_id=HOST,
        gpu_host_name="ubuntu-llm",
        gpu_model="NVIDIA GeForce RTX 3090",
        vllm_version="0.25.1",
        bench_client_location="loopback",
        driver_version="580.95.05",
        cuda_version="13.0",
        config_hash="cfg-a",
        config_name="tp2-fp8",
        workload_hash="wl-a",
        workload_name="sharegpt-64",
        gpu_count=2,
        tensor_parallel_size=2,
        max_concurrency=64,
        num_prompts=200,
        sweep_id=SWEEP_A,
        replicate_order=ReplicateOrder.GROUPED,
        metrics={},
    )
    base.update(overrides)
    return RunRecord(**base)  # type: ignore[arg-type]


class TestPerUserRates:
    def test_reciprocal_of_mean_tpot(self) -> None:
        rates = derive_per_user_rates(25.0, 50.0)
        assert rates[PER_USER_OUTPUT_TOK_S] == pytest.approx(40.0)
        assert rates[PER_USER_OUTPUT_TOK_S_P99] == pytest.approx(20.0)

    def test_zero_tpot_is_unmeasured_not_infinite(self) -> None:
        """A single-output-token benchmark has no inter-token interval to report.

        Treating it as infinite speed would put an unbounded value on a shared axis and
        flatten every other point on the chart to zero width.
        """
        assert derive_per_user_rates(0.0, 0.0) == {
            PER_USER_OUTPUT_TOK_S: None,
            PER_USER_OUTPUT_TOK_S_P99: None,
        }

    def test_missing_tpot_stays_missing(self) -> None:
        assert derive_per_user_rates(None, None)[PER_USER_OUTPUT_TOK_S] is None


class TestComparability:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("vllm_version", "0.26.0"),
            ("gpu_model", "NVIDIA L40S"),
            ("bench_client_location", "remote"),
            ("gpu_host_id", uuid.uuid4()),
        ],
    )
    def test_these_differences_split_the_chart(self, field: str, value: object) -> None:
        groups = build_groups([record(), record(**{field: value})])
        assert len(groups) == 2, f"{field} must not be overlaid silently"

    @pytest.mark.parametrize("field", ["driver_version", "cuda_version"])
    def test_these_differences_warn_instead_of_splitting(self, field: str) -> None:
        """Real enough to state, too frequent to fragment every chart over."""
        records = [record(), record(**{field: "999.0"})]
        groups = build_groups(records)
        assert len(groups) == 1
        assert groups[0].warnings and "mixed" in groups[0].warnings[0]

    def test_topology_is_not_a_comparability_key(self) -> None:
        """TP is the axis under study, so TP=1 and TP=2 belong on one chart.

        This is only safe because throughput is normalized per device (invariant 8);
        without that, putting them together would be the mistake.
        """
        groups = build_groups(
            [
                record(config_hash="tp1", tensor_parallel_size=1, gpu_count=1),
                record(config_hash="tp2", tensor_parallel_size=2, gpu_count=2),
            ]
        )
        assert len(groups) == 1
        assert len(groups[0].points) == 2

    def test_label_names_what_makes_the_group(self) -> None:
        key = comparability_key(record())
        assert "ubuntu-llm" in key.label
        assert "RTX 3090" in key.label
        assert "vLLM 0.25.1" in key.label

    def test_loopback_is_not_called_out_but_remote_is(self) -> None:
        # Loopback is the invariant-2 default; naming it on every chart is noise, while a
        # remote client is the thing a reader must not miss.
        assert "client" not in comparability_key(record()).label
        assert "remote client" in comparability_key(record(bench_client_location="remote")).label

    def test_groups_are_ordered_by_size(self) -> None:
        records = [record() for _ in range(3)] + [record(vllm_version="0.26.0")]
        groups = build_groups(records)
        assert groups[0].run_count == 3
        assert groups[1].run_count == 1

    def test_no_warning_when_a_field_is_merely_absent(self) -> None:
        # "unknown" and "different" are not the same claim; an unprobed host must not
        # manufacture a mixed-driver warning.
        assert group_warnings([record(), record(driver_version=None)]) == []


class TestSummarize:
    def test_median_min_max_and_raw_values(self) -> None:
        spread = summarize([3.0, 1.0, 2.0])
        assert spread is not None
        assert (spread.n, spread.median, spread.minimum, spread.maximum) == (3, 2.0, 1.0, 3.0)
        assert spread.values == (1.0, 2.0, 3.0), "raw replicates are kept, not summarized away"

    def test_median_resists_the_outlier_a_mean_does_not(self) -> None:
        """The reason charts plot the median: one throttled replicate in three."""
        spread = summarize([100.0, 101.0, 40.0])
        assert spread is not None
        assert spread.median == 100.0
        assert spread.mean == pytest.approx(80.333, abs=0.01)

    def test_empty_is_none_not_zero(self) -> None:
        assert summarize([]) is None

    def test_relative_range(self) -> None:
        spread = summarize([90.0, 100.0, 110.0])
        assert spread is not None
        assert spread.relative_range == pytest.approx(0.2)


class TestSpreadBasis:
    def test_single_run(self) -> None:
        assert spread_basis([record()]) == "single"

    def test_grouped_sweep(self) -> None:
        assert spread_basis([record(), record()]) == "grouped"

    def test_interleaved_sweep(self) -> None:
        rs = [record(replicate_order=ReplicateOrder.INTERLEAVED) for _ in range(2)]
        assert spread_basis(rs) == "interleaved"

    def test_two_sweeps_degrade_to_mixed(self) -> None:
        """A sweep's replicate_order only describes replicates inside that sweep.

        Two sittings differ by whatever changed between them; inheriting either sweep's
        claim would let a band labelled "repeatability" contain a week of drift.
        """
        assert spread_basis([record(sweep_id=SWEEP_A), record(sweep_id=SWEEP_B)]) == "mixed"

    def test_ad_hoc_runs_are_mixed(self) -> None:
        assert spread_basis([record(sweep_id=None), record(sweep_id=None)]) == "mixed"

    def test_sweep_run_plus_ad_hoc_run_is_mixed(self) -> None:
        assert spread_basis([record(), record(sweep_id=None)]) == "mixed"


class TestBuildPoint:
    def test_replicates_of_one_config_collapse_to_one_point(self) -> None:
        point = build_point(
            [
                record(metrics={"ttft_ms_p99": 100.0}, replicate_idx=0),
                record(metrics={"ttft_ms_p99": 120.0}, replicate_idx=1),
                record(metrics={"ttft_ms_p99": 110.0}, replicate_idx=2),
            ]
        )
        assert point.replicates == 3
        assert point.metrics["ttft_ms_p99"].median == 110.0
        assert point.key == PointKey("cfg-a", "wl-a")

    def test_a_metric_missing_from_one_replicate_still_aggregates(self) -> None:
        # Losing the whole point because one replicate lacked a field would discard two
        # good measurements; the n on the spread says how many contributed.
        point = build_point([record(metrics={"ttft_ms_p99": 100.0}), record(metrics={})])
        assert point.metrics["ttft_ms_p99"].n == 1

    def test_a_metric_missing_everywhere_is_absent_not_zero(self) -> None:
        point = build_point([record(metrics={})])
        assert "ttft_ms_p99" not in point.metrics
        assert point.value("ttft_ms_p99") is None

    def test_topology_disagreement_reports_the_largest(self) -> None:
        """A run that came up on fewer devices than asked for must not shrink the point.

        Reporting the smaller number would make the per-GPU normalization look generous
        for a config that in fact held every device.
        """
        point = build_point(
            [
                record(gpu_count=2, tensor_parallel_size=2),
                record(gpu_count=1, tensor_parallel_size=1),
            ]
        )
        assert (point.gpu_count, point.tensor_parallel_size) == (2, 2)

    def test_carries_the_runs_it_came_from(self) -> None:
        rs = [record(), record()]
        point = build_point(rs)
        assert set(point.run_ids) == {r.run_id for r in rs}
        assert point.sweep_ids == (SWEEP_A,)

    def test_latest_finish_is_the_newest_replicate(self) -> None:
        later = EPOCH + dt.timedelta(hours=1)
        point = build_point([record(finished_at=EPOCH), record(finished_at=later)])
        assert point.latest_finished_at == later


class TestPareto:
    @staticmethod
    def point_at(x: float, y: float, name: str = "c") -> Point:
        return build_point(
            [
                record(
                    config_hash=name,
                    config_name=name,
                    metrics={PARETO_X: x, PARETO_Y: y},
                )
            ]
        )

    @property
    def axes(self) -> tuple[object, object]:
        return METRICS_BY_KEY[PARETO_X], METRICS_BY_KEY[PARETO_Y]

    def test_dominated_points_are_dropped(self) -> None:
        good = self.point_at(100.0, 40.0, "good")
        worse = self.point_at(90.0, 30.0, "worse")
        frontier = pareto_frontier([good, worse], *self.axes)  # type: ignore[arg-type]
        assert [p.config_name for p in frontier] == ["good"]

    def test_a_genuine_trade_off_keeps_both(self) -> None:
        fast = self.point_at(50.0, 60.0, "fast-per-user")
        dense = self.point_at(120.0, 20.0, "high-throughput")
        frontier = pareto_frontier([fast, dense], *self.axes)  # type: ignore[arg-type]
        assert len(frontier) == 2
        # Sorted along x so a chart can draw the staircase without re-sorting.
        assert [p.config_name for p in frontier] == ["fast-per-user", "high-throughput"]

    def test_identical_points_both_survive(self) -> None:
        """Neither dominates the other, and they are different configs.

        Dropping one would hide that two configurations reach the same operating point —
        which is a finding, usually meaning a knob did nothing.
        """
        frontier = pareto_frontier(
            [self.point_at(100.0, 40.0, "a"), self.point_at(100.0, 40.0, "b")],
            *self.axes,  # type: ignore[arg-type]
        )
        assert {p.config_name for p in frontier} == {"a", "b"}

    def test_equal_on_one_axis_better_on_the_other_dominates(self) -> None:
        frontier = pareto_frontier(
            [self.point_at(100.0, 40.0, "same-x-better-y"), self.point_at(100.0, 30.0, "loser")],
            *self.axes,  # type: ignore[arg-type]
        )
        assert [p.config_name for p in frontier] == ["same-x-better-y"]

    def test_points_missing_an_axis_are_excluded_not_treated_as_zero(self) -> None:
        partial = build_point([record(config_hash="partial", metrics={PARETO_X: 500.0})])
        frontier = pareto_frontier(
            [self.point_at(100.0, 40.0, "real"), partial],
            *self.axes,  # type: ignore[arg-type]
        )
        assert [p.config_name for p in frontier] == ["real"]

    def test_lower_is_better_axis_is_oriented(self) -> None:
        """Latency axes must not be maximized.

        The orientation lives on the metric spec so a view can chart TTFT against
        throughput without every caller remembering to negate.
        """
        low = build_point(
            [
                record(
                    config_hash="low",
                    config_name="low-latency",
                    metrics={"ttft_ms_p99": 50.0, PARETO_X: 100.0},
                )
            ]
        )
        high = build_point(
            [
                record(
                    config_hash="high",
                    config_name="high-latency",
                    metrics={"ttft_ms_p99": 900.0, PARETO_X: 100.0},
                )
            ]
        )
        frontier = pareto_frontier(
            [low, high], METRICS_BY_KEY[PARETO_X], METRICS_BY_KEY["ttft_ms_p99"]
        )
        assert [p.config_name for p in frontier] == ["low-latency"]

    def test_empty_input(self) -> None:
        assert pareto_frontier([], *self.axes) == []  # type: ignore[arg-type]


class TestRunSource:
    def test_the_mixed_state_is_not_expressible(self) -> None:
        """Invariant 7 by construction rather than by filter.

        A caller picks one population. There is no argument that means "both", which is
        the only way a fabricated number reaches a chart beside a real one.
        """
        assert set(RunSource) == {RunSource.REAL, RunSource.SYNTHETIC}


class TestMetricRegistry:
    def test_per_gpu_metrics_are_marked(self) -> None:
        assert METRICS_BY_KEY[PARETO_X].per_gpu
        assert not METRICS_BY_KEY["total_token_throughput_tok_sec"].per_gpu

    def test_latency_metrics_prefer_lower(self) -> None:
        assert all(
            METRICS_BY_KEY[key].better == "lower"
            for key in METRICS_BY_KEY
            if key.endswith("_ms_p99") or key.endswith("_ms_median") or key.endswith("_ms_mean")
        )

    def test_keys_are_unique(self) -> None:
        from vllmbench_api.analysis import METRICS

        assert len({m.key for m in METRICS}) == len(METRICS)


class TestMetricKeysAreRealColumns:
    """Every non-derived metric key must name an actual ``RunSummary`` column.

    The router reads metrics with ``getattr(summary, key, None)``, so a key that does not
    match a column does not raise — it yields ``None`` for every run, and the metric
    simply never appears on a chart. That is the silent-NULL failure this repository
    exists to avoid, arriving through a typo rather than through upstream renaming a
    field.
    """

    def test_no_metric_key_is_a_typo(self) -> None:
        from vllmbench_api.analysis import METRICS
        from vllmbench_db.models import RunSummary

        derived = {PER_USER_OUTPUT_TOK_S, PER_USER_OUTPUT_TOK_S_P99}
        columns = set(RunSummary.__table__.columns.keys())
        missing = [m.key for m in METRICS if m.key not in derived and m.key not in columns]
        assert missing == [], f"not columns on run_summary: {missing}"

    def test_the_derived_keys_are_deliberately_not_columns(self) -> None:
        # Guards the inverse: if per-user rate ever becomes a stored column, this test
        # fails and the derivation should move to the flattening layer rather than being
        # computed twice from different inputs.
        from vllmbench_db.models import RunSummary

        columns = set(RunSummary.__table__.columns.keys())
        assert PER_USER_OUTPUT_TOK_S not in columns
        assert PER_USER_OUTPUT_TOK_S_P99 not in columns
