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
    DeviceSummary,
    Point,
    PointKey,
    RunBalance,
    RunRecord,
    RunSource,
    build_groups,
    build_point,
    comparability_key,
    config_diff,
    derive_per_user_rates,
    group_warnings,
    imbalance,
    metric_delta,
    pareto_frontier,
    scaling_curves,
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
        config_family="fam-a",
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


class TestScalingCurves:
    """Does TP=N earn its extra devices, and against what baseline.

    The rule that carries the weight is the grouping: a curve whose points are different
    configurations is not a scaling measurement. Everything here is about keeping the
    curve honest about what it compared.
    """

    @staticmethod
    def point_at(tp: int, per_gpu: float, *, family: str = "fam", workload: str = "wl-a") -> Point:
        return build_point(
            [
                record(
                    config_hash=f"{family}-tp{tp}",
                    config_name=f"cfg TP{tp}",
                    config_family=family,
                    workload_hash=workload,
                    workload_name=workload,
                    tensor_parallel_size=tp,
                    gpu_count=tp,
                    metrics={PARETO_X: per_gpu},
                )
            ]
        )

    def test_perfect_scaling_is_efficiency_one(self) -> None:
        curves = scaling_curves(
            [self.point_at(1, 1000.0), self.point_at(2, 1000.0)], metric_key=PARETO_X
        )
        (curve,) = curves
        assert curve.baseline_tp == 1
        assert [s.efficiency for s in curve.steps] == [1.0, 1.0]
        # Same per-GPU rate on twice the devices is twice the aggregate.
        assert [s.speedup for s in curve.steps] == [1.0, 2.0]

    def test_half_the_devices_wasted(self) -> None:
        curves = scaling_curves(
            [self.point_at(1, 1000.0), self.point_at(2, 500.0)], metric_key=PARETO_X
        )
        (curve,) = curves
        assert [s.efficiency for s in curve.steps] == [1.0, 0.5]
        # ...and the aggregate did not move at all, which is the finding.
        assert [s.speedup for s in curve.steps] == [1.0, 1.0]

    def test_superlinear_scaling_is_reported_not_clamped(self) -> None:
        # Real and explicable — a model that only fits in KV cache across two devices.
        # Clamping to 1.0 would hide the most interesting result a TP sweep can produce.
        curves = scaling_curves(
            [self.point_at(1, 1000.0), self.point_at(2, 1200.0)], metric_key=PARETO_X
        )
        assert curves[0].steps[1].efficiency == pytest.approx(1.2)

    def test_different_configs_do_not_share_a_curve(self) -> None:
        """The load-bearing rule.

        Two unrelated configs at TP=1 and TP=2 would otherwise look like one config
        scaling badly, when in fact nothing was scaled at all.
        """
        curves = scaling_curves(
            [self.point_at(1, 1000.0, family="a"), self.point_at(2, 400.0, family="b")],
            metric_key=PARETO_X,
        )
        assert curves == [], "each family had only one width; neither is a curve"

    def test_different_workloads_do_not_share_a_curve(self) -> None:
        # Scaling is only meaningful holding the traffic fixed; a curve mixing
        # concurrency 8 with concurrency 64 measures the workload, not the topology.
        curves = scaling_curves(
            [self.point_at(1, 1000.0, workload="c8"), self.point_at(2, 900.0, workload="c64")],
            metric_key=PARETO_X,
        )
        assert curves == []

    def test_one_curve_per_workload(self) -> None:
        points = [self.point_at(tp, 1000.0, workload=wl) for wl in ("c8", "c64") for tp in (1, 2)]
        curves = scaling_curves(points, metric_key=PARETO_X)
        assert {c.workload_name for c in curves} == {"c8", "c64"}
        assert all(len(c.steps) == 2 for c in curves)

    def test_baseline_is_the_narrowest_width_present(self) -> None:
        curves = scaling_curves(
            [self.point_at(2, 1000.0), self.point_at(4, 800.0)], metric_key=PARETO_X
        )
        (curve,) = curves
        assert curve.baseline_tp == 2
        assert curve.steps[0].is_baseline

    def test_a_baseline_that_is_not_one_gpu_is_flagged(self) -> None:
        """ "78% efficient" means something different measured against TP=2.

        The view has to say so; the alternative is a number that reads as parallel
        efficiency but was never measured against a single device.
        """
        relative = scaling_curves(
            [self.point_at(2, 1000.0), self.point_at(4, 800.0)], metric_key=PARETO_X
        )[0]
        absolute = scaling_curves(
            [self.point_at(1, 1000.0), self.point_at(2, 800.0)], metric_key=PARETO_X
        )[0]
        assert not relative.baseline_is_single_gpu
        assert absolute.baseline_is_single_gpu

    def test_a_single_width_is_not_a_curve(self) -> None:
        assert scaling_curves([self.point_at(2, 1000.0)], metric_key=PARETO_X) == []

    def test_points_missing_the_metric_are_skipped(self) -> None:
        blank = build_point(
            [record(config_hash="fam-tp4", config_family="fam", tensor_parallel_size=4, metrics={})]
        )
        curves = scaling_curves(
            [self.point_at(1, 1000.0), self.point_at(2, 900.0), blank], metric_key=PARETO_X
        )
        assert [s.tensor_parallel_size for s in curves[0].steps] == [1, 2]

    def test_a_zero_baseline_does_not_divide(self) -> None:
        curves = scaling_curves(
            [self.point_at(1, 0.0), self.point_at(2, 500.0)], metric_key=PARETO_X
        )
        (curve,) = curves
        assert all(s.efficiency is None for s in curve.steps)


class TestDeviceBalance:
    """Imbalance within a tensor-parallel group, the thing per-device keying exists for."""

    @staticmethod
    def devices(*utilizations: float | None) -> list[DeviceSummary]:
        return [
            DeviceSummary(gpu_index=i, samples=30, sm_utilization_pct=u)
            for i, u in enumerate(utilizations)
        ]

    def test_balanced_devices_report_zero(self) -> None:
        assert imbalance(self.devices(95.0, 95.0), "sm_utilization_pct") == 0.0

    def test_the_classic_finding(self) -> None:
        # One device at 60% while its peer sits at 95%: a third of one device idle.
        result = imbalance(self.devices(60.0, 95.0), "sm_utilization_pct")
        assert result == pytest.approx(0.368, abs=0.001)

    def test_measured_against_the_busiest_not_the_absolute_scale(self) -> None:
        """The same 20-point gap means different things at different loads.

        Twenty points apart at 90% is a mild split; twenty points apart at 25% means the
        quiet device did almost nothing. A fraction of the busiest device says so.
        """
        high = imbalance(self.devices(70.0, 90.0), "sm_utilization_pct")
        low = imbalance(self.devices(5.0, 25.0), "sm_utilization_pct")
        assert high is not None and low is not None and low > high

    def test_one_device_has_no_imbalance(self) -> None:
        # Not zero: zero would claim the devices were measured and agreed.
        assert imbalance(self.devices(80.0), "sm_utilization_pct") is None

    def test_an_idle_group_is_not_imbalanced(self) -> None:
        # Everything at zero would divide by zero and manufacture a finding out of
        # "nothing was running".
        assert imbalance(self.devices(0.0, 0.0), "sm_utilization_pct") is None

    def test_devices_missing_the_metric_are_ignored(self) -> None:
        assert imbalance(self.devices(90.0, None), "sm_utilization_pct") is None
        assert imbalance(self.devices(90.0, None, 45.0), "sm_utilization_pct") == 0.5

    def test_worst_imbalance_across_metrics(self) -> None:
        balance = RunBalance(
            run_id=uuid.uuid4(),
            config_name="c",
            workload_name="w",
            tensor_parallel_size=2,
            gpu_count=2,
            replicate_idx=0,
            finished_at=EPOCH,
            devices=(
                DeviceSummary(0, 30, sm_utilization_pct=95.0, memory_used_bytes=20e9),
                DeviceSummary(1, 30, sm_utilization_pct=60.0, memory_used_bytes=19.8e9),
            ),
        )
        assert balance.imbalances["sm_utilization_pct"] == pytest.approx(0.368, abs=0.001)
        # Memory being near-identical is the normal case — TP splits weights evenly — so
        # the headline has to be the metric that actually moved.
        assert balance.worst_imbalance == pytest.approx(0.368, abs=0.001)
        assert not balance.is_single_device

    def test_a_single_device_run_is_flagged_rather_than_scored(self) -> None:
        balance = RunBalance(
            run_id=uuid.uuid4(),
            config_name="c",
            workload_name="w",
            tensor_parallel_size=1,
            gpu_count=1,
            replicate_idx=0,
            finished_at=EPOCH,
            devices=(DeviceSummary(0, 30, sm_utilization_pct=95.0),),
        )
        assert balance.is_single_device
        assert balance.worst_imbalance is None


class TestConfigDiff:
    """A text diff of two configs, because the text *is* the config (invariant 5)."""

    def test_one_changed_value(self) -> None:
        left = "model: m\nmax-num-seqs: 8\n"
        right = "model: m\nmax-num-seqs: 64\n"
        lines = config_diff(left, right)
        assert [(line.kind, line.text) for line in lines] == [
            ("context", "model: m"),
            ("removed", "max-num-seqs: 8"),
            ("added", "max-num-seqs: 64"),
        ]

    def test_removals_precede_additions_within_a_change(self) -> None:
        # So a modified line reads as its old value then its new one, rather than the two
        # interleaved and the reader reconstructing which was which.
        lines = config_diff("a: 1\nb: 2\n", "a: 9\nb: 8\n")
        assert [line.kind for line in lines] == ["removed", "removed", "added", "added"]

    def test_line_numbers_track_both_sides(self) -> None:
        lines = config_diff("a: 1\nb: 2\n", "a: 1\nc: 3\nb: 2\n")
        added = next(line for line in lines if line.kind == "added")
        assert added.right_no == 2 and added.left_no is None

    def test_identical_configs_are_all_context(self) -> None:
        lines = config_diff("model: m\n", "model: m\n")
        assert all(line.kind == "context" for line in lines)

    def test_comments_are_compared_like_any_other_line(self) -> None:
        """A comment change is a real change to the file the engine is handed.

        Comparing parsed settings would call these identical, and the author's note about
        *why* a value is what it is would silently differ between two configs a reader
        was told were the same.
        """
        lines = config_diff("x: 1  # was 2\n", "x: 1  # tuned 2026-08\n")
        assert any(line.kind == "removed" for line in lines)

    def test_a_duplicated_key_shows_up(self) -> None:
        # Two declarations of one key change what the engine does and are invisible to
        # anything that compares parsed settings.
        lines = config_diff("tp: 1\n", "tp: 1\ntp: 2\n")
        assert [line.text for line in lines if line.kind == "added"] == ["tp: 2"]


class TestMetricDelta:
    def test_higher_is_better_improves_when_it_rises(self) -> None:
        change, better = metric_delta(100.0, 120.0, METRICS_BY_KEY[PARETO_X])
        assert change == pytest.approx(0.2)
        assert better is True

    def test_lower_is_better_improves_when_it_falls(self) -> None:
        change, better = metric_delta(200.0, 100.0, METRICS_BY_KEY["ttft_ms_p99"])
        assert change == pytest.approx(-0.5)
        assert better is True

    def test_no_change_is_neither_better_nor_worse(self) -> None:
        # Three states, so a view is not forced to render "unchanged" as a win.
        change, better = metric_delta(100.0, 100.0, METRICS_BY_KEY[PARETO_X])
        assert change == 0.0
        assert better is None

    def test_a_missing_side_is_unmeasurable(self) -> None:
        assert metric_delta(None, 100.0, METRICS_BY_KEY[PARETO_X]) == (None, None)

    def test_a_zero_baseline_does_not_divide(self) -> None:
        assert metric_delta(0.0, 100.0, METRICS_BY_KEY[PARETO_X]) == (None, None)


class TestPartialFailuresAreStated:
    """A run where some requests failed is a real measurement — of fewer requests.

    `vllm bench serve` divides throughput by the whole benchmark duration, so failures
    deflate the figure rather than invalidating it. The point on the chart cannot say
    that about itself, and a reader comparing it against a clean run would conclude the
    configuration is slower than it is.

    A run where *every* request failed never reaches analysis at all: it is refused at
    the flattening layer, because its zeros would read as the fastest result on the chart
    rather than as the absence of one.
    """

    def test_failed_requests_produce_a_warning(self) -> None:
        records = [
            record(failed_requests=0),
            record(failed_requests=3),
        ]

        warnings = group_warnings(records)

        assert any("failed requests" in w for w in warnings)
        assert any("up to 3" in w for w in warnings)

    def test_a_clean_group_says_nothing(self) -> None:
        """The warning is worthless if it appears on every chart."""
        assert not [
            w
            for w in group_warnings([record(failed_requests=0), record(failed_requests=None)])
            if "failed requests" in w
        ]
