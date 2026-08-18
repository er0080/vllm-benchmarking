"""The remaining-time estimate.

Unit tests, no database: every judgement here is a guess, and a guess is exactly the kind
of code that should be cheap to test exhaustively.

The case that matters most is the last one, which replays the real sweep this repository
measured on 2026-08-17 and checks that the decomposition recovers the two engine loads it
actually paid for.
"""

from __future__ import annotations

from vllmbench_api.duration import (
    EstimateBasis,
    PlannedRun,
    RunTiming,
    estimate_remaining,
    plan_engine_loads,
)


class TestEngineLoadPlanning:
    def test_the_first_run_always_loads(self) -> None:
        assert plan_engine_loads(["a"]) == [True]

    def test_runs_sharing_a_config_load_once(self) -> None:
        assert plan_engine_loads(["a", "a", "a"]) == [True, False, False]

    def test_each_change_costs_a_load(self) -> None:
        assert plan_engine_loads(["a", "a", "b", "b", "a"]) == [True, False, True, False, True]

    def test_an_empty_plan_needs_nothing(self) -> None:
        assert plan_engine_loads([]) == []


class TestRefusingToGuess:
    """A fabricated countdown looks measured. None is the honest answer."""

    def test_nothing_finished_yet(self) -> None:
        estimate = estimate_remaining([], [PlannedRun("w", True), PlannedRun("w", False)])
        assert estimate.seconds_remaining is None
        assert estimate.basis is EstimateBasis.NONE
        assert estimate.runs_remaining == 2
        # Still says what it *does* know: two runs and one engine load are left.
        assert estimate.engine_loads_remaining == 1
        assert estimate.caveats

    def test_nothing_left_is_zero_not_unknown(self) -> None:
        """An empty remainder is a fact about the plan, not an extrapolation."""
        estimate = estimate_remaining([], [])
        assert estimate.seconds_remaining == 0.0
        assert estimate.runs_remaining == 0


class TestDecomposition:
    def test_benchmark_time_and_load_time_are_separated(self) -> None:
        observed = [
            RunTiming("w", 300.0, included_engine_load=True),
            RunTiming("w", 60.0, included_engine_load=False),
        ]
        estimate = estimate_remaining(observed, [PlannedRun("w", False)])

        assert estimate.median_run_seconds == 60.0
        assert estimate.median_engine_load_seconds == 240.0
        assert estimate.seconds_remaining == 60.0

    def test_a_remaining_load_costs_its_own_overhead(self) -> None:
        """The whole reason the estimate is a decomposition.

        Two runs left is not one number: a run that restarts the engine costs four times
        what one inside the same config block does.
        """
        observed = [
            RunTiming("w", 300.0, included_engine_load=True),
            RunTiming("w", 60.0, included_engine_load=False),
        ]

        inside_block = estimate_remaining(observed, [PlannedRun("w", False)])
        starts_engine = estimate_remaining(observed, [PlannedRun("w", True)])

        assert inside_block.seconds_remaining == 60.0
        assert starts_engine.seconds_remaining == 300.0

    def test_workloads_are_matched_rather_than_pooled(self) -> None:
        """Concurrency 4 and concurrency 64 are not interchangeable.

        A single median across a mixed matrix puts the entire error into whichever
        workload happens to be left.
        """
        observed = [
            RunTiming("fast", 30.0, included_engine_load=False),
            RunTiming("slow", 150.0, included_engine_load=False),
        ]

        assert estimate_remaining(observed, [PlannedRun("slow", False)]).seconds_remaining == 150.0
        assert estimate_remaining(observed, [PlannedRun("fast", False)]).seconds_remaining == 30.0

    def test_an_unmeasured_workload_falls_back_and_says_so(self) -> None:
        observed = [
            RunTiming("fast", 30.0, included_engine_load=False),
            RunTiming("slow", 150.0, included_engine_load=False),
        ]
        estimate = estimate_remaining(observed, [PlannedRun("unknown", False)])

        assert estimate.seconds_remaining == 90.0  # the median of the two observed
        assert any("have not been measured" in c for c in estimate.caveats)

    def test_an_untimed_load_is_excluded_and_declared(self) -> None:
        """Silently omitting minutes of model loading is the failure mode here."""
        observed = [RunTiming("w", 60.0, included_engine_load=False)]
        estimate = estimate_remaining(observed, [PlannedRun("w", True)])

        assert estimate.median_engine_load_seconds is None
        assert estimate.seconds_remaining == 60.0
        assert any("not included" in c for c in estimate.caveats)

    def test_a_load_faster_than_its_own_workload_is_noise_not_a_refund(self) -> None:
        observed = [
            RunTiming("w", 55.0, included_engine_load=True),
            RunTiming("w", 60.0, included_engine_load=False),
        ]
        estimate = estimate_remaining(observed, [PlannedRun("w", True)])

        assert estimate.median_engine_load_seconds == 0.0
        assert estimate.seconds_remaining == 60.0

    def test_when_every_run_loaded_the_whole_duration_is_the_unit(self) -> None:
        """A matrix of many configs against one workload never observes a bare benchmark."""
        observed = [
            RunTiming("w", 300.0, included_engine_load=True),
            RunTiming("w", 320.0, included_engine_load=True),
        ]
        estimate = estimate_remaining(observed, [PlannedRun("w", True), PlannedRun("w", True)])

        assert estimate.seconds_remaining == 620.0
        assert any("could not be separated" in c for c in estimate.caveats)

    def test_when_every_run_loaded_but_the_remainder_will_not(self) -> None:
        """The one case where the unit does not apply, so nothing is claimed."""
        observed = [RunTiming("w", 300.0, included_engine_load=True)]
        estimate = estimate_remaining(observed, [PlannedRun("w", False)])

        assert estimate.seconds_remaining is None
        assert any("cannot be estimated" in c for c in estimate.caveats)


def test_it_recovers_the_real_sweeps_engine_loads() -> None:
    """Replay of the first real sweep, 2026-08-17 (docs/hardware-verification.md).

    Twelve runs, two config blocks, the durations exactly as recorded. The engine loads
    those two blocks actually paid for were 172 and 186 seconds, measured independently by
    the agent — so recovering ~179 by subtraction is the check that the decomposition
    describes the machine rather than merely fitting the arithmetic.

    Note what a naive average would have said: the twelve runs average 98 seconds, which
    is wrong for every single run in the sweep.
    """
    durations = [
        ("c4", 297.2, True),
        ("c4", 124.8, False),
        ("c16", 54.3, False),
        ("c16", 53.8, False),
        ("c64", 56.9, False),
        ("c64", 57.1, False),
        ("c4", 275.3, True),
        ("c4", 88.6, False),
        ("c16", 49.5, False),
        ("c16", 49.4, False),
        ("c64", 36.7, False),
        ("c64", 36.6, False),
    ]
    observed = [RunTiming(w, s, included_engine_load=load) for w, s, load in durations]

    estimate = estimate_remaining(observed, [PlannedRun("c4", True)])

    assert estimate.median_engine_load_seconds is not None
    assert 175.0 <= estimate.median_engine_load_seconds <= 185.0
    # And a remaining c4 run that restarts the engine is priced at load plus c4's own
    # benchmark time, not at the sweep's average run.
    assert estimate.seconds_remaining is not None
    assert 275.0 <= estimate.seconds_remaining <= 295.0
