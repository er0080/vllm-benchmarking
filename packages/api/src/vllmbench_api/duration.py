"""How much longer a sweep has to run.

Pure functions over timings, with no SQL, for the same reason :mod:`analysis` is: every
judgement here is a guess, and a guess that cannot be exhaustively tested is one nobody
should act on.

**The estimate is a decomposition, not an average.** A sweep's wall clock is dominated by
model loading, not by benchmarking — on the first real sweep this repository measured, two
engine loads accounted for 358 of 1180 seconds while the ten remaining runs shared 655.
Averaging run durations would blend those two populations and produce a number that is
wrong in both directions at once: too high for a run inside a config block, too low for
one that starts a new one.

So runs are split by whether they paid for an engine load, matched to completed runs of the
*same workload* where possible, and the load overhead is recovered by subtraction. On that
sweep the subtraction returns 172 and 186 seconds against actual loads of 172 and 186.

Nothing here invents a number. With no observations to extrapolate from, the estimate is
``None`` and says why — a made-up remaining time is worse than an absent one, because a
progress display makes it look measured.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum


class EstimateBasis(StrEnum):
    """Where the numbers came from, stated rather than implied.

    A sweep extrapolating from its own runs is on much firmer ground than one borrowing
    from the host's history, and a reader deciding whether to trust a countdown needs to
    know which they have.
    """

    #: This sweep's own completed runs. The same model, the same host, minutes ago.
    THIS_SWEEP = "this_sweep"
    #: Nothing has completed yet, so there is nothing to extrapolate from.
    NONE = "none"


@dataclass(frozen=True, slots=True)
class RunTiming:
    """One completed run, as the estimator sees it."""

    workload_hash: str
    seconds: float
    #: Whether this run's wall clock includes starting an engine. True for the first run
    #: of each block of runs sharing a server config, because ``started_at`` is stamped
    #: when the orchestrator claims the run — before the server is launched.
    included_engine_load: bool


@dataclass(frozen=True, slots=True)
class PlannedRun:
    """One run still to happen."""

    workload_hash: str
    needs_engine_load: bool


@dataclass(frozen=True, slots=True)
class DurationEstimate:
    """What is left, and how confident the arithmetic behind it is."""

    runs_remaining: int = 0
    engine_loads_remaining: int = 0
    #: ``None`` when there is nothing to extrapolate from. Not zero — zero is a claim.
    seconds_remaining: float | None = None
    #: The benchmark cost of a run, excluding any engine load.
    median_run_seconds: float | None = None
    #: What an engine load adds, recovered by subtracting the workload's own benchmark
    #: time from a run that paid for one.
    median_engine_load_seconds: float | None = None
    basis: EstimateBasis = EstimateBasis.NONE
    #: Completed runs behind the estimate. Two is enough to be useful and not enough to
    #: be trusted to the second; this is here so a caller can decide that for itself.
    sample_size: int = 0
    #: Everything the number does not say. Empty when there is nothing to warn about.
    caveats: list[str] = field(default_factory=list)


def _median(values: Iterable[float]) -> float | None:
    collected = list(values)
    return statistics.median(collected) if collected else None


def estimate_remaining(
    observed: Sequence[RunTiming], remaining: Sequence[PlannedRun]
) -> DurationEstimate:
    """Extrapolate the time left from what this sweep has already done.

    Matching a remaining run to completed runs of the same workload is what makes the
    decomposition work: concurrency 4 and concurrency 64 against the same engine differ by
    three times in duration, so a single median across a mixed matrix would put the whole
    error into whichever workload is left.
    """
    loads_remaining = sum(1 for run in remaining if run.needs_engine_load)
    base = DurationEstimate(
        runs_remaining=len(remaining),
        engine_loads_remaining=loads_remaining,
        sample_size=len(observed),
    )
    if not remaining:
        # Nothing left is a fact, not an extrapolation, so it gets a number even with no
        # observations behind it.
        return DurationEstimate(sample_size=len(observed), seconds_remaining=0.0)
    if not observed:
        return DurationEstimate(
            runs_remaining=base.runs_remaining,
            engine_loads_remaining=loads_remaining,
            caveats=["No run has finished yet, so there is nothing to extrapolate from."],
        )

    benchmarks = [t for t in observed if not t.included_engine_load]
    with_loads = [t for t in observed if t.included_engine_load]

    if not benchmarks:
        # Every completed run started an engine — a matrix of one workload against many
        # configs does this. Their whole duration is the only unit available, and it only
        # applies to remaining runs that will also pay for a load.
        whole = _median(t.seconds for t in with_loads)
        if whole is None or loads_remaining != len(remaining):
            return DurationEstimate(
                runs_remaining=base.runs_remaining,
                engine_loads_remaining=loads_remaining,
                sample_size=len(observed),
                caveats=[
                    "Every completed run in this sweep started an engine, so the benchmark "
                    "time on its own has not been observed and the runs that will not "
                    "restart the engine cannot be estimated."
                ],
            )
        return DurationEstimate(
            runs_remaining=base.runs_remaining,
            engine_loads_remaining=loads_remaining,
            seconds_remaining=whole * len(remaining),
            basis=EstimateBasis.THIS_SWEEP,
            sample_size=len(observed),
            caveats=[
                "Every run measured so far included an engine load, so load and benchmark "
                "time could not be separated."
            ],
        )

    overall = _median(t.seconds for t in benchmarks)
    assert overall is not None  # benchmarks is non-empty
    by_workload: dict[str, float] = {}
    for workload in {t.workload_hash for t in benchmarks}:
        matched = _median(t.seconds for t in benchmarks if t.workload_hash == workload)
        if matched is not None:
            by_workload[workload] = matched

    def bench_cost(workload_hash: str) -> float:
        return by_workload.get(workload_hash, overall)

    # The overhead of an engine load, recovered by subtracting what the same workload
    # costs without one. Floored at zero: a negative would mean the load-inclusive run
    # was faster than its own workload's median, which is noise, not a refund.
    overheads = [max(0.0, t.seconds - bench_cost(t.workload_hash)) for t in with_loads]
    overhead = _median(overheads)

    caveats: list[str] = []
    unmatched = {r.workload_hash for r in remaining} - by_workload.keys()
    if unmatched:
        caveats.append(
            f"{len(unmatched)} of the workloads still to run have not been measured yet; "
            "they are estimated from the median of the others."
        )
    if loads_remaining and overhead is None:
        caveats.append(
            f"{loads_remaining} engine load(s) remain and none has been timed in this "
            "sweep, so their cost — usually the largest part of the total — is not "
            "included below."
        )

    total = sum(bench_cost(run.workload_hash) for run in remaining)
    total += loads_remaining * (overhead or 0.0)

    return DurationEstimate(
        runs_remaining=len(remaining),
        engine_loads_remaining=loads_remaining,
        seconds_remaining=total,
        median_run_seconds=overall,
        median_engine_load_seconds=overhead,
        basis=EstimateBasis.THIS_SWEEP,
        sample_size=len(observed),
        caveats=caveats,
    )


def plan_engine_loads(config_hashes: Sequence[str]) -> list[bool]:
    """Which runs in an ordered plan start a new engine.

    The same rule the sweep's ``engine_starts`` count uses: a run needs a load when its
    config differs from the one before it. Derived from the materialized order rather
    than from the authoring matrix, because the order is what will actually execute.
    """
    loads: list[bool] = []
    previous: str | None = None
    for digest in config_hashes:
        loads.append(digest != previous)
        previous = digest
    return loads
