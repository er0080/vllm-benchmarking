"""Turning runs into comparable points.

Everything the analysis views draw comes through here, and the reason it is a module of
pure functions rather than a pile of SQL is that its three jobs are all judgement calls
that have to be identical everywhere and testable without a database:

*What may be charted together.* CLAUDE.md's frontend rule — "never chart runs together
that differ in provenance in ways that invalidate the comparison" — is enforced by
partitioning into comparability groups here, upstream of any chart. A view cannot
accidentally overlay two vLLM versions because it is never handed them in one series.

*What the spread means.* A point's replicates are summarized as a median with a min/max
band, never as a bare mean, and the point states what produced the band: back-to-back
repeatability, run-to-run variance across a matrix, or drift between separate sweeps.
Error bars that do not say which of those they are showing are decoration.

*Which numbers are comparable at all.* Per-GPU normalization (invariant 8) is already
persisted; the derived per-user figure is computed in this one place so the UI, MCP tools
and any export agree on it rather than each dividing by its own idea of a device count.

Invariant 7 is enforced by construction rather than by a filter: the caller chooses one
``RunSource``, so a set holding both real and synthetic runs cannot be built.
"""

from __future__ import annotations

import datetime as dt
import difflib
import enum
import statistics
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from vllmbench_db.enums import ReplicateOrder


class RunSource(enum.StrEnum):
    """Which population of runs a query is about.

    Deliberately not a boolean ``include_synthetic``. Invariant 7 forbids synthetic runs
    from appearing "alongside real measurements", and a flag makes the forbidden state
    the easiest one to ask for. With a choice of one population, mixing is not something
    the caller can express — including by accident, which is how it would actually
    happen.
    """

    REAL = "real"
    SYNTHETIC = "synthetic"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricSpec:
    key: str
    label: str
    unit: str
    better: Literal["higher", "lower"]
    #: True when this figure is already normalized by device count, and so may be
    #: compared across tensor-parallel sizes without further work (invariant 8).
    per_gpu: bool = False
    description: str = ""

    def orient(self, value: float) -> float:
        """Flip so that larger is always better, for dominance tests."""
        return value if self.better == "higher" else -value


#: Per-user output tok/s, the y-axis of the Pareto view.
#:
#: This is the token rate implied by the *mean* time-per-output-token, which is the form
#: a serving SLO is written in ("at least 20 tok/s per user"). It is deliberately not the
#: mean of each request's own token rate: those are different numbers, because the mean of
#: a reciprocal is not the reciprocal of a mean, and the second one weights short requests
#: more heavily than any SLO does.
PER_USER_OUTPUT_TOK_S = "per_user_output_tok_s"
PER_USER_OUTPUT_TOK_S_P99 = "per_user_output_tok_s_p99"

METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        key="total_token_throughput_per_gpu",
        label="Total throughput per GPU",
        unit="tok/s/GPU",
        better="higher",
        per_gpu=True,
        description="Input plus output tokens per second, divided by the devices the run held.",
    ),
    MetricSpec(
        key="output_token_throughput_per_gpu",
        label="Output throughput per GPU",
        unit="tok/s/GPU",
        better="higher",
        per_gpu=True,
    ),
    MetricSpec(
        key=PER_USER_OUTPUT_TOK_S,
        label="Per-user output rate",
        unit="tok/s/user",
        better="higher",
        description="1000 / mean TPOT — the generation speed one request experiences.",
    ),
    MetricSpec(
        key=PER_USER_OUTPUT_TOK_S_P99,
        label="Per-user output rate (p99)",
        unit="tok/s/user",
        better="higher",
        description="1000 / p99 TPOT — the speed the slowest percentile of requests sees.",
    ),
    MetricSpec(
        key="total_token_throughput_tok_sec",
        label="Total throughput (aggregate)",
        unit="tok/s",
        better="higher",
        description="Not comparable across tensor-parallel sizes; use the per-GPU figure.",
    ),
    MetricSpec(
        key="output_token_throughput_tok_sec",
        label="Output throughput (aggregate)",
        unit="tok/s",
        better="higher",
    ),
    MetricSpec(
        key="request_throughput_req_sec",
        label="Request throughput",
        unit="req/s",
        better="higher",
    ),
    MetricSpec(key="ttft_ms_median", label="TTFT median", unit="ms", better="lower"),
    MetricSpec(key="ttft_ms_p99", label="TTFT p99", unit="ms", better="lower"),
    MetricSpec(key="tpot_ms_mean", label="TPOT mean", unit="ms", better="lower"),
    MetricSpec(key="tpot_ms_p99", label="TPOT p99", unit="ms", better="lower"),
    MetricSpec(key="itl_ms_median", label="ITL median", unit="ms", better="lower"),
    MetricSpec(key="itl_ms_p99", label="ITL p99", unit="ms", better="lower"),
    # Speculative decoding. NULL on a run that was not speculating, which is why these
    # are last: a comparison that includes a non-speculative arm will show them empty on
    # that side, and that is the correct reading rather than a gap in the data.
    MetricSpec(
        key="spec_acceptance_rate",
        label="Draft acceptance rate",
        unit="%",
        better="higher",
        description=(
            "Share of drafted tokens the target model kept. The number that explains a "
            "speculative result rather than restating it, and the one that transfers to a "
            "different workload."
        ),
    ),
    MetricSpec(
        key="spec_acceptance_length",
        label="Accepted tokens per step",
        unit="tokens",
        better="higher",
        description=(
            "Tokens gained per drafting step, including the one the target produces for "
            "free. This is the figure the speed-up is proportional to; a high acceptance "
            "rate at depth 1 is worth less than a lower one at depth 3."
        ),
    ),
)

METRICS_BY_KEY: Mapping[str, MetricSpec] = {m.key: m for m in METRICS}

#: The Pareto view's default axes, straight from CLAUDE.md: "per-user output tok/s
#: against per-GPU total tok/s".
PARETO_X = "total_token_throughput_per_gpu"
PARETO_Y = PER_USER_OUTPUT_TOK_S


def derive_per_user_rates(
    tpot_ms_mean: float | None, tpot_ms_p99: float | None
) -> dict[str, float | None]:
    """Per-user token rates from time-per-output-token.

    A TPOT of zero is treated as unmeasured rather than as infinite speed. It shows up
    when a benchmark produced a single output token per request — there is no
    inter-token interval to measure — and dividing by it would put an infinity on a
    chart axis and take every other point with it.
    """

    def rate(tpot: float | None) -> float | None:
        return 1000.0 / tpot if tpot is not None and tpot > 0 else None

    return {
        PER_USER_OUTPUT_TOK_S: rate(tpot_ms_mean),
        PER_USER_OUTPUT_TOK_S_P99: rate(tpot_ms_p99),
    }


# ---------------------------------------------------------------------------
# Input record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One succeeded run, flattened to what analysis needs and nothing else.

    Built from the ORM in the router. Keeping this a plain dataclass is what lets the
    grouping and aggregation rules — the parts with actual judgement in them — be tested
    exhaustively without a Postgres round trip per case.
    """

    run_id: uuid.UUID
    finished_at: dt.datetime | None

    # Comparability (see comparability_key)
    gpu_host_id: uuid.UUID
    gpu_host_name: str
    gpu_model: str | None
    vllm_version: str | None
    bench_client_location: str
    driver_version: str | None = None
    cuda_version: str | None = None
    #: Set when this run was imported rather than measured here, so its hardware and
    #: vLLM version were *declared by a person* rather than observed by an agent.
    imported_from: str | None = None
    #: Requests that failed during the benchmark. A run with some completions is a real
    #: measurement of those completions, but throughput is divided by the whole duration
    #: — so failures deflate it, and the chart cannot show that on its own.
    failed_requests: int | None = None

    # Point identity
    config_hash: str = ""
    config_name: str = ""
    workload_hash: str = ""
    workload_name: str = ""
    #: Hash of the config text with its tensor-parallel line normalized out. Points
    #: sharing it are the same engine configuration measured at different widths, which
    #: is the only grouping under which a scaling curve means anything.
    config_family: str = ""

    # Topology (invariant 8 provenance, reported by the agent from what actually ran)
    gpu_count: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    device_indices: tuple[int, ...] = ()

    # Workload axes worth plotting against
    max_concurrency: int | None = None
    request_rate: float | None = None
    num_prompts: int = 0

    # Provenance of the replicate itself
    sweep_id: uuid.UUID | None = None
    sweep_name: str | None = None
    replicate_idx: int = 0
    replicate_order: ReplicateOrder | None = None

    metrics: Mapping[str, float | None] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Comparability
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComparabilityKey:
    """What must match before two runs may share a chart series.

    Each field is here because a difference in it makes the numbers mean different
    things, not merely because it differs:

    ``gpu_host`` — two hosts with identical cards still differ in CPU, memory bandwidth
    and PCIe topology, all of which move TTFT.

    ``gpu_model`` — the obvious one, and kept separate from the host so a re-carded host
    does not silently pool two generations of device.

    ``vllm_version`` — CLAUDE.md's version policy is explicit that charts group by it.
    Comparing versions is a headline use of this tool, which is exactly why the
    comparison must be deliberate rather than accidental.

    ``bench_client_location`` — invariant 2. Latency measured across a network hop and
    latency measured over loopback are not the same quantity, and nothing downstream can
    separate them once they are averaged together.

    Driver and CUDA version are *not* keys. They move numbers less than any of the above
    and change often enough that partitioning on them would fragment every chart into
    single-point series; they surface as a warning on the group instead.
    """

    gpu_host_id: uuid.UUID
    gpu_host_name: str
    gpu_model: str | None
    vllm_version: str | None
    bench_client_location: str

    @property
    def label(self) -> str:
        parts = [self.gpu_host_name]
        if self.gpu_model:
            parts.append(self.gpu_model)
        parts.append(f"vLLM {self.vllm_version}" if self.vllm_version else "vLLM unknown")
        if self.bench_client_location != "loopback":
            parts.append(f"{self.bench_client_location} client")
        return " · ".join(parts)


def comparability_key(record: RunRecord) -> ComparabilityKey:
    return ComparabilityKey(
        gpu_host_id=record.gpu_host_id,
        gpu_host_name=record.gpu_host_name,
        gpu_model=record.gpu_model,
        vllm_version=record.vllm_version,
        bench_client_location=record.bench_client_location,
    )


def group_warnings(records: Sequence[RunRecord]) -> list[str]:
    """Differences that are worth stating but not worth splitting a chart over."""
    warnings: list[str] = []
    for label, values in (
        ("driver version", {r.driver_version for r in records if r.driver_version}),
        ("CUDA version", {r.cuda_version for r in records if r.cuda_version}),
    ):
        if len(values) > 1:
            warnings.append(f"mixed {label}: {', '.join(sorted(values))}")

    # Imported runs sit in a group because their *declared* provenance matched, not
    # because anything observed it. That is a weaker claim than the rest of the group
    # rests on, and ADR 0003 says it is stated rather than assumed — this is the "group
    # or warn, never silently overlay" rule applied to a difference we cannot see.
    imported = {r.imported_from for r in records if r.imported_from}
    if imported:
        measured = sum(1 for r in records if not r.imported_from)
        sources = ", ".join(sorted(imported))
        if measured:
            warnings.append(
                f"mixes {measured} measured run(s) with imported ones ({sources}); the "
                "imported hardware and vLLM version were declared by a person, not observed"
            )
        else:
            warnings.append(
                f"imported from {sources}; hardware and vLLM version were declared by a "
                "person, not observed"
            )

    # Runs where some requests failed. `vllm bench serve` divides throughput by the whole
    # benchmark duration, so a partially failed run is understated rather than invalid —
    # a real measurement of fewer requests than were asked for. Nothing about the point
    # on the chart says so, which is why it is said here.
    #
    # A run where *every* request failed never reaches this: it is refused at the
    # flattening layer, because its zeros would read as the fastest result on the chart
    # rather than as the absence of one.
    partial = [r for r in records if r.failed_requests]
    if partial:
        worst = max(r.failed_requests or 0 for r in partial)
        warnings.append(
            f"{len(partial)} run(s) had failed requests (up to {worst} in one run); "
            "throughput is divided by the whole benchmark duration, so those points "
            "understate the configuration rather than describe it"
        )

    return warnings


# ---------------------------------------------------------------------------
# Replicate aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Spread:
    """A measurement and how much it moved across replicates.

    ``median`` is what charts plot. With the default of three replicates a mean is
    dragged by a single thermally-throttled outlier in a way the median is not, and the
    outlier is still visible because ``min``/``max`` and the raw values are carried
    alongside rather than summarized away.
    """

    n: int
    median: float
    mean: float
    minimum: float
    maximum: float
    values: tuple[float, ...]

    @property
    def range(self) -> float:
        return self.maximum - self.minimum

    @property
    def relative_range(self) -> float | None:
        """Spread as a fraction of the median — the form worth flagging on."""
        return self.range / abs(self.median) if self.median else None


def summarize(values: Iterable[float]) -> Spread | None:
    ordered = sorted(values)
    if not ordered:
        return None
    return Spread(
        n=len(ordered),
        median=statistics.median(ordered),
        mean=statistics.fmean(ordered),
        minimum=ordered[0],
        maximum=ordered[-1],
        values=tuple(ordered),
    )


SpreadBasis = Literal["single", "grouped", "interleaved", "mixed"]

_SPREAD_NOTES: Mapping[SpreadBasis, str] = {
    "single": "One run — no spread was measured.",
    "grouped": (
        "Replicates ran back-to-back, so the band shows repeatability under near-identical "
        "conditions. It understates the variance between separate sittings."
    ),
    "interleaved": (
        "Replicates were spread across the sweep, so the band shows run-to-run variance a "
        "reader would actually meet."
    ),
    "mixed": (
        "Replicates come from more than one sweep or from ad-hoc runs, so the band includes "
        "drift between sittings as well as measurement noise."
    ),
}


def spread_basis(records: Sequence[RunRecord]) -> SpreadBasis:
    """What a point's error band is actually measuring.

    ``replicate_order`` only describes replicates within one sweep. Two runs of the same
    config and workload from different sweeps are still legitimately the same measurement
    point, but their difference includes whatever changed between the two sittings — so
    the basis degrades to ``mixed`` rather than inheriting either sweep's claim.
    """
    if len(records) < 2:
        return "single"
    sweeps = {r.sweep_id for r in records}
    if len(sweeps) > 1 or None in sweeps:
        return "mixed"
    orders = {r.replicate_order for r in records}
    if len(orders) != 1:
        return "mixed"
    order = next(iter(orders))
    return "interleaved" if order == ReplicateOrder.INTERLEAVED else "grouped"


def spread_note(basis: SpreadBasis) -> str:
    return _SPREAD_NOTES[basis]


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PointKey:
    config_hash: str
    workload_hash: str


@dataclass(frozen=True, slots=True)
class Point:
    """One measurement point: a config running a workload, over its replicates."""

    key: PointKey
    config_name: str
    workload_name: str
    #: See RunRecord.config_family.
    family: str

    tensor_parallel_size: int
    pipeline_parallel_size: int
    gpu_count: int
    max_concurrency: int | None
    request_rate: float | None
    num_prompts: int

    run_ids: tuple[uuid.UUID, ...]
    sweep_ids: tuple[uuid.UUID, ...]
    basis: SpreadBasis
    latest_finished_at: dt.datetime | None

    metrics: Mapping[str, Spread]

    @property
    def replicates(self) -> int:
        return len(self.run_ids)

    def value(self, key: str) -> float | None:
        spread = self.metrics.get(key)
        return spread.median if spread else None


def _topology_of(records: Sequence[RunRecord]) -> tuple[int, int, int]:
    """TP, PP and device count for a point, taken from what actually ran.

    Replicates of one config should agree. If they do not — a run that came up on fewer
    devices than asked for, say — the largest is reported, because the alternative is a
    point that silently claims a topology none of its runs had.
    """
    return (
        max(r.tensor_parallel_size for r in records),
        max(r.pipeline_parallel_size for r in records),
        max(r.gpu_count for r in records),
    )


def build_point(records: Sequence[RunRecord]) -> Point:
    if not records:  # pragma: no cover - guarded by the caller's grouping
        raise ValueError("a point needs at least one run")
    unfinished = dt.datetime.min.replace(tzinfo=dt.UTC)
    ordered = sorted(records, key=lambda r: (r.finished_at or unfinished, r.replicate_idx))
    head = ordered[0]
    tp, pp, gpus = _topology_of(ordered)

    metrics: dict[str, Spread] = {}
    for spec in METRICS:
        present = [
            value for value in (r.metrics.get(spec.key) for r in ordered) if value is not None
        ]
        summary = summarize(present)
        if summary is not None:
            metrics[spec.key] = summary

    finished = [r.finished_at for r in ordered if r.finished_at is not None]
    return Point(
        key=PointKey(config_hash=head.config_hash, workload_hash=head.workload_hash),
        config_name=head.config_name,
        workload_name=head.workload_name,
        family=head.config_family or head.config_hash,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        gpu_count=gpus,
        max_concurrency=head.max_concurrency,
        request_rate=head.request_rate,
        num_prompts=head.num_prompts,
        run_ids=tuple(r.run_id for r in ordered),
        sweep_ids=tuple(sorted({r.sweep_id for r in ordered if r.sweep_id}, key=str)),
        basis=spread_basis(ordered),
        latest_finished_at=max(finished) if finished else None,
        metrics=metrics,
    )


@dataclass(frozen=True, slots=True)
class Group:
    """Points that may legitimately share a chart."""

    key: ComparabilityKey
    points: tuple[Point, ...]
    warnings: tuple[str, ...]

    @property
    def run_count(self) -> int:
        return sum(p.replicates for p in self.points)


def build_groups(records: Iterable[RunRecord]) -> list[Group]:
    """Partition runs into comparable sets, then into points within each.

    Groups are ordered by run count so the fullest comparison leads; a stray run on a
    different vLLM version becomes a small trailing group rather than silently joining
    the main series.
    """
    by_group: dict[ComparabilityKey, list[RunRecord]] = {}
    for record in records:
        by_group.setdefault(comparability_key(record), []).append(record)

    groups: list[Group] = []
    for key, group_records in by_group.items():
        by_point: dict[PointKey, list[RunRecord]] = {}
        for record in group_records:
            by_point.setdefault(PointKey(record.config_hash, record.workload_hash), []).append(
                record
            )
        points = tuple(
            sorted(
                (build_point(rs) for rs in by_point.values()),
                key=lambda p: (p.config_name, p.max_concurrency or 0, p.workload_name),
            )
        )
        groups.append(Group(key=key, points=points, warnings=tuple(group_warnings(group_records))))

    groups.sort(key=lambda g: (-g.run_count, g.key.label))
    return groups


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------


def pareto_frontier(points: Sequence[Point], x: MetricSpec, y: MetricSpec) -> list[Point]:
    """The points no other point beats on both axes at once.

    The tuning question this answers is "which configurations are worth considering at
    all": a config that is slower per user *and* lower throughput per GPU than another is
    never the right answer, whatever the operator's preference between the two. What
    survives is the set where the choice is a genuine trade-off.

    Computed on the median of each point's replicates, which is what the chart plots.
    Doing it on a mean would let one throttled replicate push a config off the frontier
    it actually holds.
    """
    usable = [
        (p, x.orient(px), y.orient(py))
        for p in points
        if (px := p.value(x.key)) is not None and (py := p.value(y.key)) is not None
    ]

    frontier = [
        point
        for point, px, py in usable
        # Ties are kept: two points with identical coordinates neither dominate each
        # other nor should one be silently dropped, since they are different configs.
        if not any((qx >= px and qy >= py) and (qx > px or qy > py) for _, qx, qy in usable)
    ]
    frontier.sort(key=lambda p: (p.value(x.key) or 0.0, p.value(y.key) or 0.0))
    return frontier


# ---------------------------------------------------------------------------
# Tensor-parallel scaling
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScalingStep:
    """One tensor-parallel width on a scaling curve."""

    tensor_parallel_size: int
    point: Point

    #: Aggregate throughput relative to the baseline width. What "twice the GPUs went
    #: twice as fast" means, and the number an operator feels.
    speedup: float | None
    #: Per-GPU throughput relative to the baseline width — parallel efficiency. 1.0 means
    #: each added device pulled its weight; 0.5 means half of them were wasted.
    efficiency: float | None

    #: Stated rather than inferred from ``speedup == 1.0``. The ratios are floats and
    #: which step defines them is a fact about the curve, not something to recover by
    #: comparing one to a literal.
    is_baseline: bool = False


@dataclass(frozen=True, slots=True)
class ScalingCurve:
    """How one configuration, running one workload, responds to more devices.

    Keyed by *config family* rather than by config: a curve whose points are different
    configurations is not a scaling measurement, it is a comparison of configurations
    that happen to have different widths. The family is the config text with its
    tensor-parallel line normalized out, so members are the same engine setup at
    different widths and nothing else.
    """

    family: str
    config_name: str
    workload_hash: str
    workload_name: str
    max_concurrency: int | None
    baseline_tp: int
    steps: tuple[ScalingStep, ...]

    @property
    def baseline_is_single_gpu(self) -> bool:
        """Whether efficiency here means what "parallel efficiency" usually means.

        Relative to TP=1 it is the standard figure. Relative to anything else it is a
        ratio against a baseline that was itself already parallel, and a view reporting
        "78% efficient" without saying so is making a claim it did not measure.
        """
        return self.baseline_tp == 1


def scaling_curves(points: Sequence[Point], *, metric_key: str) -> list[ScalingCurve]:
    """Group points into scaling curves, one per config family and workload.

    A curve needs at least two widths to say anything, so single-width families are
    dropped — with one point there is no scaling to report, and drawing it would invite
    reading a single measurement as a trend.

    Efficiency is computed from each width's median, which is what the throughput chart
    plots. Propagating the replicate spread through a ratio would need the joint
    distribution of two independently-measured widths; the spread stays visible on the
    throughput axis instead of being folded into a derived number that hides it.
    """
    by_curve: dict[tuple[str, str], list[Point]] = {}
    for point in points:
        by_curve.setdefault((point.family, point.key.workload_hash), []).append(point)

    curves: list[ScalingCurve] = []
    for (family, workload_hash), members in by_curve.items():
        widths = sorted(members, key=lambda p: p.tensor_parallel_size)
        usable = [p for p in widths if p.value(metric_key) is not None]
        if len({p.tensor_parallel_size for p in usable}) < 2:
            continue

        baseline = usable[0]
        base_per_gpu = baseline.value(metric_key)
        base_tp = baseline.tensor_parallel_size
        # Aggregate is reconstructed from the per-GPU figure rather than read from a
        # separate column, so speedup and efficiency are guaranteed to be the same
        # measurement seen two ways rather than two numbers that can disagree.
        base_aggregate = (base_per_gpu or 0.0) * baseline.gpu_count

        steps: list[ScalingStep] = []
        for point in usable:
            per_gpu = point.value(metric_key)
            if per_gpu is None or not base_per_gpu:
                steps.append(
                    ScalingStep(
                        tensor_parallel_size=point.tensor_parallel_size,
                        point=point,
                        speedup=None,
                        efficiency=None,
                        is_baseline=point is baseline,
                    )
                )
                continue
            aggregate = per_gpu * point.gpu_count
            steps.append(
                ScalingStep(
                    tensor_parallel_size=point.tensor_parallel_size,
                    point=point,
                    speedup=aggregate / base_aggregate if base_aggregate else None,
                    efficiency=per_gpu / base_per_gpu,
                    is_baseline=point is baseline,
                )
            )

        head = baseline
        curves.append(
            ScalingCurve(
                family=family,
                config_name=head.config_name,
                workload_hash=workload_hash,
                workload_name=head.workload_name,
                max_concurrency=head.max_concurrency,
                baseline_tp=base_tp,
                steps=tuple(steps),
            )
        )

    curves.sort(key=lambda c: (c.workload_name, c.max_concurrency or 0, c.config_name))
    return curves


# ---------------------------------------------------------------------------
# Per-device balance
# ---------------------------------------------------------------------------

#: Metrics whose imbalance across a tensor-parallel group is diagnostic, and how to read
#: a gap in each. Kept small deliberately: temperature and clocks differ between devices
#: for reasons that have nothing to do with the work split, so reporting their spread
#: would train the reader to ignore the number.
BALANCE_METRICS: tuple[tuple[str, str], ...] = (
    ("sm_utilization_pct", "SM utilization"),
    ("memory_used_bytes", "Memory used"),
    ("power_watts", "Power draw"),
)


@dataclass(frozen=True, slots=True)
class DeviceSummary:
    """One GPU's behaviour over one run."""

    gpu_index: int
    samples: int
    sm_utilization_pct: float | None = None
    sm_utilization_max: float | None = None
    memory_used_bytes: float | None = None
    power_watts: float | None = None

    def value(self, key: str) -> float | None:
        return getattr(self, key, None)


def imbalance(devices: Sequence[DeviceSummary], key: str) -> float | None:
    """How far apart the devices sat, as a fraction of the busiest one.

    ``(max - min) / max``: 0.0 is perfectly balanced, 0.35 means the quietest device did
    a third less than the busiest. A fraction rather than an absolute gap because the
    same 20-point spread means something different at 90% utilization than at 25%.

    ``None`` when fewer than two devices reported, or when the busiest is at zero —
    "nothing was running" is not an imbalance, and dividing by it would manufacture one.
    """
    values = [value for device in devices if (value := device.value(key)) is not None]
    if len(values) < 2:
        return None
    top = max(values)
    return (top - min(values)) / top if top > 0 else None


@dataclass(frozen=True, slots=True)
class RunBalance:
    """One run's devices, and how evenly they shared the work."""

    run_id: uuid.UUID
    config_name: str
    workload_name: str
    tensor_parallel_size: int
    gpu_count: int
    #: Which replicate of its point this was. Needed because this view lists executions
    #: rather than points, so three replicates otherwise appear as three identically
    #: labelled bars with no way to tell which is which.
    replicate_idx: int
    finished_at: dt.datetime | None
    devices: tuple[DeviceSummary, ...]

    @property
    def imbalances(self) -> dict[str, float | None]:
        return {key: imbalance(self.devices, key) for key, _ in BALANCE_METRICS}

    @property
    def worst_imbalance(self) -> float | None:
        found = [value for value in self.imbalances.values() if value is not None]
        return max(found) if found else None

    @property
    def is_single_device(self) -> bool:
        """A one-device run has no balance to report, and that is not a warning."""
        return len(self.devices) < 2


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DiffLine:
    """One line of a config diff, tagged by what happened to it."""

    kind: Literal["context", "added", "removed", "header"]
    text: str
    #: Line number in the left and right configs, where the line exists in each.
    left_no: int | None = None
    right_no: int | None = None


def config_diff(left: str, right: str) -> list[DiffLine]:
    """A line diff of two configs' exact text.

    A *text* diff, not a comparison of parsed settings. Invariant 5 makes the YAML the
    config — what is stored is what runs — so the honest answer to "what is different
    about these two" is which bytes differ. Parsing to compare key by key would need
    opinions about vLLM's option set that rot with every release, would silently normalize
    away the comments an author wrote to explain a value, and would report two configs as
    identical when one of them has a duplicated key that changes what the engine does.
    """
    left_lines = left.splitlines()
    right_lines = right.splitlines()
    out: list[DiffLine] = []

    matcher = difflib.SequenceMatcher(a=left_lines, b=right_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                out.append(
                    DiffLine(
                        kind="context",
                        text=left_lines[i1 + offset],
                        left_no=i1 + offset + 1,
                        right_no=j1 + offset + 1,
                    )
                )
            continue
        # Removals before additions within a change, so a modified line reads as its old
        # value followed by its new one rather than the two interleaved.
        for index in range(i1, i2):
            out.append(DiffLine(kind="removed", text=left_lines[index], left_no=index + 1))
        for index in range(j1, j2):
            out.append(DiffLine(kind="added", text=right_lines[index], right_no=index + 1))
    return out


@dataclass(frozen=True, slots=True)
class ProvenanceDifference:
    """One fact that differs between two runs being compared."""

    field: str
    label: str
    left: str | None
    right: str | None
    #: ``invalidating`` differences are the ones that stop the two being one series on a
    #: chart. They do *not* stop a deliberate side-by-side: comparing two vLLM versions is
    #: a headline use of this tool. The difference between the two situations is that here
    #: the reader named both sides and is being told exactly what changed.
    invalidating: bool


_COMPARED_FIELDS: tuple[tuple[str, str, bool], ...] = (
    ("gpu_host_name", "GPU host", True),
    ("gpu_model", "GPU model", True),
    ("vllm_version", "vLLM version", True),
    ("bench_client_location", "Benchmark client", True),
    ("driver_version", "Driver version", False),
    ("cuda_version", "CUDA version", False),
    ("workload_name", "Workload", False),
    ("tensor_parallel_size", "Tensor parallel size", False),
    ("gpu_count", "GPU count", False),
)


def provenance_differences(
    left: Point, right: Point, sides: Mapping[str, RunRecord]
) -> list[ProvenanceDifference]:
    """Every recorded fact that is not the same on both sides.

    Listed rather than used to refuse. A chart must not silently overlay two vLLM
    versions; a comparison the reader explicitly asked for is where that difference is
    the subject, and hiding it behind a refusal would block the feature the version
    policy exists to enable.
    """
    out: list[ProvenanceDifference] = []
    for field_name, label, invalidating in _COMPARED_FIELDS:
        a = getattr(sides["left"], field_name, None)
        b = getattr(sides["right"], field_name, None)
        if field_name in {"tensor_parallel_size", "gpu_count"}:
            a = getattr(left, field_name, a)
            b = getattr(right, field_name, b)
        if a != b:
            out.append(
                ProvenanceDifference(
                    field=field_name,
                    label=label,
                    left=None if a is None else str(a),
                    right=None if b is None else str(b),
                    invalidating=invalidating,
                )
            )
    return out


def metric_delta(
    left: float | None, right: float | None, spec: MetricSpec
) -> tuple[float | None, bool | None]:
    """Relative change from left to right, and whether it is an improvement.

    ``None`` improvement when the change is zero or unmeasurable, so a view has three
    states to render rather than being forced to call "no change" a win.
    """
    if left is None or right is None or left == 0:
        return None, None
    change = (right - left) / abs(left)
    if change == 0:
        return 0.0, None
    return change, (change > 0) if spec.better == "higher" else (change < 0)
