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
from vllmbench_protocol import NO_SPECULATION, PeerAccessStatus


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
    #: True when this figure counts the gaps between *emissions* rather than between
    #: tokens. Speculative decoding delivers several accepted tokens in one emission, so
    #: an emission-based figure grows with drafting depth while the tokens arrive faster.
    #: Both readings are correct; putting them on one axis is what is not. See
    #: :func:`group_warnings`.
    emission_based: bool = False
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
    # Inter-token latency, which is a slight misnomer that upstream's field names fix in
    # place. It is the wait between *emissions*, and only equals the wait between tokens
    # when each emission carries one. Under speculation an emission carries however many
    # drafted tokens the target accepted, so this rises with depth while generation gets
    # faster: measured across the MTP sweep, median ITL went 25.0 → 44.1 ms while per-user
    # throughput went 39.6 → 69.4 tok/s. 44.1 ms / 3.16 accepted tokens ≈ 14.0 ms, which is
    # that run's measured TPOT. Neither figure is wrong; they answer different questions.
    #
    # Kept, charted, and labelled — not dropped. Three tokens arriving every 44 ms reads
    # differently in a streaming UI from one every 25 ms, and for an interactive assistant
    # that is worth measuring. What must not happen is the two being averaged together.
    MetricSpec(
        key="itl_ms_median",
        label="Emission gap median",
        unit="ms",
        better="lower",
        emission_based=True,
        description=(
            "Median wait between streamed emissions. One emission is one token without "
            "speculative decoding, and as many tokens as were accepted with it — so this "
            "is a measure of delivery smoothness, not of generation speed. Use TPOT or "
            "per-user output rate to compare speed across speculation settings."
        ),
    ),
    MetricSpec(
        key="itl_ms_p99",
        label="Emission gap p99",
        unit="ms",
        better="lower",
        emission_based=True,
        description="The wait the slowest percentile of emissions sees. See the median.",
    ),
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
    #: Whether the devices this run used could reach each other's memory directly. A
    #: patched driver reports the version it was patched from, so without this a run over
    #: a direct GPU-to-GPU path and one staging through host memory are identical in every
    #: field above.
    peer_access: str | None = None
    #: The engine's launch environment, filtered to what can change a measurement. None is
    #: a run from before protocol 9; `{}` is an agent reporting none of it was set. The
    #: settings in here are invisible to `config_hash` by construction -- they are not in
    #: the config -- so two runs can agree on every other field above and have run
    #: different collectives.
    engine_env: dict[str, str] | None = None
    #: Set when this run was imported rather than measured here, so its hardware and
    #: vLLM version were *declared by a person* rather than observed by an agent.
    imported_from: str | None = None
    #: Requests that failed during the benchmark, and the ones that did not. Both, so a
    #: warning can say "4 of 8" — the count that tells a reader whether to look harder or
    #: move on, which "4 failed" on its own does not.
    failed_requests: int | None = None
    successful_requests: int | None = None

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

    #: What the engine resolved for speculation, from the same source and under the same
    #: rule. ``"none"`` is the engine stating it was not speculating; ``None`` is nobody
    #: having asked — every run before protocol 7 — and the two must not be merged, or a
    #: chart claims a comparison it has no evidence for.
    speculative_method: str | None = None
    speculative_tokens: int | None = None

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


#: Metrics whose value changes meaning across the speculation boundary, by name, so the
#: warning below names them from the metric table rather than from a second hardcoded list
#: that can drift away from it.
EMISSION_BASED_METRICS: tuple[str, ...] = tuple(m.key for m in METRICS if m.emission_based)


def _speculation_label(record: RunRecord) -> str | None:
    """How this run speculated, as a phrase, or None if nobody asked the engine."""
    method = record.speculative_method
    if method is None:
        return None
    if method == NO_SPECULATION:
        return "no speculation"
    return f"{method} depth {record.speculative_tokens}"


def _completion_ratio(record: RunRecord) -> str:
    """ "4 of 8", or just the failure count when the total is not known.

    A bare "4 failed" is unreadable without the denominator: four of eight is a broken run
    and four of two hundred is a flake.
    """
    failed = record.failed_requests or 0
    completed = record.successful_requests
    if completed is None:
        return f"{failed} failed"
    return f"{completed} of {completed + failed}"


def speculation_warning(records: Sequence[RunRecord]) -> str | None:
    """Say so when a group's emission-based metrics are not measuring one quantity.

    Speculation is *not* a :class:`ComparabilityKey` field, and that is deliberate.
    Comparing speculation settings is the reason someone runs this sweep — partitioning
    them into separate charts would break the tool's headline use, the same way
    partitioning on config hash would. Almost every metric compares fine across the
    boundary: TPOT, TTFT, throughput and acceptance rate all mean one thing throughout.

    Emission-based metrics do not, and they do not fail loudly. A speculating run's
    emission gap is larger than a non-speculating one's *because it is faster*, so the
    chart draws the winner as the loser and nothing marks the bars as different
    quantities. This is the same class of problem as overlaying two vLLM versions, which
    this framework partitions rather than permits silently — and the same remedy applies
    at a lower grade, because here the reader asked for the comparison.

    Depth counts too: three tokens per emission and one token per emission are as
    different from each other as speculating and not.
    """
    states = {_speculation_label(r) for r in records}
    if len(states) < 2:
        return None

    metrics = ", ".join(METRICS_BY_KEY[k].label for k in EMISSION_BASED_METRICS)
    if None in states:
        known = sorted(state for state in states if state is not None)
        return (
            f"{metrics} may not be comparable here: "
            f"{sum(1 for r in records if r.speculative_method is None)} run(s) predate "
            "speculation being recorded, so whether they were speculating is unknown"
            + (f" (the rest: {', '.join(known)})" if known else "")
        )
    return (
        f"mixes {', '.join(sorted(s for s in states if s))}; "
        f"{metrics} count the wait between emissions, and a speculative emission carries "
        "several tokens — so those figures rise with depth even as generation gets faster. "
        "Compare speed with TPOT or per-user output rate."
    )


def peer_access_warning(records: Sequence[RunRecord]) -> str | None:
    """Say so when a group was not all measured over the same interconnect.

    Structured like :func:`speculation_warning`, and for the same reason: this is a
    provenance field added after runs already existed, so "nobody asked" is a state the
    group can be mixed with and is not the same as an observation.

    Splitting the group instead was considered and rejected. Every run recorded before
    protocol 8 has NULL here, so keying on it would fragment every existing deployment's
    history away from everything measured after the upgrade — asserting a difference that
    was never observed, which is the mirror image of the failure this column prevents.
    """
    states = {r.peer_access for r in records}
    if len(states) <= 1:
        return None

    observed = sorted(s for s in states if s and s != PeerAccessStatus.NOT_REPORTED.value)
    unknown = sum(
        1
        for r in records
        if not r.peer_access or r.peer_access == PeerAccessStatus.NOT_REPORTED.value
    )

    if unknown and not observed:
        return None
    if unknown:
        return (
            f"mixes {', '.join(observed)} with {unknown} run(s) that predate peer access "
            "being recorded, so the interconnect underneath them is unknown. A "
            "tensor-parallel run over a direct GPU-to-GPU path and one staging through "
            "host memory differ in every emission-based metric while agreeing on driver "
            "version, so these are not safely one series"
        )
    return (
        f"mixes {', '.join(observed)}; these runs were measured over different "
        "interconnects, which changes what every all-reduce in a tensor-parallel step "
        "costs. Compare them deliberately rather than reading them as one series"
    )


#: Environment whose value is worth naming in a warning rather than counting. Everything
#: the agent captures is recorded; this is only about which differences are worth a
#: sentence, because a group that happens to differ in `CUDA_HOME` is not a finding and a
#: group that differs in `NCCL_P2P_LEVEL` is.
#:
#: A prefix, so a variable nobody has invented yet still gets named. The agent's capture
#: is deliberately wider than this — recording more than we warn about is the safe
#: direction, since the record cannot be added to after the fact and the warning can.
_NOTABLE_ENGINE_ENV_PREFIXES = ("VLLM_", "NCCL_")


def engine_env_warning(records: Sequence[RunRecord]) -> str | None:
    """Say so when a group's runs were launched under different engine settings.

    The gap this closes is specific and was hit in practice. `config_hash` is the hash of
    the config text, so it cannot see anything set in the environment — and a driver patch
    plus `NCCL_P2P_LEVEL=SYS` moved per-GPU throughput 13.4% at concurrency 16 with the
    config text byte-identical. Without this, that is one series.

    Structured like :func:`peer_access_warning`: runs recorded before protocol 9 carry
    None, and a group mixing "nobody asked" with real observations is not evidence of a
    difference. Only keys actually disagreeing between two runs that both reported are
    named — an absent key in one reporting run and a set key in the other is a real
    disagreement, and is named.
    """
    reported = [r.engine_env for r in records if r.engine_env is not None]
    if len(reported) < 2:
        return None

    notable: set[str] = set()
    for env in reported:
        notable.update(k for k in env if k.startswith(_NOTABLE_ENGINE_ENV_PREFIXES))

    differing = sorted(k for k in notable if len({env.get(k) for env in reported}) > 1)
    if not differing:
        return None

    unknown = sum(1 for r in records if r.engine_env is None)
    detail = (
        f"mixes runs launched with different {', '.join(differing)}. These are not in the "
        "config, so the runs share a config hash while the engine did something different "
        "under it — a different all-reduce kernel or a different peer-to-peer level "
        "changes throughput without changing a single line of YAML"
    )
    if unknown:
        detail += f", and {unknown} run(s) predate the engine environment being recorded at all"
    return detail


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

    # Runs that did not complete every request. This used to say they "understate the
    # configuration", on the reasoning that throughput divides by the whole benchmark
    # duration while the failed requests contribute no tokens. Measured against a healthy
    # replicate of the same configuration on the same engine, that is not what happens:
    #
    #     healthy   8 of 8   42.8s    94.4 output tok/s   53.2 tok/s per user
    #     partial   5 of 8   16.4s   117.7                45.2
    #     partial   4 of 8   10.3s   125.8                65.8
    #     partial   5 of 8   15.9s   121.4                45.2
    #
    # Throughput rose in every one. The assumption the old wording rested on is that the
    # benchmark still ran its full course — but a run that loses its requests also stops,
    # at 10.3s against 42.8s, so the denominator is truncated along with the numerator and
    # which shrinks further depends on when the engine died. Latency moved both ways too.
    #
    # So no direction is claimed. These points measure a shorter benchmark under changing
    # conditions, which is a different thing from this configuration measured badly.
    #
    # A run where *every* request failed never reaches this: it is refused at the
    # flattening layer, because its zeros would read as the fastest result on the chart
    # rather than as the absence of one.
    partial = [r for r in records if r.failed_requests]
    if partial:
        worst = min(_completion_ratio(r) for r in partial)
        warnings.append(
            f"{len(partial)} run(s) did not complete every request (as few as {worst}); "
            "a benchmark that loses requests also stops early, so both what was counted "
            "and the window it was divided by are cut short. Those points describe a "
            "shorter benchmark under changing conditions, not a degraded measurement of "
            "this configuration, and they lean no predictable way"
        )

    speculation = speculation_warning(records)
    if speculation:
        warnings.append(speculation)

    interconnect = peer_access_warning(records)
    if interconnect:
        warnings.append(interconnect)

    engine_env = engine_env_warning(records)
    if engine_env:
        warnings.append(engine_env)

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

    #: How this point's runs speculated, when they agree. ``None`` when they do not, or
    #: when nobody asked the engine — a point whose replicates disagree about whether they
    #: were speculating cannot be plotted as one speculation setting, and saying so beats
    #: picking one of them.
    speculative_method: str | None
    speculative_tokens: int | None

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


def _speculation_of(records: Sequence[RunRecord]) -> tuple[str | None, int | None]:
    """How this point speculated, if its replicates agree.

    Unlike topology, disagreement here is not resolved by taking the largest. A point whose
    replicates disagree is one where the drafter loaded for some runs and not others, and
    the honest summary of "three runs, one of which was not speculating" is not a
    speculation setting at all. Reporting NULL routes it into the mixed-group warning
    rather than into a series it only half belongs to.
    """
    states = {(r.speculative_method, r.speculative_tokens) for r in records}
    if len(states) != 1:
        return None, None
    return states.pop()


def build_point(records: Sequence[RunRecord]) -> Point:
    if not records:  # pragma: no cover - guarded by the caller's grouping
        raise ValueError("a point needs at least one run")
    unfinished = dt.datetime.min.replace(tzinfo=dt.UTC)
    ordered = sorted(records, key=lambda r: (r.finished_at or unfinished, r.replicate_idx))
    head = ordered[0]
    tp, pp, gpus = _topology_of(ordered)
    spec_method, spec_tokens = _speculation_of(ordered)

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
        speculative_method=spec_method,
        speculative_tokens=spec_tokens,
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
    # Invalidating, alongside GPU model and bench-client location, because it changes the
    # machine underneath the measurement rather than the configuration being measured. It
    # does not stop the deliberate side-by-side — that is the comparison it exists for.
    ("peer_access", "Peer access", True),
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
