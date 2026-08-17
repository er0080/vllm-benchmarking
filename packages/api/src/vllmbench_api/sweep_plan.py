"""Turning a sweep's axes into an ordered list of runs.

Two jobs live here, and both are pure functions so they can be tested exhaustively
without a database: deriving a config variant for a tensor-parallel value, and expanding
the matrix into an execution order.

**On generating configs, and invariant 5.** Invariant 5 says configs are native vLLM YAML,
that what is stored is what is executed, and that we validate rather than transform. This
module writes YAML, which deserves an explicit defence rather than a quiet exception.

The invariant forbids an *intermediate schema* — a representation that gets translated on
the way to the engine, so that what runs is not what you read. That is not what happens
here. A TP variant is derived once, at authoring time, and stored as an ordinary
content-addressed ``server_config``. From that moment it is an ordinary config: byte-identical
to what reaches ``vllm serve --config``, hashed by its own text, editable, readable. Nothing
is transformed at execution time, and no run's config is synthesized on the fly.

The edit itself is deliberately the narrowest one that can work: a single value on a single
top-level line, everything else preserved byte for byte, and a refusal when the input is
ambiguous. Rewriting the file through a YAML round-trip would reorder keys, drop comments,
and normalize quoting — turning "the config you wrote" into "the config we re-emitted",
which is the thing invariant 5 actually protects against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vllmbench_db.enums import ReplicateOrder


class SweepPlanError(ValueError):
    """The sweep as described cannot be turned into runs."""


# A top-level `tensor-parallel-size: N` line, with its optional trailing comment.
#
# Anchored at column zero on purpose. An indented occurrence belongs to some nested
# structure we have no business editing, and a `#`-prefixed line is a comment — neither
# is the engine's tensor-parallel setting, and both would be matched by a looser search.
_TP_LINE = re.compile(
    r"^(?P<key>tensor[-_]parallel[-_]size)(?P<sep>[ \t]*:[ \t]*)"
    r"(?P<value>[^#\r\n]*?)(?P<trail>[ \t]*(?:#[^\r\n]*)?)$"
)


def tensor_parallel_variant(yaml_text: str, tensor_parallel_size: int) -> str:
    """Return ``yaml_text`` with its tensor-parallel size set to ``tensor_parallel_size``.

    Preserves every other byte, including comments, key order and the trailing comment on
    the edited line itself. Appends the key when the config does not set it.

    Raises when the config sets it more than once: two top-level keys with the same name
    is already a config whose meaning depends on which one vLLM honours, and guessing
    which to edit would produce a variant that does not mean what its name says.
    """
    if tensor_parallel_size < 1:
        raise SweepPlanError(f"tensor_parallel_size must be at least 1, got {tensor_parallel_size}")

    lines = yaml_text.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if _TP_LINE.match(line.rstrip("\r\n"))]

    if len(matches) > 1:
        raise SweepPlanError(
            f"config sets tensor-parallel-size on {len(matches)} lines "
            f"({', '.join(str(i + 1) for i in matches)}); "
            "which one vLLM honours is not something this should guess at"
        )

    if not matches:
        # Absent, so append. Preserve the file's own line ending style and make sure the
        # existing last line is terminated before adding to it.
        newline = "\r\n" if yaml_text.endswith("\r\n") else "\n"
        prefix = (
            yaml_text if not yaml_text or yaml_text.endswith(("\n", "\r")) else yaml_text + newline
        )
        return f"{prefix}tensor-parallel-size: {tensor_parallel_size}{newline}"

    index = matches[0]
    raw = lines[index]
    body = raw.rstrip("\r\n")
    ending = raw[len(body) :]
    match = _TP_LINE.match(body)
    assert match is not None  # guarded by the search above

    rebuilt = f"{match['key']}{match['sep']}{tensor_parallel_size}{match['trail']}"
    lines[index] = rebuilt + ending
    return "".join(lines)


def read_tensor_parallel_size(yaml_text: str) -> int | None:
    """The tensor-parallel size a config asks for, or None if it does not say.

    A narrow line read, matching how the agent reads the same value. Not a YAML parse:
    invariant 5 keeps the config opaque, and a parser means holding opinions about vLLM
    options that rot with every release.
    """
    for line in yaml_text.splitlines():
        match = _TP_LINE.match(line)
        if match:
            try:
                return int(match["value"].strip().strip("\"'"))
            except ValueError:
                return None
    return None


def validate_tensor_parallel(
    tensor_parallel_size: int, *, host_gpu_count: int, host_name: str
) -> None:
    """Refuse a topology the host cannot provide.

    Caught here rather than left to fail at run time because the failure mode downstream
    is not a clean error. vLLM may start with fewer devices than requested, and the run
    would then be normalized per-GPU against a device count that never existed — a wrong
    number rather than a missing one.
    """
    if tensor_parallel_size < 1:
        raise SweepPlanError(f"tensor_parallel_size must be at least 1, got {tensor_parallel_size}")
    if host_gpu_count and tensor_parallel_size > host_gpu_count:
        raise SweepPlanError(
            f"tensor_parallel_size={tensor_parallel_size} exceeds the {host_gpu_count} "
            f"GPU(s) {host_name} reports. Per-GPU figures would be normalized against a "
            "device count the run never had."
        )


@dataclass(frozen=True)
class PlannedRun:
    """One point of the matrix, at its position in the execution order."""

    seq: int
    config_index: int
    workload_index: int
    replicate_idx: int


def expand(
    *,
    config_count: int,
    workload_count: int,
    replicates: int,
    order: ReplicateOrder = ReplicateOrder.GROUPED,
) -> list[PlannedRun]:
    """Expand the matrix into the order the runs will execute in.

    Configs are the outermost loop in both orderings, because a config change is the only
    axis that costs an engine restart — minutes for a large model. Workloads and
    replicates re-use the running server, so they are cheap to vary underneath it.

    The two orderings differ in where replicates sit, and the difference is about what the
    resulting spread means rather than about speed:

    ``GROUPED`` runs a point's replicates consecutively. One engine start per config.
    ``INTERLEAVED`` runs the whole matrix, then runs it again. Every replicate meets
    different thermal and clock state, so the spread reflects variance a reader would
    actually see — at the cost of ``config_count * replicates`` engine starts.
    """
    if config_count < 1:
        raise SweepPlanError("a sweep needs at least one server config")
    if workload_count < 1:
        raise SweepPlanError("a sweep needs at least one workload")
    if replicates < 1:
        raise SweepPlanError(f"replicates must be at least 1, got {replicates}")

    planned: list[PlannedRun] = []

    if order is ReplicateOrder.INTERLEAVED:
        for replicate in range(replicates):
            for config in range(config_count):
                for workload in range(workload_count):
                    planned.append(
                        PlannedRun(
                            seq=len(planned),
                            config_index=config,
                            workload_index=workload,
                            replicate_idx=replicate,
                        )
                    )
    else:
        for config in range(config_count):
            for workload in range(workload_count):
                for replicate in range(replicates):
                    planned.append(
                        PlannedRun(
                            seq=len(planned),
                            config_index=config,
                            workload_index=workload,
                            replicate_idx=replicate,
                        )
                    )

    return planned


def engine_starts(plan: list[PlannedRun]) -> int:
    """How many times the engine will restart for this plan.

    Surfaced to the author because it is most of the wall-clock cost of a sweep, and the
    difference between the two replicate orderings is otherwise invisible until it is
    running.
    """
    starts = 0
    previous: int | None = None
    for run in plan:
        if run.config_index != previous:
            starts += 1
            previous = run.config_index
    return starts
