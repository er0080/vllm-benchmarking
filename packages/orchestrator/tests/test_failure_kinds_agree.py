"""The two FailureKind enums are the same enum, and must stay that way.

Its own file, without the integration mark the run-execution tests carry, so it runs in
tier 1. A drifting enum is a plain import-time fact — needing Postgres to notice it would
mean noticing it late.
"""

from __future__ import annotations

from vllmbench_db.enums import FailureKind as StoredKind
from vllmbench_protocol.failures import FailureKind as WireKind


def test_the_two_failure_kind_enums_agree() -> None:
    """Duplicated deliberately, for reasons that do not permit them to differ.

    The protocol package owns the *classification* and has to stay installable into a
    user's vLLM environment, so it must not drag the database package onto a GPU host.
    The database package owns what can be *stored*.

    Without this test the duplication is a slow-motion bug: a kind added on one side only
    would fall back to a phase default in the orchestrator's `_kind_of` and never appear
    in a single query result — no error anywhere, just a category that silently does not
    exist.
    """
    assert {k.value for k in StoredKind} == {k.value for k in WireKind}


def test_every_kind_is_short_enough_for_the_column() -> None:
    """`run.failure_kind` is String(32). A longer name would truncate or error on insert."""
    assert all(len(k.value) <= 32 for k in WireKind)
