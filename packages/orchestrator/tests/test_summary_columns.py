"""The flattener's output against the columns it is unpacked into.

Here rather than beside the other flattening tests because this is where the coupling
lives: the orchestrator is what writes ``RunSummary(run_id=..., **flatten_bench_result(...))``,
and it is the package that depends on both sides. The protocol package must stay
installable into a user's vLLM environment, so it cannot depend on the database.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from vllmbench_db.models import RunSummary
from vllmbench_protocol.bench_result import (
    CONTEXT_FIELDS,
    SUMMARY_FIELD_MAP,
    flatten_bench_result,
)

FIXTURE = Path(__file__).parents[2] / "protocol" / "tests" / "fixtures" / "bench_serve_v0.25.1.json"


@pytest.fixture
def payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


class TestTheColumnsItFills:
    """Flattener output against `RunSummary`, in both directions.

    The orchestrator does `RunSummary(run_id=..., **flat)`, so the two must correspond
    exactly. One direction fails loudly and one fails silently, and the silent one is why
    this class exists: a column the flattener never produces stays NULL on every row
    forever, with no error anywhere — a metric that simply does not exist, discovered
    months later by someone asking why a chart is empty.
    """

    def test_every_key_it_produces_is_a_real_column(self, payload: dict[str, Any]) -> None:
        columns = {c.name for c in RunSummary.__table__.columns}
        produced = set(flatten_bench_result(payload))

        assert produced <= columns, f"flattener produces non-columns: {sorted(produced - columns)}"

    def test_every_column_is_filled_by_the_flattener(self, payload: dict[str, Any]) -> None:
        """The direction that fails silently.

        A column added to `RunSummary` without a corresponding entry in
        `SUMMARY_FIELD_MAP` is not an error. It is a metric that is NULL on every row
        ever written, and nothing complains — which is precisely the corruption class
        this repository exists to avoid.
        """
        columns = {c.name for c in RunSummary.__table__.columns} - {"run_id"}
        produced = set(flatten_bench_result(payload))

        missing = columns - produced
        assert not missing, (
            f"columns no flattener output ever fills, so they are NULL on every row: "
            f"{sorted(missing)}"
        )

    def test_the_result_actually_constructs_a_row(self, payload: dict[str, Any]) -> None:
        """The assignment the orchestrator makes, made here.

        Correspondence of *names* is not correspondence of types. This is the cheapest
        place to find out that a mapped value cannot go in the column it is mapped to.
        """

        summary = RunSummary(run_id=uuid.uuid4(), **flatten_bench_result(payload))

        assert summary.successful_requests == 8
        assert summary.extra is not None

    def test_nothing_in_the_payload_is_silently_dropped(self, payload: dict[str, Any]) -> None:
        """The module's own promise, asserted rather than stated.

        Every key is either mapped to a column, deliberately listed as context, or lands
        verbatim in `extra`. A key that fell through all three would be data we were
        handed and threw away — and the raw record would be the only place it survived.
        """
        flat = flatten_bench_result(payload)
        accounted = set(SUMMARY_FIELD_MAP) | CONTEXT_FIELDS | set(flat["extra"])

        assert set(payload) <= accounted, f"dropped: {sorted(set(payload) - accounted)}"
