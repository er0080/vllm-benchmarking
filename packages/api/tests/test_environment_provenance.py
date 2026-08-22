"""Silence about the environment must never read as a clean environment.

The agent reports whether its virtualenv satisfies the constraints everything installed
there declares (issue #60, protocol 6). Every row written before that existed has NULL,
and so would any row from an agent that could not run the check.

The whole value of the check is destroyed by one plausible shortcut — treating the absent
case as fine. These tests pin the four states apart at the layer where that shortcut is
tempting, which is the one that turns a database NULL into JSON.
"""

from __future__ import annotations

import datetime as dt
import uuid

from vllmbench_api.schemas import HostOut
from vllmbench_protocol import EnvironmentStatus


def host(**overrides: object) -> HostOut:
    fields: dict[str, object] = {
        "id": uuid.uuid4(),
        "name": "gpu-1",
        "agent_url": "http://10.0.0.5:9110",
        "created_at": dt.datetime.now(dt.UTC),
    }
    fields.update(overrides)
    return HostOut.model_validate(fields)


class TestTheAbsentCaseIsItsOwnAnswer:
    def test_a_null_status_becomes_not_reported(self) -> None:
        """A row written before protocol 6, or by an agent that stayed quiet."""
        assert host(environment_status=None).environment_status == EnvironmentStatus.NOT_REPORTED

    def test_an_unset_status_becomes_not_reported(self) -> None:
        assert host().environment_status == EnvironmentStatus.NOT_REPORTED

    def test_not_reported_is_not_ok(self) -> None:
        """The assertion this file exists for, stated outright.

        If these two ever compare equal, a host nobody checked is indistinguishable from
        a host that came back clean — and the check has been reduced to decoration.
        """
        assert EnvironmentStatus.NOT_REPORTED != EnvironmentStatus.OK
        assert host(environment_status=None).environment_status != EnvironmentStatus.OK

    def test_a_null_conflict_list_becomes_empty(self) -> None:
        """Empty because there is nothing to show, not because there is nothing wrong.

        The status carries the claim; this list is only ever detail underneath it.
        """
        assert host(environment_status=None).environment_conflicts == []


class TestReportedStatesSurviveIntact:
    def test_ok_is_preserved(self) -> None:
        assert host(environment_status="ok").environment_status == EnvironmentStatus.OK

    def test_conflicts_arrive_with_their_lines(self) -> None:
        conflict = "vllm 0.25.1 requires fastapi[standard]<0.137.0,>=0.133.0, but 0.141.1 installed"
        reported = host(environment_status="conflicts", environment_conflicts=[conflict])
        assert reported.environment_status == EnvironmentStatus.CONFLICTS
        assert reported.environment_conflicts == [conflict]

    def test_unavailable_is_preserved(self) -> None:
        """The check ran and could not answer — a third thing again, not a conflict."""
        found = host(environment_status="unavailable").environment_status
        assert found == EnvironmentStatus.UNAVAILABLE
        assert found != EnvironmentStatus.CONFLICTS
        assert found != EnvironmentStatus.OK
