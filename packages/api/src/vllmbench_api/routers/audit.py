"""Reading the MCP write audit log.

Read-only, and there is no endpoint here that writes or deletes a row. The log is written
by the MCP surface itself; anything that could edit it afterwards would defeat the point
of having it.

Only the MCP surface is covered, which is why the path says so. A person clicking *create
sweep* in the UI remembers doing it and `initiated_by` records the rest; an agent working
unattended remembers nothing, and a call it *refused* leaves no trace in any other table —
no sweep row, no run rows, nothing. Those refusals are the reason this exists.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query
from sqlalchemy import select

from vllmbench_api.deps import SessionDep
from vllmbench_api.schemas import McpWriteAuditOut
from vllmbench_db.models import McpWriteAudit

router = APIRouter(prefix="/api/mcp-audit", tags=["audit"])

#: Rows per page, and the ceiling on what a caller can ask for. The log grows with every
#: write an agent makes, so an unbounded read of it is a request that gets slower forever.
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000


@router.get("", response_model=list[McpWriteAuditOut])
async def list_write_calls(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1)] = DEFAULT_LIMIT,
    outcome: str | None = None,
) -> list[McpWriteAudit]:
    """Write calls the MCP surface received, newest first.

    ``outcome`` narrows to ``succeeded``, ``refused`` or ``failed``. Refused is usually
    the one worth looking at: a run of them is an agent that has been given a task it
    cannot complete, and nothing else in the database shows that happening.
    """
    stmt = select(McpWriteAudit).order_by(McpWriteAudit.called_at.desc())
    if outcome:
        stmt = stmt.where(McpWriteAudit.outcome == outcome)
    return list((await session.execute(stmt.limit(min(limit, MAX_LIMIT)))).scalars())
