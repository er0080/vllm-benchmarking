"""Saved analysis views.

A saved view is a *query*, never a set of results: which chart, over which runs, on which
axes. Reopening one next month includes the runs measured since it was saved, which is
what makes it a view of the data rather than a snapshot of it — and a snapshot that keeps
looking current while quietly ceasing to track reality is the worse of the two.

These are interface state, so they are the one place in this service where a JSON blob is
the right shape. The "raw before derived" rule that governs measurements does not apply to
a record of what somebody had selected: nothing queries across it, no chart is drawn from
it, and giving every control a column would mean a migration each time a view gains one.

``source`` is the exception and has a column, because it is load-bearing. A view saved
over synthetic runs must reopen as synthetic; inside the blob it would be one typo from
showing real numbers where a developer expected the mock's (invariant 7).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from vllmbench_api.analysis import RunSource
from vllmbench_api.deps import SessionDep
from vllmbench_api.schemas import SavedViewCreate, SavedViewOut
from vllmbench_db.models import SavedView

router = APIRouter(prefix="/api/views", tags=["views"])


@router.post("", response_model=SavedViewOut, status_code=status.HTTP_201_CREATED)
async def create_view(payload: SavedViewCreate, session: SessionDep) -> SavedView:
    # Validated here rather than trusted from the client: an unrecognised population would
    # be stored and later reopened as something no filter can express.
    if payload.source not in {s.value for s in RunSource}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"source must be one of {', '.join(sorted(s.value for s in RunSource))}",
        )

    view = SavedView(
        name=payload.name.strip(),
        description=payload.description,
        view=payload.view,
        source=payload.source,
        filters=payload.filters,
        options=payload.options,
    )
    session.add(view)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        # Names are the handle people use, so a duplicate is a conflict to report rather
        # than a second view they will not be able to tell apart.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"a saved view named {payload.name!r} already exists",
        ) from exc
    await session.refresh(view)
    return view


@router.get("", response_model=list[SavedViewOut])
async def list_views(session: SessionDep) -> list[SavedView]:
    result = await session.execute(select(SavedView).order_by(SavedView.created_at.desc()))
    return list(result.scalars())


@router.delete("/{view_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_view(view_id: uuid.UUID, session: SessionDep) -> Response:
    view = await session.get(SavedView, view_id)
    if view is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="saved view not found")
    # Deleting a view deletes a bookmark. It touches no run, and there is deliberately no
    # endpoint here that can: runs are immutable once terminal.
    await session.delete(view)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
