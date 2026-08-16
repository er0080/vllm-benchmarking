"""FastAPI dependencies."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_api.settings import ApiSettings


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    factory = request.app.state.sessions
    async with factory() as session:
        yield session


def get_settings(request: Request) -> ApiSettings:
    return request.app.state.settings


# Annotated aliases rather than `= Depends(...)` defaults: same behaviour, but the
# dependency lives in the type, so it reads as a normal signature and does not trip
# linters that reasonably object to function calls in argument defaults.
SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[ApiSettings, Depends(get_settings)]
