"""Engine and session construction."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DEFAULT_URL = "postgresql+psycopg://vllmbench:vllmbench@localhost:5432/vllmbench"


def database_url(*, sync: bool = False) -> str:
    """Resolve the database URL from the environment.

    ``sync`` returns the psycopg driver without the async layer, which Alembic needs for
    its migration context.
    """
    url = os.environ.get("DATABASE_URL", DEFAULT_URL)
    if sync:
        return url.replace("+psycopg_async", "+psycopg")
    return url


def create_engine(url: str | None = None, *, echo: bool = False) -> AsyncEngine:
    return create_async_engine(
        url or database_url(),
        echo=echo,
        pool_pre_ping=True,
        # Sweeps idle for long stretches while a model loads. Recycle rather than
        # discover a dead connection at the moment a result needs writing.
        pool_recycle=1800,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
