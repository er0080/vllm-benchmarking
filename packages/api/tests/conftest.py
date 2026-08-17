"""Shared database setup for the API's integration tests.

``reset_database`` lives here because it was duplicated across three test modules and
they had already drifted: one of them did not delete ``sweep``, so a module that left a
sweep behind made the *next* module fail on a foreign key during setup. The failure
surfaced as a dozen errors in a file that had nothing to do with the change.

One definition means a new table can only be forgotten once.
"""

from __future__ import annotations

import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

#: Child-to-parent. Telemetry and summaries reference runs; runs reference sweeps, hosts,
#: configs and workloads. TRUNCATE ... CASCADE would be shorter and would also silently
#: erase whatever a future table happens to reference, so the order stays explicit.
_TABLES = (
    "engine_sample",
    "gpu_sample",
    "run_summary",
    "run",
    "sweep",
    "gpu_device",
    "gpu_host",
)


def database_url() -> str:
    return os.environ.get(
        "DATABASE_URL", "postgresql+psycopg://vllmbench:vllmbench@localhost:5432/vllmbench"
    )


async def reset_database(engine: AsyncEngine) -> None:
    """Empty every table these tests write to.

    Configs and workloads are deliberately left alone: they are content-addressed, so
    re-creating one is idempotent and keeping them costs nothing.
    """
    async with engine.begin() as connection:
        for table in _TABLES:
            await connection.execute(text(f"DELETE FROM {table}"))
