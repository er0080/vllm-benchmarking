"""Detect a database whose schema is behind the code that is talking to it.

This exists because of a failure that took a while to recognise: a stale ``migrate``
image left the schema two revisions behind, everything started cleanly, and the first
write failed with ``DatatypeMismatch: column is of type synthetic_source but expression
is of type character varying``. Nothing in that message says "your schema is out of
date", and nothing before it hinted anything was wrong.

Comparing the applied revision against the code's head turns that into one clear line at
startup. It is cheap, it runs once, and the class of bug it catches is otherwise
diagnosed by reading Postgres type errors backwards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SchemaState:
    applied: str | None
    expected: str | None

    @property
    def ok(self) -> bool:
        # Unknown expectation is not a failure: an installed wheel may not ship the
        # migration scripts, and refusing to run then would be worse than not checking.
        if self.expected is None:
            return True
        return self.applied == self.expected

    def describe(self) -> str:
        if self.ok:
            return f"schema at {self.applied or 'unknown'}"
        return (
            f"database schema is at {self.applied or 'nothing'} but this build expects "
            f"{self.expected}. Run the migrate service — writes will fail with confusing "
            "type errors until you do."
        )


def expected_head() -> str | None:
    """The head revision of the migration scripts shipped with this build."""
    migrations = Path(__file__).parent / "migrations"
    if not (migrations / "env.py").is_file():
        return None
    config = Config()
    config.set_main_option("script_location", str(migrations))
    try:
        return ScriptDirectory.from_config(config).get_current_head()
    except Exception as exc:
        log.debug("could not determine migration head: %s", exc)
        return None


async def check_schema_version(engine: AsyncEngine) -> SchemaState:
    applied: str | None = None
    async with engine.connect() as connection:
        result = await connection.execute(text("SELECT version_num FROM alembic_version LIMIT 1"))
        row = result.first()
        if row is not None:
            applied = str(row[0])

    state = SchemaState(applied=applied, expected=expected_head())
    if state.ok:
        log.info("%s", state.describe())
    else:
        log.error("%s", state.describe())
    return state
