"""Test-support helpers that need to know the schema.

These live with the schema rather than in any one package's ``tests/`` directory because
what they encode is a fact about the model, and every package that touches the database
needs them.

The reset was duplicated in four places before this, and the copies had drifted: two of
them did not delete ``sweep``, so a module that left a sweep behind made the *next*
module fail on a foreign key during its own setup — surfacing as a pile of errors in a
file that had nothing to do with the change, in a different package.
"""

from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncEngine

from vllmbench_db.base import Base
from vllmbench_db.models import Run as _Run  # noqa: F401  (imported to register mappings)


def delete_order() -> list[str]:
    """Every mapped table, children before parents.

    Derived from the metadata rather than listed by hand, because a hand-written list is
    exactly what drifted: add a table, forget to include it, and stale rows leak into
    somebody else's failing test months later. ``sorted_tables`` is in dependency order —
    parents first — so reversing it gives a delete order the foreign keys accept.

    Everything is emptied, including the content-addressed ``server_config``,
    ``workload`` and ``model``. Re-creating those is idempotent by construction, so
    keeping them buys nothing and costs one more thing to remember.
    """
    return [table.name for table in reversed(Base.metadata.sorted_tables)]


async def reset_database(engine: AsyncEngine) -> None:
    """Empty every table this schema defines.

    Alembic's version table is not mapped here, so the applied migration state survives:
    a reset returns the database to empty, never to un-migrated.
    """
    tables = Base.metadata.tables
    async with engine.begin() as connection:
        for name in delete_order():
            await connection.execute(tables[name].delete())


#: Where the integration suite goes when nothing says otherwise.
#:
#: Deliberately not ``vllmbench``, which is the database the compose stack keeps results
#: in. ``reset_database`` empties every table in whatever this resolves to, so a default
#: pointing at the working database means ``make test-integration`` destroys a developer's
#: recorded runs — silently, and with no way to get them back. This repository's first
#: rule is that recorded measurements outrank everything else; a default that deletes them
#: is the opposite of that rule.
#:
#: A name-based refusal in ``reset_database`` would be the more thorough guard and cannot
#: be had: CI legitimately points at ``vllmbench`` and ``vllmbench_seeded``. Making the
#: default safe is what is available, and it is enough — destroying real results now
#: requires naming the database that holds them.
DEFAULT_TEST_DATABASE_URL = "postgresql+psycopg://vllmbench:vllmbench@localhost:5432/vllmbench_test"


def test_database_url() -> str:
    """Where integration tests find Postgres.

    Here rather than in a ``conftest.py`` so every package's tests import it from one
    place. A shared ``conftest`` module only works while exactly one package has a file
    by that name — pytest puts each test directory on ``sys.path``, so a second
    ``conftest.py`` would collide with the first at import time.
    """
    return os.environ.get("DATABASE_URL", DEFAULT_TEST_DATABASE_URL)


# pytest collects by name, so this helper is otherwise gathered as a test in every module
# that imports it — a passing item that asserts nothing, and a warning on every run.
# `__test__` is pytest's documented opt-out; it is not part of the function type, hence
# the ignore.
test_database_url.__test__ = False  # type: ignore[attr-defined]
