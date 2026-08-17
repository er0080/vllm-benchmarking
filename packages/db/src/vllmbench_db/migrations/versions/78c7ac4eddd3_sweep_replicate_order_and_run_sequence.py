"""sweep replicate order, and each run's position in its sweep

``sweep.replicate_order`` records whether a point's replicates ran back-to-back or spread
across the matrix. It is stored rather than assumed because it decides what the spread
*means* — repeatability under near-identical conditions, or run-to-run variance — and a
chart drawing error bars is claiming one of them.

``run.sweep_seq`` is the run's position in its sweep's plan. Explicit rather than derived
from ``queued_at``, because the order is a deliberate choice (runs sharing a server config
are kept adjacent so the engine restarts once per config rather than once per run) and an
ordering that matters should not rest on timestamp ties.

Added with a server default rather than plain NOT NULL. Autogenerate proposed the latter,
which cannot run against a ``sweep`` table that already holds rows — the exact class of
migration CI's seeded-database check exists to reject.

Revision ID: 78c7ac4eddd3
Revises: a3f81b2c4d70
Create Date: 2026-08-16 22:33:50.791914
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "78c7ac4eddd3"
down_revision: str | None = "a3f81b2c4d70"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_REPLICATE_ORDER = sa.Enum("grouped", "interleaved", name="replicate_order")


def upgrade() -> None:
    # Created explicitly so the type exists before the column references it, rather than
    # relying on add_column's implicit creation.
    _REPLICATE_ORDER.create(op.get_bind(), checkfirst=True)

    op.add_column("run", sa.Column("sweep_seq", sa.Integer(), nullable=True))
    op.add_column(
        "sweep",
        sa.Column(
            "replicate_order",
            _REPLICATE_ORDER,
            nullable=False,
            server_default="grouped",
        ),
    )


def downgrade() -> None:
    op.drop_column("sweep", "replicate_order")
    op.drop_column("run", "sweep_seq")
    _REPLICATE_ORDER.drop(op.get_bind(), checkfirst=True)
