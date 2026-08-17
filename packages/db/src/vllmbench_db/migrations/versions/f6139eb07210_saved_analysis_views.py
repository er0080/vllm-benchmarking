"""saved analysis views

A named analysis view: which chart, over which runs, on which axes. Purely additive —
a new table, nothing altered — so it cannot fail against a database holding results,
which is what the forward-only rule is protecting.

``source`` is a column while everything else about the selection lives in ``filters``
jsonb. That asymmetry is deliberate: a view saved over synthetic runs must reopen as
synthetic, and buried in a JSON blob that is one typo away from a saved view quietly
showing real numbers where a developer expected the mock's (invariant 7).

The table stores a query, never a set of run ids. A view reopened next month should
include the runs measured since it was saved; pinning ids would produce something that
silently stops tracking reality while continuing to look current.

Revision ID: f6139eb07210
Revises: 78c7ac4eddd3
Create Date: 2026-08-17 08:24:17.087915
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "f6139eb07210"
down_revision: str | None = "78c7ac4eddd3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "saved_view",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("view", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("filters", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("options", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_saved_view")),
        sa.UniqueConstraint("name", name=op.f("uq_saved_view_name")),
    )
    op.create_index(op.f("ix_saved_view_created_at"), "saved_view", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_saved_view_created_at"), table_name="saved_view")
    op.drop_table("saved_view")
