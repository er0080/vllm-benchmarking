"""telemetry retention record

One new leaf table. Additive, and it touches nothing that holds results.

`run_telemetry_pruned` records that a run's telemetry was deleted under a retention
policy, as opposed to never having been recorded. Those two states look identical on a
run detail page — an empty timeline — and call for opposite responses: one is the policy
working, the other is a sampling bug that has been quietly losing diagnostic data.

A separate table rather than a column on `run` because a terminal run is immutable and a
trigger enforces it. This is data *about* a run, not a correction to what it measured.

Revision ID: c3a8e17f4d92
Revises: b7d2f0a91c53
Create Date: 2026-08-18 00:15:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c3a8e17f4d92"
down_revision: str | None = "b7d2f0a91c53"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "run_telemetry_pruned",
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("pruned_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["run.id"],
            name=op.f("fk_run_telemetry_pruned_run_id_run"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_run_telemetry_pruned")),
    )


def downgrade() -> None:
    op.drop_table("run_telemetry_pruned")
