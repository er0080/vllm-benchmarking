"""failure kind and timeout budgets

Three additions, all forward-safe against a database holding results.

``run.failure_kind`` makes failures countable. ``run.error`` already held the full story
but is free text, so a sweep with eleven failed points could not be asked whether it hit
one cause or eleven.

``gpu_host`` and ``sweep`` gain model-load and benchmark timeout budgets. Both limits
already existed as hard-coded wire defaults (900s and 3600s); this only makes them
raisable without editing code, so the defaults here are exactly those numbers and no
existing behaviour changes.

The check constraint is added ``NOT VALID`` deliberately. It binds every new and updated
row — a failed run must name its kind — while leaving history alone. Runs that failed
before this column existed keep a NULL, which is the honest value: filling one in would
mean guessing a kind from old free text, and guessing is precisely what the
classification is written not to do. Validating it later, once no NULL-kind failures
remain that anyone cares about, is a one-line follow-up; it is not scheduled, because
the constraint already does its job on everything written from here on.

`alembic check` emits `SAWarning: Can't validate argument 'not_valid'` after this lands.
That is SQLAlchemy 2.0's own reflection constructing the reflected CheckConstraint with a
bare `not_valid=True` rather than the `postgresql_` prefix it requires from us. It is
cosmetic, it comes from library code, and the check itself reports no drift.

Revision ID: b7d2f0a91c53
Revises: d41b91e24340
Create Date: 2026-08-17 23:05:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7d2f0a91c53"
down_revision: str | None = "d41b91e24340"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("run", sa.Column("failure_kind", sa.String(length=32), nullable=True))
    op.create_index(op.f("ix_run_failure_kind"), "run", ["failure_kind"], unique=False)

    # server_default alongside the NOT NULL: without it the ALTER cannot fill existing
    # rows and fails on any database with a host in it.
    op.add_column(
        "gpu_host",
        sa.Column(
            "model_load_timeout_seconds",
            sa.Integer(),
            nullable=False,
            server_default="900",
        ),
    )
    op.add_column(
        "gpu_host",
        sa.Column(
            "benchmark_timeout_seconds",
            sa.Integer(),
            nullable=False,
            server_default="3600",
        ),
    )

    op.add_column("sweep", sa.Column("model_load_timeout_seconds", sa.Integer(), nullable=True))
    op.add_column("sweep", sa.Column("benchmark_timeout_seconds", sa.Integer(), nullable=True))

    # NOT VALID is the point: enforce going forward, do not rewrite history.
    op.create_check_constraint(
        "failed_run_names_its_failure",
        "run",
        "status <> 'failed' OR failure_kind IS NOT NULL",
        postgresql_not_valid=True,
    )


def downgrade() -> None:
    # Forward-only against tables holding results (CLAUDE.md). Present because Alembic
    # expects the function, and because dropping purely additive columns on a database
    # that has none of the new data is harmless — but it is not part of any procedure.
    op.drop_constraint("ck_run_failed_run_names_its_failure", "run", type_="check")
    op.drop_column("sweep", "benchmark_timeout_seconds")
    op.drop_column("sweep", "model_load_timeout_seconds")
    op.drop_column("gpu_host", "benchmark_timeout_seconds")
    op.drop_column("gpu_host", "model_load_timeout_seconds")
    op.drop_index(op.f("ix_run_failure_kind"), table_name="run")
    op.drop_column("run", "failure_kind")
