"""environment consistency

Three additive nullable columns. Nothing is rewritten and nothing that holds results is
touched.

The agent installs into vLLM's own virtualenv — that is the documented deployment — so its
dependency resolution and vLLM's ceilings meet in one environment with nothing arbitrating
between them. They had already diverged on the first real GPU host, and nothing said so
(issue #60). Protocol 6 has the agent report whether its environment satisfies the
constraints everything installed there declares; this is where that lands.

`gpu_host` carries the current answer and the conflict lines, because the host is where
somebody fixes it. `run` carries only the status, point-in-time, alongside the other
provenance columns: what a result needs to state is whether it was measured on a coherent
machine, not the full diagnostics, which are identical across every run of a sweep.

NULL is a third state, not a synonym for "fine". It means an agent older than protocol 6,
which could not say — and reading code substitutes `EnvironmentStatus.NOT_REPORTED` so the
distinction survives instead of being rounded down to a clean bill of health. Existing rows
are therefore correct as NULL: they were measured by agents that never ran this check.

Revision ID: c2e221f68704
Revises: c3a8e17f4d92
Create Date: 2026-08-22 19:32:46.628860
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c2e221f68704"
down_revision: str | None = "c3a8e17f4d92"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("gpu_host", sa.Column("environment_status", sa.String(length=16), nullable=True))
    op.add_column(
        "gpu_host",
        sa.Column("environment_conflicts", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column("run", sa.Column("environment_status", sa.String(length=16), nullable=True))
    # Indexed because the question it answers is a filter — "show me the runs measured on
    # a host that was not internally consistent" — over a table that grows without bound.
    op.create_index(
        op.f("ix_run_environment_status"), "run", ["environment_status"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_run_environment_status"), table_name="run")
    op.drop_column("run", "environment_status")
    op.drop_column("gpu_host", "environment_conflicts")
    op.drop_column("gpu_host", "environment_status")
