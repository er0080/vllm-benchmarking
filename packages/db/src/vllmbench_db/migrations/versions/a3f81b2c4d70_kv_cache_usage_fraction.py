"""name kv cache usage for what it actually holds

vLLM's metric is ``vllm:kv_cache_usage_perc`` but the value is a 0..1 fraction — a live
engine with an eighth-full cache reports ``0.11223878211531546``. The column inherited
that misleading suffix, and a chart or export reading ``kv_cache_usage_pct`` would render
11.2% as "0.1%" with nothing to indicate anything was wrong.

A plain rename rather than the add-backfill-drop dance the forward-only rule normally
requires: ``engine_sample`` has never been written to. Telemetry lands in this same
milestone, so there is exactly one moment when this is free, and it is now.

Revision ID: a3f81b2c4d70
Revises: c8094ed880d2
Create Date: 2026-08-17 01:15:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "a3f81b2c4d70"
down_revision: str | None = "c8094ed880d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "engine_sample",
        "kv_cache_usage_pct",
        new_column_name="kv_cache_usage_fraction",
    )


def downgrade() -> None:
    op.alter_column(
        "engine_sample",
        "kv_cache_usage_fraction",
        new_column_name="kv_cache_usage_pct",
    )
