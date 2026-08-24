"""engine environment provenance

One additive nullable column on `run`. Nothing is rewritten.

`a4f7c1b83e20` gave a run the ability to say which interconnect was underneath it. This
gives it the ability to say what the engine was told to do with that interconnect, which
turns out to be the wider gap of the two.

Neither setting that motivated this appears in the config YAML, so neither reaches the
content-addressed `config_hash`:

  * `NCCL_P2P_LEVEL=SYS` moved per-GPU output throughput 13.4% at concurrency 16 on this
    project's own GPU host — measured, three replicates per arm, standard deviations under
    0.03 tok/s. NCCL declines the peer-to-peer path at `PHB` topology by default no matter
    what the driver permits, so the driver patch alone was worth 0.00%.
  * `VLLM_CUSTOM_ALLREDUCE_PUSH` selects a different all-reduce kernel entirely.

Runs differing in either are byte-identical in every column this schema had. The concrete
collision this was written to prevent: a config whose YAML says
`disable-custom-all-reduce: false` hashes the same whether the engine then ran a kernel
that returns correct sums or one that returns NaN for 100% of elements, because which of
those happens is decided by environment. Fourteen NaN runs are already recorded under one
such hash. Content addressing promises that two runs claiming the same config had
byte-identical effective configuration, and without this column that promise was false.

JSONB rather than columns, and the whole mapping rather than a decoded verdict. Naming the
interesting variables in the schema would mean editing the schema every time vLLM invents
another one — and the variable that forced this column into existence was invented weeks
after the column would have been written. The agent prefix-matches instead
(`vllmbench_agent.hardware.ENGINE_ENV_PREFIXES`), so the next one is captured without a
migration. "Raw before derived" already governs `run.raw_result` for the same reason: keep
what was observed, derive later, because the derivation is what turns out to be wrong.

Not indexed. Unlike `peer_access` this is not a filter or a chart series — it is read per
run and compared between runs, and a GIN index on a mapping nothing queries across would
be cost with no reader. Add one when a query wants it.

Every existing row gets NULL, which is the truth about them: protocol 8 had no way to carry
the answer. NULL is "nobody asked", distinct from `{}`, which is an agent stating the
engine was launched with none of these set. Collapsing those would let silence pass for an
observation — the same failure `peer_access` NULL-vs-`unavailable` avoids, one layer up.

Secret-looking names arrive with their values already replaced by the agent, before this
column ever sees them. A key that changes engine behaviour is provenance; its value is a
secret, and this column is served by a JSON API.

Revision ID: c7e3a91d5f48
Revises: a4f7c1b83e20
Create Date: (generated)
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c7e3a91d5f48"
down_revision: str | None = "a4f7c1b83e20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "run",
        sa.Column("engine_env", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("run", "engine_env")
