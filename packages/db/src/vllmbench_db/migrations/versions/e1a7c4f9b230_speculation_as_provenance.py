"""speculation as provenance

Two additive nullable indexed columns on `run`. Nothing is rewritten.

`run` has recorded `tensor_parallel_size` and `pipeline_parallel_size` since the initial
schema, reported by the agent from what actually ran and — per invariant 8 — never parsed
back out of the config YAML. Speculation is the same kind of fact and had no equivalent, so
a sweep across speculation depths could only be grouped by regexing the *name* somebody gave
the configuration (issue #86).

The reason invariant 8 forbids reading the YAML applies here unchanged. A config saying
`num_speculative_tokens: 3` is not proof the engine drafted three tokens. These values come
from the engine's own `/server_info`, alongside the devices NVML observed.

Three states, kept distinct:

- a method name and a depth — the engine is speculating, and this is how
- `speculative_method = 'none'`, `speculative_tokens = 0` — the engine says it is not
- both NULL — nobody asked it

Every row that predates this is the third, which is the truth about them: protocol 6 had no
way to carry the answer. Defaulting them to 'none' would be inventing a measurement, and it
is exactly the case an ITL comparison needs to treat carefully — see
`analysis.group_warnings`, which will not claim two runs are comparable on an emission-based
metric when it does not know whether either was speculating.

`String(32)` rather than an enum: vLLM's method list grows (`ngram`, `eagle`, `eagle3`,
`medusa`, `mtp`, `suffix`, ...), and a native enum would turn "the engine named a method we
have not seen" into an insert that errors instead of a run that records it. Failing closed
here means recording it, on the same reasoning as `run.synthetic_source`.

Revision ID: e1a7c4f9b230
Revises: 09f50f232684
Create Date: (generated)
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e1a7c4f9b230"
down_revision: str | None = "09f50f232684"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("run", sa.Column("speculative_method", sa.String(length=32), nullable=True))
    op.add_column("run", sa.Column("speculative_tokens", sa.Integer(), nullable=True))
    op.create_index(op.f("ix_run_speculative_method"), "run", ["speculative_method"])
    op.create_index(op.f("ix_run_speculative_tokens"), "run", ["speculative_tokens"])


def downgrade() -> None:
    op.drop_index(op.f("ix_run_speculative_tokens"), table_name="run")
    op.drop_index(op.f("ix_run_speculative_method"), table_name="run")
    op.drop_column("run", "speculative_tokens")
    op.drop_column("run", "speculative_method")
