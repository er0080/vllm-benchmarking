"""interconnect provenance

Additive nullable columns on `run` and `gpu_host`. Nothing is rewritten.

A run has been able to say which devices it used since the initial schema, and — per
invariant 8 — has never been allowed to infer that from the config text. It has never been
able to say anything about the link between those devices, and on consumer hardware that
link is not a fixed property of the machine.

Enabling peer-to-peer DMA between consumer GPUs is done by rebuilding the *same driver
version* from patched sources. `nvmlSystemGetDriverVersion` reports the version it was
patched from, so a run measured over a direct GPU-to-GPU path and a run measured over one
that stages through host memory agree on driver version, CUDA version, GPU model, vLLM
version, parallelism topology and device indices — every provenance field a run has ever
carried. What they do not agree on is what a tensor-parallel decode step costs. Two
populations, one series, and nothing for `analysis.provenance_differences` to catch.

`peer_access` is the comparability guard. On `run` it is observed
over that run's own `device_indices` rather than over the host's full complement. That
scoping is the point rather than an optimisation: a single-device run has no peer access
to report, and calling it 'unsupported' would put a TP=1 run on one side of a boundary it
cannot be on either side of — which matters most precisely when a single-device run is
being used as the control for a change to the interconnect. `gpu_host` keeps the host-wide
value and the pairwise detail behind it, where an operator checking their setup will look.

Recording the *cause* rather than the effect — which module build was loaded — was tried
and dropped. NVIDIA's `srcversion` is the obvious candidate and it does not work: built
against the patched sources, `nvidia.ko` and `nvidia-uvm.ko` both carry byte-identical
`srcversion` values to the stock modules, because the patch's substance lives in
`src/nvidia/`, which is linked in as a prebuilt object and never reaches modpost's source
list. No module parameters are exposed in sysfs either. A field documented as telling a
patched driver from a stock one, which demonstrably cannot, would be worse than the gap it
was meant to fill.

`String(16)` rather than a native enum, on the same reasoning as `run.synthetic_source`: an
agent reporting a state this schema has not heard of must still be *recorded*, and a native
enum would turn that into an insert that errors. Failing closed here means writing it down.

Every existing row gets NULL, which is the truth about them — protocol 7 had no way to
carry the answer. NULL reads as `PeerAccessStatus.NOT_REPORTED` and is not a synonym for
'peer access was unavailable'; collapsing the two would let a silent absence pass for an
observation, which is the failure the column exists to prevent, reintroduced one layer up.

Revision ID: a4f7c1b83e20
Revises: e1a7c4f9b230
Create Date: (generated)
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a4f7c1b83e20"
down_revision: str | None = "e1a7c4f9b230"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("run", sa.Column("peer_access", sa.String(length=16), nullable=True))
    op.create_index(op.f("ix_run_peer_access"), "run", ["peer_access"])

    op.add_column("gpu_host", sa.Column("peer_access", sa.String(length=16), nullable=True))
    op.add_column(
        "gpu_host",
        sa.Column("peer_access_detail", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("gpu_host", "peer_access_detail")
    op.drop_column("gpu_host", "peer_access")
    op.drop_index(op.f("ix_run_peer_access"), table_name="run")
    op.drop_column("run", "peer_access")
