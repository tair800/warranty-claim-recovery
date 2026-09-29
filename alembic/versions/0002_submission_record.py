"""submission record: the durable backstop that a disappearing Valkey key cannot defeat

Revision ID: 0002_submission_record
Revises: 0001_initial_schema

### Why this table exists at all

Duplicate prevention was first built entirely on Valkey: `SubmissionGuard.claim_once` is a `SET NX`,
and exactly one caller wins it. That is correct for concurrency and it is not durable, and the
deployment makes the difference concrete rather than theoretical. The Key Value instance this
project runs on is Render's free plan, and **the free plan has no persistence**. Any restart of the
instance — a platform maintenance window, a crash, a redeploy of the instance itself — empties it,
and every idempotency marker goes with it.

Within one case that loss is harmless, because the tool ledger in PostgreSQL replays a completed
filing rather than repeating it. Across two cases it is not. The corpus deliberately carries claims
that reach the system twice under different case identifiers for the same physical recovery — the
`duplicate_invoice` construction exists to exercise exactly that — and the second case's ledger
knows nothing about the first case's filing. Once Valkey forgot the marker, the second case would
win `claim_once`, and the manufacturer would receive the same recovery twice. That is kill
condition E, arriving through an infrastructure event rather than a race.

So the durable answer to "has this recovery identity been filed?" lives here, in PostgreSQL, and
the primary key **is** the constraint. A second claim for an identity is an `INSERT` that conflicts,
and a conflict is a refusal no restart of anything can undo. Valkey keeps the roles it is right
for — the work queue, the leases, and the fast-path claim that serialises concurrent callers before
they reach the database — and none of those needs to survive a restart, because a lease that
vanishes simply expires sooner and a queue can be rebuilt from the cases PostgreSQL says are still
open.

### Why `state` has three values and not two

`CLAIMED` is written **before** the portal is called and `FILED` after it answers, with the effect
identifier. Between the two lies the one genuinely ambiguous moment in the system: the request has
left, and whether it arrived is unknown. A worker that dies there leaves `CLAIMED` behind, and the
correct response is **not** to file again — the filing may well have happened. `AMBIGUOUS` records
that a caller came back, found the claim without an effect, and declined to guess. Nothing here
promises exactly-once transport, because nothing can: the guarantee is one business effect per
recovery identity, and an ambiguous transmission is resolved by reconciliation against the portal,
never by a second transmission.

### Why `approval_fingerprint` is stored

A filing is only valid against the approval that authorised it, and "the approval" means the
approval of *these* numbers and *this* evidence. Recording the fingerprint the filing was made under
lets an auditor reconcile the portal's receipt against the decision a person actually saw, without
trusting the graph's account of either.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_submission_record"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "submission_record",
        # The primary key is the constraint this whole migration exists for.
        sa.Column("recovery_identity", sa.String(length=160), nullable=False),
        sa.Column("case_id", sa.String(length=96), nullable=False),
        sa.Column("case_version", sa.Integer(), nullable=False),
        sa.Column("approval_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("effect_id", sa.String(length=64), nullable=True),
        sa.Column("claimed_on", sa.Date(), nullable=False),
        sa.Column("filed_on", sa.Date(), nullable=True),
        sa.PrimaryKeyConstraint("recovery_identity", name="pk_submission_record"),
        sa.CheckConstraint(
            "state IN ('CLAIMED', 'FILED', 'AMBIGUOUS')", name="ck_submission_record_state"
        ),
        # A FILED row without an effect is a filing nobody can reconcile, and a CLAIMED row with one
        # is a filing the system has forgotten it completed. Both are refused by the database rather
        # than trusted to the code that writes them.
        sa.CheckConstraint(
            "(state = 'FILED') = (effect_id IS NOT NULL)",
            name="ck_submission_record_effect_iff_filed",
        ),
    )
    op.create_index("ix_submission_record_case_id", "submission_record", ["case_id"])


def downgrade() -> None:
    op.drop_index("ix_submission_record_case_id", table_name="submission_record")
    op.drop_table("submission_record")
