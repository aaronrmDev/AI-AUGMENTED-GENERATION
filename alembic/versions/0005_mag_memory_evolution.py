"""mag memory evolution: archived_at column and history table

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-29

"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Distinct from valid_until (0003): valid_until means "this fact is
    # wrong/stale" (Invalidate, #63), archived_at means "this fact might
    # still be true but is rarely needed" (Archive, #64). A fact can be
    # either, both, or neither independently.
    op.add_column(
        "semantic_memory",
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )

    # semantic_memory's upsert-by-(user_id, fact_key) unique constraint
    # (0003) means the current row is the ONLY row for that key -- Update
    # (#62) and Refine (#66) both overwrite fact_value, and #62's own
    # worked example is explicit that the old value is "archived with its
    # timestamp rather than simply deleted." This table is where that
    # superseded value goes, since there's nowhere else for it to live.
    op.create_table(
        "semantic_memory_history",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        # original_fact_id is a real FK, not just a same-shaped column:
        # RecordSemanticFact derives semantic_memory.id deterministically
        # (uuid5 of user_id + fact_key), so it stays stable and valid
        # across every future overwrite of the same key -- semantic_memory
        # rows are never deleted, only updated in place.
        sa.Column(
            "original_fact_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("semantic_memory.id"),
            nullable=False,
        ),
        sa.Column(
            "user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False
        ),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fact_key", sa.String, nullable=False),
        sa.Column("fact_value", sa.Text, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("source", sa.String, nullable=False),
        # 'invalidate' is never written here -- Invalidate (#63) flips a
        # status on the SAME value rather than replacing it, so it has
        # nothing to snapshot. Only 'update' and 'refine' appear: Update
        # and Refine are the two operations that overwrite fact_value.
        sa.Column("operation", sa.String, nullable=False),
        sa.Column(
            "superseded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_semantic_memory_history_user_id_fact_key",
        "semantic_memory_history",
        ["user_id", "fact_key"],
    )
    op.create_index(
        "ix_semantic_memory_history_tenant_id", "semantic_memory_history", ["tenant_id"]
    )

    op.execute("ALTER TABLE semantic_memory_history ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE semantic_memory_history FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenant_isolation ON semantic_memory_history
        USING (tenant_id = current_setting('app.current_tenant_id', true)::uuid)
        """
    )
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON semantic_memory_history TO app_user")


def downgrade() -> None:
    op.execute("REVOKE ALL ON semantic_memory_history FROM app_user")
    op.drop_table("semantic_memory_history")
    op.drop_column("semantic_memory", "archived_at")
