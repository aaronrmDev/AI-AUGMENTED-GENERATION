"""mag procedural memory and consolidation tracking

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-26

"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "procedural_memory",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False
        ),
        # Direct column, RLS from this table's first version -- not a
        # follow-up fix. 0003's semantic_memory shipped without this and
        # needed a review round (and a factually wrong justification undone)
        # to add it; every tenant-scoped table from here on gets it up front.
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("task_pattern", sa.String, nullable=False),
        # No embedding column: docs/database/DATABASE.md's ProceduralMemory
        # table never had one (unlike EpisodicMemory/SemanticMemory) --
        # retrieval here is by task_pattern match, not vector similarity.
        sa.Column("success_rate", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("last_used", sa.DateTime(timezone=True), nullable=True),
        sa.Column("workflow", postgresql.JSONB, nullable=False, server_default="{}"),
    )
    op.create_index("ix_procedural_memory_user_id", "procedural_memory", ["user_id"])
    op.create_index("ix_procedural_memory_tenant_id", "procedural_memory", ["tenant_id"])
    # UNIQUE, not a plain index -- RecordProcedure upserts by (user_id,
    # task_pattern), same reasoning as semantic_memory's fact_key constraint
    # (0003) and the same id-determinism pattern in RecordProcedure itself.
    op.create_unique_constraint(
        "uq_procedural_memory_user_id_task_pattern",
        "procedural_memory",
        ["user_id", "task_pattern"],
    )

    op.execute("ALTER TABLE procedural_memory ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE procedural_memory FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenant_isolation ON procedural_memory
        USING (tenant_id = current_setting('app.current_tenant_id', true)::uuid)
        """
    )
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON procedural_memory TO app_user")

    # Consolidation tracking: which episodes have already been reflected on.
    # No new index -- get_unconsolidated_by_session filters by session_id
    # (already indexed by 0003) and this column IS NULL, selective enough at
    # this scale without a dedicated partial index; worth revisiting once a
    # real access pattern says otherwise.
    op.add_column(
        "episodic_memory",
        sa.Column("consolidated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("episodic_memory", "consolidated_at")
    op.execute("REVOKE ALL ON procedural_memory FROM app_user")
    op.drop_table("procedural_memory")
