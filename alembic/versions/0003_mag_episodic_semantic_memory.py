"""mag episodic and semantic memory

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-26

"""
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "episodic_memory",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sessions.id"),
            nullable=False,
        ),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("content", postgresql.JSONB, nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("salience_score", sa.Float, nullable=False, server_default="0.0"),
    )
    op.create_index("ix_episodic_memory_session_id", "episodic_memory", ["session_id"])
    op.create_index("ix_episodic_memory_tenant_id", "episodic_memory", ["tenant_id"])

    op.create_table(
        "semantic_memory",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False
        ),
        # Explicit column, same reasoning as episodic_memory.tenant_id below and
        # chunks.tenant_id in 0002: RLS needs it directly on the row rather than
        # joined in through user_id -> users.tenant_id on every query. An
        # earlier draft of this migration scoped semantic_memory through
        # user_id alone and skipped RLS entirely, on the mistaken belief that
        # sessions does the same -- it doesn't (0001 gives sessions its own
        # tenant_id + RLS policy); this table now follows that same pattern
        # instead of inventing a weaker one.
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fact_key", sa.String, nullable=False),
        sa.Column("fact_value", sa.Text, nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("source", sa.String, nullable=False, server_default=""),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_semantic_memory_user_id", "semantic_memory", ["user_id"])
    op.create_index("ix_semantic_memory_tenant_id", "semantic_memory", ["tenant_id"])
    # UNIQUE, not a plain index: a fact_key is meant to be looked up and
    # overwritten (RecordSemanticFact is an upsert), not accumulated as
    # unbounded duplicates resolved by an arbitrary tie-break at read time.
    # Two different users (even in the same tenant) can still share a
    # fact_key freely -- the constraint is per-user, not global.
    op.create_unique_constraint(
        "uq_semantic_memory_user_id_fact_key", "semantic_memory", ["user_id", "fact_key"]
    )

    for table in ("episodic_memory", "semantic_memory"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY tenant_isolation ON {table}
            USING (tenant_id = current_setting('app.current_tenant_id', true)::uuid)
            """
        )

    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON episodic_memory, semantic_memory TO app_user"
    )


def downgrade() -> None:
    op.execute("REVOKE ALL ON episodic_memory, semantic_memory FROM app_user")
    op.drop_table("semantic_memory")
    op.drop_table("episodic_memory")
