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
        sa.Column("fact_key", sa.String, nullable=False),
        sa.Column("fact_value", sa.Text, nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("source", sa.String, nullable=False, server_default=""),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_semantic_memory_user_id", "semantic_memory", ["user_id"])
    # A user's own facts are looked up by key on every RecordSemanticFact call
    # (find-before-write, so a Batch B Update/Refine can tell whether a fact
    # already exists) -- not unique, since two facts can share a key across
    # different users.
    op.create_index(
        "ix_semantic_memory_user_id_fact_key", "semantic_memory", ["user_id", "fact_key"]
    )

    # RLS on episodic_memory only, direct tenant_id column, same pattern as
    # chunks (0002) -- semantic_memory scopes through user_id and no
    # user_id-keyed RLS pattern exists yet anywhere in this schema (see design
    # spec's Migration section for why inventing one here is out of scope).
    op.execute("ALTER TABLE episodic_memory ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE episodic_memory FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenant_isolation ON episodic_memory
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
