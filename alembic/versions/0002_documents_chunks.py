"""documents and chunks with row-level security

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-22

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("filename", sa.String, nullable=False),
        sa.Column("mime_type", sa.String, nullable=False),
        sa.Column("storage_path", sa.String, nullable=False),
        sa.Column("chunk_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("status", sa.String, nullable=False, server_default="processing"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_documents_tenant_id", "documents", ["tenant_id"])

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("parent_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chunks.id"), nullable=True),
        sa.Column("metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_tenant_id", "chunks", ["tenant_id"])

    for table in ("documents", "chunks"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY tenant_isolation ON {table}
            USING (tenant_id = current_setting('app.current_tenant_id', true)::uuid)
            """
        )

    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON documents, chunks TO app_user")


def downgrade() -> None:
    op.execute("REVOKE ALL ON documents, chunks FROM app_user")
    op.drop_table("chunks")
    op.drop_table("documents")
