"""freshness-aware data router: data_sources and data_source_versions

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-13

"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

_TABLES = ("data_sources", "data_source_versions")


def upgrade() -> None:
    op.create_table(
        "data_sources",
        # Not server-defaulted: ids are derived from (tenant, user, key) so a retried
        # ingestion replaces the same RAG document (freshness_router.source_id_for).
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Deleting a user removes their user-scoped sources, and those sources' versions
        # cascade in turn.
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("source_key", sa.String, nullable=False),
        sa.Column("scope", sa.String, nullable=False),
        sa.Column("expected_change_interval", sa.Interval, nullable=False),
        sa.Column("route", sa.String, nullable=False),
        sa.Column("content_hash", sa.String, nullable=False),
        sa.Column("last_changed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cached_until", sa.DateTime(timezone=True), nullable=True),
        # Set before a change's effects run, cleared by the save that records the change.
        # Refresh and review skip a source while it is set.
        sa.Column("pending_content_hash", sa.String, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint("scope IN ('tenant', 'user')", name="ck_data_sources_scope"),
        sa.CheckConstraint(
            "route IN ('rag_only', 'cag_with_rag_backup', 'mag')", name="ck_data_sources_route"
        ),
        sa.CheckConstraint(
            "expected_change_interval > interval '0'", name="ck_data_sources_interval_positive"
        ),
        # A user-scoped source belongs to exactly one user; a tenant-scoped one to none.
        sa.CheckConstraint(
            "(scope = 'user') = (user_id IS NOT NULL)", name="ck_data_sources_scope_user"
        ),
    )
    # NULLS NOT DISTINCT (PostgreSQL 15+): a tenant-scoped key is unique per tenant even
    # though its user_id is NULL, while each user can hold a source under the same key.
    op.execute(
        "CREATE UNIQUE INDEX uq_data_sources_tenant_user_key "
        "ON data_sources (tenant_id, user_id, source_key) NULLS NOT DISTINCT"
    )
    op.create_index("ix_data_sources_tenant_id", "data_sources", ["tenant_id"])

    op.create_table(
        "data_source_versions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        # Insertion order, so two versions ingested under one timestamp still have a
        # well-defined latest.
        sa.Column("seq", sa.BigInteger, sa.Identity(always=True), nullable=False),
        sa.Column(
            "data_source_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("data_sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("content_hash", sa.String, nullable=False),
        # NULL for a MAG-routed source: its text lives in the user's MAG fact, and the
        # registry keeps no second copy of personal data.
        sa.Column("content", sa.Text, nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_data_source_versions_source_seq", "data_source_versions", ["data_source_id", "seq"]
    )
    op.create_index("ix_data_source_versions_tenant_id", "data_source_versions", ["tenant_id"])

    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY tenant_isolation ON {table}
            USING (tenant_id = current_setting('app.current_tenant_id', true)::uuid)
            """
        )
    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON data_sources, data_source_versions TO app_user"
    )


def downgrade() -> None:
    op.execute("REVOKE ALL ON data_sources, data_source_versions FROM app_user")
    op.drop_table("data_source_versions")
    op.drop_table("data_sources")
