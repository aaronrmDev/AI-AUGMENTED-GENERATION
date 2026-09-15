"""sessions: index each user's sessions by creation time

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-13

"""
import sqlalchemy as sa

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # GET /sessions lists one user's sessions newest first; RLS supplies the tenant.
    op.create_index(
        "ix_sessions_user_id_created_at",
        "sessions",
        ["user_id", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_sessions_user_id_created_at", table_name="sessions")
