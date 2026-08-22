"""users and sessions with row-level security

Revision ID: 0001
Revises:
Create Date: 2026-08-22

"""
import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("email", sa.String, nullable=False, unique=True),
        sa.Column("hashed_password", sa.String, nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])

    op.create_table(
        "sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String, nullable=True),
        sa.Column("context_budget", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_sessions_tenant_id", "sessions", ["tenant_id"])

    op.execute("ALTER TABLE sessions ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE sessions FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenant_isolation ON sessions
        USING (tenant_id = current_setting('app.current_tenant_id', true)::uuid)
        """
    )

    # A non-superuser, non-owner role for the running application to connect
    # as. The migration itself keeps running as the bootstrap superuser
    # (needed for CREATE TABLE/EXTENSION), but nothing in the application
    # ever should — a superuser connection bypasses every RLS policy above
    # regardless of how correct the policy itself is. The password comes from
    # the environment, never a literal in this file; single quotes are
    # doubled (the standard SQL string-literal escape) because CREATE ROLE's
    # PASSWORD clause is a keyword-value pair in PostgreSQL's grammar, not an
    # expression context, so it cannot take a bound query parameter the way
    # a normal DML value can — this is a deployment-time secret being placed
    # into one-time role-provisioning DDL, not a user-supplied query value,
    # which is what the project's "no raw/string-interpolated SQL" rule
    # exists to prevent.
    #
    # OPERATIONAL NOTE — the CREATE/ALTER ROLE statements below embed the
    # password as a literal, and PostgreSQL logs statement text verbatim when
    # `log_statement` is set to 'ddl' or 'all' (a common production setting).
    # That would write APP_DB_PASSWORD in plaintext into the server log,
    # contradicting docs/security/SECRETS_MANAGEMENT.md's "never logged" rule
    # — and a server log is typically shipped somewhere with a much broader
    # audience than the database itself. Anyone running this migration against
    # a cluster with verbose statement logging should either set
    # `log_statement = 'none'` for the duration of the migration, or provision
    # app_user out of band (an init container, or the cloud provider's own
    # role management) so this block finds the role already present and takes
    # its ALTER branch with a value the operator already controls. There is no
    # way to hide it from within the SQL itself: PASSWORD is a keyword-value
    # pair in PostgreSQL's grammar, so it can never be a bound parameter that
    # the log would elide.
    #
    # The dollar-quote tag is $approle$, not a bare $$: this body is
    # interpolated with a password, and a bare $$ delimiter would be
    # terminated early by a password containing the literal substring "$$",
    # truncating the block mid-statement. This project's own password
    # generation is hex, so that can't happen here — but the tag costs nothing
    # and stops the failure mode from depending on a convention elsewhere in
    # the system continuing to hold.
    app_db_password = os.environ["APP_DB_PASSWORD"].replace("'", "''")
    op.execute(
        f"""
        DO $approle$
        BEGIN
          IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'app_user') THEN
            CREATE ROLE app_user LOGIN PASSWORD '{app_db_password}';
          ELSE
            ALTER ROLE app_user WITH PASSWORD '{app_db_password}';
          END IF;
        END
        $approle$;
        """
    )
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON users, sessions TO app_user")


def downgrade() -> None:
    # Order matters: PostgreSQL refuses to drop a role that still owns objects
    # or holds privileges on them, so the grants come off first, then the
    # tables that were granted, and only then the role itself.
    op.execute("REVOKE ALL ON users, sessions FROM app_user")
    op.drop_table("sessions")
    op.drop_table("users")
    # IF EXISTS because upgrade()'s role provisioning is itself conditional —
    # a downgrade on a database where app_user was created out of band and
    # then dropped separately should not fail here.
    op.execute("DROP ROLE IF EXISTS app_user")
    # The uuid-ossp extension is deliberately NOT dropped. upgrade() creates it
    # with IF NOT EXISTS precisely because it may already have been present,
    # and it is database-scoped rather than owned by these two tables — the
    # remaining five tables docs/database/DATABASE.md specifies will want the
    # same generator. Dropping it here would break anything else in the
    # database that came to depend on it, which is a worse outcome than
    # leaving one idempotent extension behind.
