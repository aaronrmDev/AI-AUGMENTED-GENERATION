from sqlalchemy import text


async def test_sessions_table_exists_with_rls_enabled(db_session):
    result = await db_session.execute(
        text("SELECT relrowsecurity FROM pg_class WHERE relname = 'sessions'")
    )
    assert result.scalar_one() is True


async def test_tenant_isolation_policy_exists_on_sessions(db_session):
    result = await db_session.execute(
        text("SELECT tablename FROM pg_policies WHERE policyname = 'tenant_isolation' AND tablename = 'sessions'")
    )
    tables = {row.tablename for row in result}
    assert tables == {"sessions"}


async def test_users_table_has_no_rls_policy(db_session):
    # Deliberate: users is the tenant-identity root, not tenant-scoped data —
    # see the note above Step 5 for why RLS on users would break login/
    # registration. This test guards against a future edit accidentally
    # reintroducing a policy on users the way the original draft of this
    # migration did.
    result = await db_session.execute(
        text("SELECT relrowsecurity FROM pg_class WHERE relname = 'users'")
    )
    assert result.scalar_one() is False


async def test_documents_and_chunks_tables_exist_with_rls_enabled(db_session):
    result = await db_session.execute(
        text("SELECT relname, relrowsecurity FROM pg_class WHERE relname IN ('documents', 'chunks')")
    )
    rows = {row.relname: row.relrowsecurity for row in result}
    assert rows == {"documents": True, "chunks": True}


async def test_tenant_isolation_policy_exists_on_documents_and_chunks(db_session):
    result = await db_session.execute(
        text("SELECT tablename FROM pg_policies WHERE policyname = 'tenant_isolation'")
    )
    tables = {row.tablename for row in result}
    assert tables == {"sessions", "documents", "chunks"}
