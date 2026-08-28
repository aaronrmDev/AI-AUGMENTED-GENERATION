from sqlalchemy import text


async def test_sessions_table_exists_with_rls_enabled(db_session):
    result = await db_session.execute(
        text("SELECT relrowsecurity FROM pg_class WHERE relname = 'sessions'")
    )
    assert result.scalar_one() is True


async def test_tenant_isolation_policy_exists_on_sessions(db_session):
    result = await db_session.execute(
        text(
            "SELECT tablename FROM pg_policies "
            "WHERE policyname = 'tenant_isolation' AND tablename = 'sessions'"
        )
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
        text(
            "SELECT relname, relrowsecurity FROM pg_class "
            "WHERE relname IN ('documents', 'chunks')"
        )
    )
    rows = {row.relname: row.relrowsecurity for row in result}
    assert rows == {"documents": True, "chunks": True}


async def test_tenant_isolation_policy_exists_on_documents_and_chunks(db_session):
    # Scoped to just these two tables (not "every table with this policy
    # name"), matching test_tenant_isolation_policy_exists_on_sessions above
    # -- an unscoped assertion broke the moment migration 0003 added a third
    # RLS table (episodic_memory) under the same policy name; scoping per
    # table is what keeps this test from needing an edit every time a later
    # migration adds another one.
    result = await db_session.execute(
        text(
            "SELECT tablename FROM pg_policies "
            "WHERE policyname = 'tenant_isolation' AND tablename IN ('documents', 'chunks')"
        )
    )
    tables = {row.tablename for row in result}
    assert tables == {"documents", "chunks"}


async def test_episodic_memory_table_exists_with_rls_enabled(db_session):
    result = await db_session.execute(
        text("SELECT relrowsecurity FROM pg_class WHERE relname = 'episodic_memory'")
    )
    assert result.scalar_one() is True


async def test_tenant_isolation_policy_exists_on_episodic_memory(db_session):
    result = await db_session.execute(
        text(
            "SELECT tablename FROM pg_policies "
            "WHERE policyname = 'tenant_isolation' AND tablename = 'episodic_memory'"
        )
    )
    tables = {row.tablename for row in result}
    assert tables == {"episodic_memory"}


async def test_semantic_memory_table_exists_with_rls_enabled(db_session):
    # Corrected post-review: an earlier version of this migration scoped
    # semantic_memory through user_id alone with no RLS, on the mistaken
    # belief that Sessions does the same (it doesn't -- see the design
    # spec's Infrastructure section for the correction). semantic_memory now
    # carries a direct tenant_id column and RLS, same as episodic_memory.
    result = await db_session.execute(
        text("SELECT relrowsecurity FROM pg_class WHERE relname = 'semantic_memory'")
    )
    assert result.scalar_one() is True


async def test_tenant_isolation_policy_exists_on_semantic_memory(db_session):
    result = await db_session.execute(
        text(
            "SELECT tablename FROM pg_policies "
            "WHERE policyname = 'tenant_isolation' AND tablename = 'semantic_memory'"
        )
    )
    tables = {row.tablename for row in result}
    assert tables == {"semantic_memory"}


async def test_semantic_memory_has_a_unique_constraint_on_user_id_and_fact_key(db_session):
    # Regression test: without this constraint, RecordSemanticFact's upsert
    # had no database-level backstop -- two saves with the same
    # (user_id, fact_key) produced two rows resolved nondeterministically by
    # find_by_key's read-time tie-break.
    result = await db_session.execute(
        text(
            "SELECT conname FROM pg_constraint "
            "WHERE conname = 'uq_semantic_memory_user_id_fact_key' AND contype = 'u'"
        )
    )
    assert result.scalar_one_or_none() == "uq_semantic_memory_user_id_fact_key"
