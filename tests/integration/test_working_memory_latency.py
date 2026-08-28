"""Live latency comparison testing the Memory Hierarchy claim: that reading
recent session state from the fast working-memory tier (Redis) is
meaningfully faster than reading it from the durable, transactionally
-consistent store (Postgres) -- per docs/architecture/MAG.md's tier table.
Baseline and treatment read the *same* logical thing (the last N turns of
one session); only the storage tier differs.

HONEST RESULT, stated up front because an earlier version of this test's own
docstring stated the opposite as settled fact: at this benchmark's scale
(12,000 background rows, real testcontainers Postgres/Redis, Docker Desktop
for Windows over loopback), the claim does NOT reliably hold. A real,
correctly-indexed Postgres query (verified below via EXPLAIN, not assumed)
and a real Redis read complete in comparable, sub-millisecond time -- across
four independent runs during this fix wave, Postgres was at or slightly
FASTER than Redis (p50 ratios of 0.94x-0.98x), not slower. An earlier,
already-merged version of this same test reported a consistent 1.3-1.4x
REDIS speedup across three runs -- that number was real but was produced
under a broken methodology: its ANALYZE call ran through the application's
own app_user connection, which (confirmed empirically, not assumed) lacks
the privilege to actually update table statistics and silently no-ops. Once
ANALYZE was fixed to run through a properly-privileged connection (see
_seed_postgres_episodes below), the previously-claimed gap did not survive.

Why report a null/negative result instead of quietly reverting the fix or
dropping the test: this project's standing rule is to disclose what a live
measurement actually shows, not what the architecture would predict. The
UNDERLYING architectural reasoning MAG.md and the original version of this
docstring gave -- Postgres has to use an index across a table that grows
with every session in the system, Redis's per-session key does not -- is
still sound in principle, and should show up at a larger table size or under
concurrent load than this test exercises. What this test actually
demonstrates at ITS scale is narrower and still worth having: a real,
correctly-indexed Postgres read and a real Redis read are both comparably
fast in the sub-millisecond range, so at this scale the case for the
working-memory tier rests on Redis's bounded cost as the table keeps
growing (untested here), not on a demonstrated latency win at 12,000 rows.
"""
import statistics
import time
import uuid
from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.queries.retrieve_working_memory import RetrieveWorkingMemory
from src.mag.domain.entities import WorkingMemoryTurn
from src.mag.infrastructure.redis_working_memory_store import RedisWorkingMemoryStore

_TURN_COUNT = 20
_READ_COUNT = 50
_WARMUP_COUNT = 5
_BACKGROUND_SESSIONS = 400
_BACKGROUND_ROWS_PER_SESSION = 30


async def _seed_postgres_episodes(
    db_session, database_url: str, tenant_id: uuid.UUID, session_id: uuid.UUID
) -> None:
    now = datetime.now(UTC)
    user_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id, created_at, updated_at) "
            "VALUES (:id, :email, :hashed_password, :tenant_id, :created_at, :updated_at)"
        ),
        {
            "id": user_id,
            "email": f"{user_id}@latency-bench.example.com",
            "hashed_password": "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl",
            "tenant_id": tenant_id,
            "created_at": now,
            "updated_at": now,
        },
    )

    await set_tenant_context(db_session, tenant_id)
    await db_session.execute(
        text(
            "INSERT INTO sessions (id, user_id, tenant_id, title) "
            "VALUES (:id, :user_id, :tenant_id, :title)"
        ),
        {"id": session_id, "user_id": user_id, "tenant_id": tenant_id, "title": "latency bench"},
    )

    embedding = str([0.0] * 384)
    target_rows = [
        {
            "id": uuid.uuid4(),
            "session_id": session_id,
            "tenant_id": tenant_id,
            "content": f'{{"turn": {i}}}',
            "embedding": embedding,
            "timestamp": now,
        }
        for i in range(_TURN_COUNT)
    ]
    episode_insert = text(
        "INSERT INTO episodic_memory (id, session_id, tenant_id, content, embedding, timestamp) "
        "VALUES (:id, :session_id, :tenant_id, :content, :embedding, :timestamp)"
    )
    await db_session.execute(episode_insert, target_rows)
    await db_session.commit()

    # Background load: many *other* sessions' episodes in the same table (see
    # module docstring for why this is what makes the comparison honest).
    # Bulk inserts (one executemany call per table) rather than a Python loop
    # of individual awaited round trips -- this is test setup, not the thing
    # being measured, and thousands of one-by-one round trips would make the
    # suite slow for no reason.
    await set_tenant_context(db_session, tenant_id)
    background_session_rows = [
        {"id": uuid.uuid4(), "user_id": user_id, "tenant_id": tenant_id, "title": "background"}
        for _ in range(_BACKGROUND_SESSIONS)
    ]
    await db_session.execute(
        text(
            "INSERT INTO sessions (id, user_id, tenant_id, title) "
            "VALUES (:id, :user_id, :tenant_id, :title)"
        ),
        background_session_rows,
    )
    background_episode_rows = [
        {
            "id": uuid.uuid4(),
            "session_id": row["id"],
            "tenant_id": tenant_id,
            "content": f'{{"turn": {i}}}',
            "embedding": embedding,
            "timestamp": now,
        }
        for row in background_session_rows
        for i in range(_BACKGROUND_ROWS_PER_SESSION)
    ]
    await db_session.execute(episode_insert, background_episode_rows)
    await db_session.commit()

    # Without this, the planner's row-count estimate for a table it has never
    # analyzed is PostgreSQL's default guess, not the true ~12,000-row
    # reality just seeded -- on a freshly created table that guess can still
    # produce a sequential scan, which would mean the timed reads below are
    # scanning an unanalyzed table rather than exercising the real,
    # correctly-indexed lookup this benchmark is meant to measure (see
    # module docstring for what this benchmark does and doesn't
    # demonstrate). The EXPLAIN assertion in the test itself is what
    # actually confirms the index gets used, rather than trusting this
    # comment.
    #
    # Run through a SEPARATE connection using the migration's own bootstrap
    # credentials (database_url), not db_session (app_user): ANALYZE only
    # updates pg_class's statistics when run by the table's owner or a
    # superuser, and app_user is deliberately neither (this project's own
    # rule against ever running the application as a superuser -- see
    # migration 0001's comment on why app_user exists at all). Confirmed
    # empirically, not assumed: running "ANALYZE episodic_memory" as
    # app_user returns successfully with no error, but leaves
    # pg_stat_user_tables.last_analyze NULL -- a silent no-op that an
    # earlier version of this test didn't know it had.
    bootstrap_engine = create_async_engine(database_url)
    async with bootstrap_engine.connect() as bootstrap_conn:
        await bootstrap_conn.execute(text("ANALYZE episodic_memory"))
        await bootstrap_conn.commit()
    await bootstrap_engine.dispose()


async def _seed_redis_turns(store: RedisWorkingMemoryStore, session_id: uuid.UUID) -> None:
    for i in range(_TURN_COUNT):
        await store.push_turn(
            session_id,
            WorkingMemoryTurn(role="user", content=f"turn {i}", recorded_at=datetime.now(UTC)),
        )


def _p50_ms(durations_seconds: list[float]) -> float:
    return statistics.median(durations_seconds) * 1000


async def test_working_memory_and_postgres_reads_both_return_correct_results_with_live_latency(
    db_session, database_url, redis_url
):
    tenant_id = uuid.uuid4()
    postgres_session_id = uuid.uuid4()
    redis_session_id = uuid.uuid4()

    await _seed_postgres_episodes(db_session, database_url, tenant_id, postgres_session_id)

    store = RedisWorkingMemoryStore(redis_url)
    await _seed_redis_turns(store, redis_session_id)
    query = RetrieveWorkingMemory(store)

    # Baseline: cold-store round trip -- a real query against Postgres's
    # episodic_memory table for the last N turns of a session, exactly what
    # docs/superpowers/specs/2026-08-26-mag-foundation-design.md's evaluation
    # section names as the no-working-memory-tier baseline. One SET LOCAL
    # tenant context before the loop (not per-read) is enough: nothing in the
    # loop commits, so the transaction-local setting survives every read.
    await set_tenant_context(db_session, tenant_id)
    postgres_query = text(
        "SELECT content FROM episodic_memory WHERE session_id = :session_id "
        "ORDER BY timestamp DESC LIMIT :limit"
    )
    postgres_params = {"session_id": postgres_session_id, "limit": _TURN_COUNT}

    # Confirms the planner actually chose ix_episodic_memory_session_id
    # (migration 0003) -- without this, "this is a real, correctly-indexed
    # Postgres read, not a strawman" is a claim in a comment, not something
    # this test verified. Checking only for the absence of "Seq Scan" isn't
    # enough on its own: a bitmap scan against a DIFFERENT index (e.g.
    # ix_episodic_memory_tenant_id) would also satisfy that check while not
    # actually being the index this claim is about.
    explain_result = await db_session.execute(
        text(f"EXPLAIN {postgres_query.text}"), postgres_params
    )
    explain_plan = "\n".join(row[0] for row in explain_result)
    assert "Seq Scan" not in explain_plan, (
        f"expected an index scan on episodic_memory, got a sequential scan:\n{explain_plan}"
    )
    assert "ix_episodic_memory_session_id" in explain_plan, (
        f"expected the session_id index specifically, got:\n{explain_plan}"
    )

    # A handful of untimed reads first, for both paths, so neither measurement
    # is skewed by one-time costs a real hot/warm session would never pay on
    # every read -- asyncpg's first statement on a connection prepares and
    # caches a plan, and redis-py's client opens its TCP connection lazily on
    # its first command. Timing those one-off costs into the comparison would
    # measure connection setup, not the tier difference this test is about.
    for _ in range(_WARMUP_COUNT):
        (await db_session.execute(postgres_query, postgres_params)).fetchall()
        await query.execute(redis_session_id, limit=_TURN_COUNT)

    postgres_durations: list[float] = []
    for _ in range(_READ_COUNT):
        start = time.perf_counter()
        result = await db_session.execute(postgres_query, postgres_params)
        rows = result.fetchall()
        postgres_durations.append(time.perf_counter() - start)
    assert len(rows) == _TURN_COUNT

    # Treatment: the fast tier -- RetrieveWorkingMemory reading Redis.
    redis_durations: list[float] = []
    for _ in range(_READ_COUNT):
        start = time.perf_counter()
        turns = await query.execute(redis_session_id, limit=_TURN_COUNT)
        redis_durations.append(time.perf_counter() - start)
        assert len(turns) == _TURN_COUNT

    await store._client.aclose()

    postgres_p50 = _p50_ms(postgres_durations)
    redis_p50 = _p50_ms(redis_durations)

    total_background_rows = _BACKGROUND_SESSIONS * _BACKGROUND_ROWS_PER_SESSION
    print(f"\nBackground episodic_memory rows seeded: {total_background_rows}")
    print(f"Postgres (cold store) read p50 over {_READ_COUNT} reads: {postgres_p50:.3f} ms")
    print(f"Redis (working memory) read p50 over {_READ_COUNT} reads: {redis_p50:.3f} ms")
    print(f"Postgres/Redis p50 ratio: {postgres_p50 / redis_p50:.2f}x")

    # No directional assertion here on purpose -- see the module docstring.
    # A prior version of this test asserted redis_p50 < postgres_p50, which
    # held under a broken ANALYZE call (silently a no-op) and stopped
    # holding once ANALYZE was fixed to actually update table statistics:
    # across four independent runs with the corrected methodology, Postgres
    # was consistently AT OR SLIGHTLY FASTER than Redis (0.94x-0.98x) at
    # this benchmark's scale, not slower. Asserting a specific direction
    # here would mean asserting whichever one happened to be true on the
    # day this was last edited -- exactly the kind of unverified claim this
    # project's evaluation reports are supposed to catch, not commit. What
    # this test actually proves: both reads return correct, complete
    # results (checked above), and real p50 latency for each is measured
    # and printed for disclosure -- not gated on a specific bound, since
    # this suite's own timing is dominated by testcontainers/Docker Desktop
    # overhead that varies run to run.
    assert postgres_p50 > 0
    assert redis_p50 > 0
