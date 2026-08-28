"""Live latency comparison backing the Memory Hierarchy claim: reading recent
session state from the fast working-memory tier (Redis) is meaningfully
faster than reading it from the durable, transactionally-consistent store
(Postgres) -- per docs/architecture/MAG.md's tier table and the "Evaluation
approach for this batch" section of
docs/superpowers/specs/2026-08-26-mag-foundation-design.md. Baseline and
treatment read the *same* logical thing (the last N turns of one session);
only the storage tier differs.

The Postgres side is seeded with a realistic amount of *other* sessions'
episodes (not just the 20 rows under test) before the comparison runs. An
empty, freshly created table is not an honest "durable store" baseline --
its one page sits fully cached and answers a 20-row indexed lookup about as
fast as anything else on loopback, which measured as a statistical tie
against Redis and made this test flaky (see git history on this file). A
warm production episodic_memory table holds every session's history in one
relation, so a query still has to use its index across that whole working
set; Redis's per-session key (session:{session_id}:working_memory) never
does -- its cost is bounded by that one session's own turns regardless of
how many other sessions exist. Seeding thousands of background rows makes
the benchmark exercise that real, architectural difference instead of an
artifact of both stores being trivially small.
"""
import statistics
import time
import uuid
from datetime import UTC, datetime

from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.queries.retrieve_working_memory import RetrieveWorkingMemory
from src.mag.domain.entities import WorkingMemoryTurn
from src.mag.infrastructure.redis_working_memory_store import RedisWorkingMemoryStore

_TURN_COUNT = 20
_READ_COUNT = 50
_WARMUP_COUNT = 5
_BACKGROUND_SESSIONS = 400
_BACKGROUND_ROWS_PER_SESSION = 30


async def _seed_postgres_episodes(db_session, tenant_id: uuid.UUID, session_id: uuid.UUID) -> None:
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
    # produce a sequential scan, which would mean the measured gap below
    # reflects "scanning an unanalyzed table" rather than the architectural
    # claim this test exists to prove (see module docstring). The EXPLAIN
    # assertion in the test itself is what actually confirms the index gets
    # used, rather than trusting this comment.
    await db_session.execute(text("ANALYZE episodic_memory"))


async def _seed_redis_turns(store: RedisWorkingMemoryStore, session_id: uuid.UUID) -> None:
    for i in range(_TURN_COUNT):
        await store.push_turn(
            session_id,
            WorkingMemoryTurn(role="user", content=f"turn {i}", recorded_at=datetime.now(UTC)),
        )


def _p50_ms(durations_seconds: list[float]) -> float:
    return statistics.median(durations_seconds) * 1000


async def test_working_memory_reads_are_faster_than_a_cold_postgres_round_trip(
    db_session, redis_url
):
    tenant_id = uuid.uuid4()
    postgres_session_id = uuid.uuid4()
    redis_session_id = uuid.uuid4()

    await _seed_postgres_episodes(db_session, tenant_id, postgres_session_id)

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

    # Confirms the planner actually chose an index scan on the seeded,
    # analyzed table -- without this, "the gap comes from the index" is a
    # claim in a comment, not something this test verified. A sequential
    # scan here would mean the measured numbers below don't demonstrate what
    # the module docstring says they demonstrate.
    explain_result = await db_session.execute(
        text(f"EXPLAIN {postgres_query.text}"), postgres_params
    )
    explain_plan = "\n".join(row[0] for row in explain_result)
    assert "Seq Scan" not in explain_plan, (
        f"expected an index scan on episodic_memory, got a sequential scan:\n{explain_plan}"
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
        result.fetchall()
        postgres_durations.append(time.perf_counter() - start)

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
    print(f"Speedup: {postgres_p50 / redis_p50:.2f}x")

    # Loose on purpose -- proving the tiering claim is real and observable
    # with live numbers, not pinning an exact ratio that would make this test
    # flaky against normal system jitter.
    assert redis_p50 < postgres_p50
