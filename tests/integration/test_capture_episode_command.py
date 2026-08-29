import uuid
from datetime import UTC, datetime

import ollama
from neo4j import AsyncGraphDatabase
from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.commands.capture_episode import CaptureEpisode
from src.mag.infrastructure.neo4j_memory_graph_repository import Neo4jMemoryGraphRepository
from src.mag.infrastructure.postgres_episodic_memory_repository import (
    PostgresEpisodicMemoryRepository,
)
from src.mag.infrastructure.qdrant_episodic_memory_index import QdrantEpisodicMemoryIndex
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
_MODEL_ID = "qwen3.5"


async def _create_user_and_session(db_session, tenant_id: uuid.UUID) -> tuple[uuid.UUID, uuid.UUID]:
    await set_tenant_context(db_session, tenant_id)
    now = datetime.now(UTC)
    user_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id, created_at, updated_at) "
            "VALUES (:id, :email, :hashed_password, :tenant_id, :created_at, :updated_at)"
        ),
        {
            "id": user_id, "email": f"{user_id}@example.com", "hashed_password": VALID_HASH,
            "tenant_id": tenant_id, "created_at": now, "updated_at": now,
        },
    )
    session_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO sessions (id, user_id, tenant_id, title) "
            "VALUES (:id, :user_id, :tenant_id, :title)"
        ),
        {"id": session_id, "user_id": user_id, "tenant_id": tenant_id, "title": "t"},
    )
    await db_session.commit()
    return user_id, session_id


async def _memory_graph_repository(neo4j_url) -> Neo4jMemoryGraphRepository:
    url, username, password = neo4j_url
    repository = Neo4jMemoryGraphRepository(url, auth=(username, password))
    await repository.ensure_schema()
    return repository


async def _raw_neo4j_query(neo4j_url, cypher: str, **params: object) -> int:
    # Independent of Neo4jMemoryGraphRepository, matching
    # test_neo4j_memory_graph_repository.py's identical verification
    # pattern -- proves the real write landed, not just that the command
    # accepted the dependency without error.
    url, username, password = neo4j_url
    driver = AsyncGraphDatabase.driver(url, auth=(username, password))
    try:
        async with driver.session() as session:
            result = await session.run(cypher, **params)
            record = await result.single()
            return record["c"]
    finally:
        await driver.close()


async def test_execute_dual_writes_to_real_postgres_and_qdrant(
    db_session, qdrant_url, embedding_model, neo4j_url
):
    # Same seam as test_record_semantic_fact_command.py -- CaptureEpisode's
    # own unit tests inject fakes for both stores, so this is the only test
    # that actually exercises the real dual write together.
    tenant_id = uuid.uuid4()
    user_id, session_id = await _create_user_and_session(db_session, tenant_id)
    repository = PostgresEpisodicMemoryRepository(db_session)
    index = QdrantEpisodicMemoryIndex(qdrant_url)
    await index.ensure_collection()
    chat_model = OllamaChatModel(client=ollama.AsyncClient(), model_id=_MODEL_ID)
    graph = await _memory_graph_repository(neo4j_url)
    command = CaptureEpisode(
        episodic_memory_repository=repository, episodic_memory_index=index,
        embedding_model=embedding_model, chat_model=chat_model,
        memory_graph_repository=graph,
    )

    await set_tenant_context(db_session, tenant_id)
    content = {"input": "what's the weather", "output": "sunny", "tool_calls": []}
    episode = await command.execute(
        tenant_id=tenant_id, user_id=user_id, session_id=session_id, content=content
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    from_postgres = await repository.get_by_session(session_id, tenant_id)
    assert [e.id for e in from_postgres] == [episode.id]
    assert from_postgres[0].content == content

    from_qdrant = await index.search(
        query_embedding=episode.embedding,
        tenant_id=tenant_id,
        session_id=session_id,
        top_k=5,
    )
    assert [s.episode.id for s in from_qdrant] == [episode.id]
    # Not a dedicated live-measurement of the salience score itself (see
    # test_execute_scores_a_failure_episode_more_salient_than_a_routine_one
    # below for that) -- just confirming the real chat_model call this test
    # now depends on actually ran and produced something in the documented
    # [0.0, 1.0] range, not that execute() silently fell back to the
    # parse-failure default every time.
    assert 0.0 <= episode.salience_score <= 1.0

    # The graph write is a real quadruple-store write now (MAG Batch D):
    # confirms the episode and its PARTICIPATED_IN edge actually landed in
    # Neo4j too, not just Postgres/Qdrant.
    episode_node_count = await _raw_neo4j_query(
        neo4j_url,
        "MATCH (e:Episode {id: $id, tenant_id: $tenant_id}) RETURN count(e) AS c",
        id=str(episode.id),
        tenant_id=str(tenant_id),
    )
    assert episode_node_count == 1
    participated_in_count = await _raw_neo4j_query(
        neo4j_url,
        "MATCH (:User {id: $user_id})-[:PARTICIPATED_IN]->(:Session {id: $session_id}) "
        "RETURN count(*) AS c",
        user_id=str(user_id),
        session_id=str(session_id),
    )
    assert participated_in_count == 1
    await graph.close()


async def test_execute_scores_a_failure_episode_more_salient_than_a_routine_one(
    db_session, qdrant_url, embedding_model, neo4j_url
):
    # Real Ollama model (not a fake), same reasoning as
    # test_consolidate_episodes_command.py's live reflection test: a fake
    # chat model can prove the retry/parsing logic works against a SCRIPTED
    # response, but not that a real model's judgment on real content
    # actually differentiates a critical failure from a routine turn the
    # way #74's own language ("weights critical decisions or failures more
    # heavily than routine turns") describes.
    tenant_id = uuid.uuid4()
    user_id, session_id = await _create_user_and_session(db_session, tenant_id)
    repository = PostgresEpisodicMemoryRepository(db_session)
    index = QdrantEpisodicMemoryIndex(qdrant_url)
    await index.ensure_collection()
    chat_model = OllamaChatModel(client=ollama.AsyncClient(), model_id=_MODEL_ID)
    graph = await _memory_graph_repository(neo4j_url)
    command = CaptureEpisode(
        episodic_memory_repository=repository, episodic_memory_index=index,
        embedding_model=embedding_model, chat_model=chat_model,
        memory_graph_repository=graph,
    )

    await set_tenant_context(db_session, tenant_id)
    failure_episode = await command.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        content={
            "input": "deploy the payments service to production",
            "reasoning": "ran the deploy script without checking the health endpoint first",
            "tool_calls": [{"name": "deploy", "args": {"service": "payments"}}],
            "output": "Traceback (most recent call last): ConnectionRefusedError: "
            "database connection failed",
            "outcome": "failure",
            "entities": ["payments-service"],
        },
    )
    await db_session.commit()
    await set_tenant_context(db_session, tenant_id)
    routine_episode = await command.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        content={
            "input": "what's the weather like today",
            "output": "It's sunny with a high of 72F.",
            "outcome": "success",
        },
    )
    await db_session.commit()

    print(
        f"\nSalience scoring (real Ollama, {_MODEL_ID}): "
        f"failure episode={failure_episode.salience_score}, "
        f"routine episode={routine_episode.salience_score}"
    )
    assert failure_episode.salience_score > routine_episode.salience_score

    # Real graph check: spreading activation from the entity the failure
    # episode mentioned reaches that episode in real Neo4j, one hop away
    # (MENTIONS). max_hops=1 deliberately, not 2: at 2 hops, undirected
    # traversal would also reach routine_episode via payments-service ->
    # failure_episode -[:TEMPORALLY_FOLLOWS]- routine_episode (the two
    # episodes are temporally linked since they're in the same session) --
    # that's correct spread_activation behavior (fully covered by
    # test_neo4j_memory_graph_repository.py's dedicated multi-hop/
    # undirected-traversal tests), not something this test is about. This
    # test is specifically about CaptureEpisode's own MENTIONS wiring, so it
    # stays at the hop count that isolates just that.
    activated = await graph.spread_activation(
        tenant_id=tenant_id,
        start_entity_names=["payments-service"],
        max_hops=1,
        decay_factor=0.5,
        activation_threshold=0.05,
    )
    activated_ids = {node.node_id for node in activated}
    assert str(failure_episode.id) in activated_ids
    assert str(routine_episode.id) not in activated_ids
    await graph.close()
