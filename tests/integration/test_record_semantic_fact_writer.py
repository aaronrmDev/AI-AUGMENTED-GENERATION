import uuid

from sqlalchemy import text

from src.identity.infrastructure.db import get_sessionmaker, set_tenant_context
from src.mag.infrastructure.neo4j_memory_graph_repository import Neo4jMemoryGraphRepository
from src.mag.infrastructure.qdrant_semantic_memory_index import QdrantSemanticMemoryIndex
from src.orchestration.infrastructure.record_semantic_fact_writer import RecordSemanticFactWriter

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"


async def _create_user(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
    user_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id) "
            "VALUES (:id, :email, :hashed_password, :tenant_id)"
        ),
        {
            "id": user_id, "email": f"{user_id}@example.com", "hashed_password": VALID_HASH,
            "tenant_id": tenant_id,
        },
    )
    await db_session.commit()
    return user_id


async def _facts(db_session, tenant_id, user_id):
    await set_tenant_context(db_session, tenant_id)
    result = await db_session.execute(
        text("SELECT fact_key, fact_value, source FROM semantic_memory WHERE user_id = :user_id"),
        {"user_id": user_id},
    )
    return [tuple(row) for row in result]


async def test_a_user_scoped_source_becomes_one_updatable_fact_visible_only_in_its_tenant(
    db_session, qdrant_url, neo4j_url, embedding_model
):
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()
    url, username, password = neo4j_url
    graph = Neo4jMemoryGraphRepository(url, auth=(username, password))
    await graph.ensure_schema()
    writer = RecordSemanticFactWriter(
        get_sessionmaker(db_session.bind), index, embedding_model, graph
    )
    try:
        await writer.record(tenant_id, user_id, "size-preference", "wears size 10")
        await writer.record(tenant_id, user_id, "size-preference", "wears size 11")
    finally:
        await graph.close()

    assert await _facts(db_session, tenant_id, user_id) == [
        ("source:size-preference", "wears size 11", "freshness-router")
    ]
    assert await _facts(db_session, uuid.uuid4(), user_id) == []
