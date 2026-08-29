import uuid
from datetime import UTC, datetime

import ollama
from neo4j import AsyncGraphDatabase
from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.commands.archive_memory import ArchiveMemory
from src.mag.application.commands.evolve_memory import EvolveMemory
from src.mag.application.commands.invalidate_memory import InvalidateMemory
from src.mag.application.commands.record_semantic_fact import RecordSemanticFact
from src.mag.application.commands.refine_memory import RefineMemory
from src.mag.application.commands.update_memory import UpdateMemory
from src.mag.application.queries.classify_fact_evolution import ClassifyFactEvolution
from src.mag.infrastructure.neo4j_memory_graph_repository import Neo4jMemoryGraphRepository
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.mag.infrastructure.qdrant_semantic_memory_index import QdrantSemanticMemoryIndex
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
_MODEL_ID = "qwen3.5"


async def _create_user(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
    await set_tenant_context(db_session, tenant_id)
    user_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id) "
            "VALUES (:id, :email, :hashed_password, :tenant_id)"
        ),
        {
            "id": user_id,
            "email": f"{user_id}@example.com",
            "hashed_password": VALID_HASH,
            "tenant_id": tenant_id,
        },
    )
    await db_session.commit()
    return user_id


async def _memory_graph_repository(neo4j_url) -> Neo4jMemoryGraphRepository:
    url, username, password = neo4j_url
    repository = Neo4jMemoryGraphRepository(url, auth=(username, password))
    await repository.ensure_schema()
    return repository


async def _raw_neo4j_fact_property(neo4j_url, fact_id: uuid.UUID, tenant_id: uuid.UUID, prop: str):
    url, username, password = neo4j_url
    driver = AsyncGraphDatabase.driver(url, auth=(username, password))
    try:
        async with driver.session() as session:
            result = await session.run(
                f"MATCH (f:Fact {{id: $id, tenant_id: $tenant_id}}) RETURN f.{prop} AS value",
                id=str(fact_id),
                tenant_id=str(tenant_id),
            )
            record = await result.single()
            return record["value"] if record else None
    finally:
        await driver.close()


async def test_update_round_trip_writes_history_and_updates_both_search_paths(
    db_session, qdrant_url, embedding_model, neo4j_url
):
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repository = PostgresSemanticMemoryRepository(db_session)
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()
    graph = await _memory_graph_repository(neo4j_url)
    record = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        embedding_model=embedding_model,
        memory_graph_repository=graph,
    )
    update = UpdateMemory(semantic_memory_repository=repository, record_semantic_fact=record)

    await set_tenant_context(db_session, tenant_id)
    original = await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="location", fact_value="lives in New York"
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    updated = await update.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        fact_key="location",
        new_fact_value="moved to Berlin last week",
    )
    await db_session.commit()

    # Same id -- the deterministic-uuid5 identity survives an Update, not
    # just a plain re-record.
    assert updated.id == original.id

    await set_tenant_context(db_session, tenant_id)
    from_postgres = await repository.find_by_key(user_id, "location", tenant_id)
    assert from_postgres is not None
    assert from_postgres.fact_value == "moved to Berlin last week"

    from_postgres_search = await repository.search_by_similarity(
        embedding_model.embed("Where does the user live?"), user_id, tenant_id, top_k=5
    )
    assert [f.fact.fact_value for f in from_postgres_search] == ["moved to Berlin last week"]

    from_qdrant = await index.search(
        embedding_model.embed("Where does the user live?"), user_id, tenant_id, top_k=5
    )
    assert [r.fact.fact_value for r in from_qdrant] == ["moved to Berlin last week"]

    history = await repository.find_history(user_id, "location", tenant_id)
    assert len(history) == 1
    assert history[0].fact_value == "lives in New York"
    assert history[0].operation == "update"
    await graph.close()


async def test_update_on_a_previously_archived_fact_preserves_its_archived_status(
    db_session, qdrant_url, embedding_model, neo4j_url
):
    # Regression test for the most severe finding from adversarial review:
    # UpdateMemory/RefineMemory delegated their overwrite to
    # RecordSemanticFact without forwarding the existing fact's
    # archived_at/valid_until, and RecordSemanticFact defaulted both to
    # None -- so correcting an already-archived fact's VALUE silently
    # un-archived it as an unrelated side effect, reappearing in default
    # search with no error, no log, and (before this fix) no test
    # covering the interaction at all. Content and status must stay
    # independent.
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repository = PostgresSemanticMemoryRepository(db_session)
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()
    graph = await _memory_graph_repository(neo4j_url)
    record = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        embedding_model=embedding_model,
        memory_graph_repository=graph,
    )
    archive = ArchiveMemory(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        memory_graph_repository=graph,
    )
    update = UpdateMemory(semantic_memory_repository=repository, record_semantic_fact=record)

    await set_tenant_context(db_session, tenant_id)
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", fact_value="rarely needed but true"
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    archived_at = datetime(2026, 1, 1, tzinfo=UTC)
    await archive.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", archived_at=archived_at
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    updated = await update.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", new_fact_value="corrected value"
    )
    await db_session.commit()

    assert updated.archived_at == archived_at
    assert updated.fact_value == "corrected value"

    await set_tenant_context(db_session, tenant_id)
    from_postgres = await repository.find_by_key(user_id, "k", tenant_id)
    assert from_postgres is not None
    assert from_postgres.fact_value == "corrected value"
    assert from_postgres.archived_at == archived_at

    # Still excluded from default search -- Update didn't silently
    # un-archive it.
    from_postgres_search = await repository.search_by_similarity(
        embedding_model.embed("corrected value"), user_id, tenant_id, top_k=5
    )
    assert from_postgres_search == []

    from_qdrant = await index.search(
        embedding_model.embed("corrected value"), user_id, tenant_id, top_k=5
    )
    assert from_qdrant == []
    await graph.close()


async def test_invalidate_round_trip_excludes_from_both_search_paths_but_not_find_by_key(
    db_session, qdrant_url, embedding_model, neo4j_url
):
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repository = PostgresSemanticMemoryRepository(db_session)
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()
    graph = await _memory_graph_repository(neo4j_url)
    record = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        embedding_model=embedding_model,
        memory_graph_repository=graph,
    )
    invalidate = InvalidateMemory(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        memory_graph_repository=graph,
    )

    await set_tenant_context(db_session, tenant_id)
    fact = await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="pet", fact_value="has a pet named Rex"
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    invalidated_at = datetime(2026, 1, 1, tzinfo=UTC)
    await invalidate.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="pet", invalidated_at=invalidated_at
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    from_postgres_search = await repository.search_by_similarity(
        embedding_model.embed("Does the user have a pet?"), user_id, tenant_id, top_k=5
    )
    assert from_postgres_search == []

    from_qdrant = await index.search(
        embedding_model.embed("Does the user have a pet?"), user_id, tenant_id, top_k=5
    )
    assert from_qdrant == []

    # "exclude it from retrieval, without necessarily replacing it" (#63)
    # -- still directly reachable by key, the value untouched.
    direct = await repository.find_by_key(user_id, "pet", tenant_id)
    assert direct is not None
    assert direct.fact_value == "has a pet named Rex"
    assert direct.valid_until == invalidated_at

    graph_valid_until = await _raw_neo4j_fact_property(
        neo4j_url, fact.id, tenant_id, "valid_until"
    )
    assert graph_valid_until == invalidated_at.isoformat()
    await graph.close()


async def test_archive_round_trip_excludes_from_both_search_paths_but_not_find_by_key(
    db_session, qdrant_url, embedding_model, neo4j_url
):
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repository = PostgresSemanticMemoryRepository(db_session)
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()
    graph = await _memory_graph_repository(neo4j_url)
    record = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        embedding_model=embedding_model,
        memory_graph_repository=graph,
    )
    archive = ArchiveMemory(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        memory_graph_repository=graph,
    )

    await set_tenant_context(db_session, tenant_id)
    fact = await record.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        fact_key="old_address",
        fact_value="used to live on Elm Street",
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    archived_at = datetime(2026, 1, 1, tzinfo=UTC)
    await archive.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="old_address", archived_at=archived_at
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    from_postgres_search = await repository.search_by_similarity(
        embedding_model.embed("Where did the user used to live?"), user_id, tenant_id, top_k=5
    )
    assert from_postgres_search == []

    from_qdrant = await index.search(
        embedding_model.embed("Where did the user used to live?"), user_id, tenant_id, top_k=5
    )
    assert from_qdrant == []

    direct = await repository.find_by_key(user_id, "old_address", tenant_id)
    assert direct is not None
    assert direct.fact_value == "used to live on Elm Street"
    assert direct.archived_at == archived_at

    graph_archived_at = await _raw_neo4j_fact_property(
        neo4j_url, fact.id, tenant_id, "archived_at"
    )
    assert graph_archived_at == archived_at.isoformat()
    await graph.close()


async def test_a_fact_both_invalidated_and_archived_is_excluded_from_both_search_paths(
    db_session, qdrant_url, embedding_model, neo4j_url
):
    # Neither operation's own round-trip test exercises the OTHER field
    # being set too -- the exclusion filters on both search paths are an
    # AND of two independent conditions, and the two commands' targeted,
    # single-field-owning writes (set_valid_until/set_archived_at) are
    # only genuinely race-free if calling both on the same fact actually
    # leaves both fields set correctly, not just each field individually.
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repository = PostgresSemanticMemoryRepository(db_session)
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()
    graph = await _memory_graph_repository(neo4j_url)
    record = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        embedding_model=embedding_model,
        memory_graph_repository=graph,
    )
    invalidate = InvalidateMemory(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        memory_graph_repository=graph,
    )
    archive = ArchiveMemory(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        memory_graph_repository=graph,
    )

    await set_tenant_context(db_session, tenant_id)
    fact = await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", fact_value="stale and rarely needed"
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    invalidated_at = datetime(2026, 1, 1, tzinfo=UTC)
    await invalidate.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", invalidated_at=invalidated_at
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    archived_at = datetime(2026, 2, 1, tzinfo=UTC)
    await archive.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", archived_at=archived_at
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    direct = await repository.find_by_key(user_id, "k", tenant_id)
    assert direct is not None
    # Both fields set -- ArchiveMemory's write didn't clobber Invalidate's.
    assert direct.valid_until == invalidated_at
    assert direct.archived_at == archived_at

    from_postgres_search = await repository.search_by_similarity(
        embedding_model.embed("stale and rarely needed"), user_id, tenant_id, top_k=5
    )
    assert from_postgres_search == []

    from_qdrant = await index.search(
        embedding_model.embed("stale and rarely needed"), user_id, tenant_id, top_k=5
    )
    assert from_qdrant == []

    # Both Neo4j properties set too -- proves set_fact_valid_until and
    # set_fact_archived_at (targeted, single-field writes) don't clobber
    # each other the way a combined upsert_fact_node(dataclasses.replace(
    # stale snapshot)) call used to be able to.
    graph_valid_until = await _raw_neo4j_fact_property(neo4j_url, fact.id, tenant_id, "valid_until")
    graph_archived_at = await _raw_neo4j_fact_property(neo4j_url, fact.id, tenant_id, "archived_at")
    assert graph_valid_until == invalidated_at.isoformat()
    assert graph_archived_at == archived_at.isoformat()
    await graph.close()


async def test_refine_round_trip_against_real_ollama_merges_nuance_without_dropping_either_side(
    db_session, qdrant_url, embedding_model, neo4j_url
):
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repository = PostgresSemanticMemoryRepository(db_session)
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()
    graph = await _memory_graph_repository(neo4j_url)
    record = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        embedding_model=embedding_model,
        memory_graph_repository=graph,
    )
    chat_model = OllamaChatModel(client=ollama.AsyncClient(), model_id=_MODEL_ID)
    refine = RefineMemory(
        semantic_memory_repository=repository, record_semantic_fact=record, chat_model=chat_model
    )

    await set_tenant_context(db_session, tenant_id)
    await record.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        fact_key="language_preference",
        fact_value="prefers Python",
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    refined = await refine.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        fact_key="language_preference",
        new_information="especially for data analysis, though open to Go for CLI tools",
    )
    await db_session.commit()

    print(f"\nRefine merge (real Ollama, {_MODEL_ID}): {refined.fact_value!r}")

    # RefineMemory._merge's exhausted-retries fallback is a literal
    # "{existing}; {new}" concatenation -- for THESE exact fixture values
    # that fallback also contains "python" and "data analysis", so the
    # substring checks below alone can't distinguish a genuine LLM merge
    # from the merge silently failing (confirmed as a real test-quality
    # gap by review). Asserting the result ISN'T that literal fallback
    # string closes the gap: this test now fails, rather than passing
    # green, if qwen3.5 (or a future model swap) stops returning
    # parseable JSON.
    fallback_value = "prefers Python; especially for data analysis, though open to Go for CLI tools"
    assert refined.fact_value != fallback_value

    # MAG.md's own worked example: the original preference isn't
    # discarded, and the new nuance isn't dropped either -- asserted on
    # real semantic content a live model actually produced, not an exact
    # string match against one hardcoded phrasing.
    merged_lower = refined.fact_value.lower()
    assert "python" in merged_lower
    assert "data analysis" in merged_lower or "go" in merged_lower

    # set_tenant_context's SET LOCAL scoping resets on every commit --
    # the commit right after refine.execute() above requires this before
    # the next RLS-sensitive read, the same gotcha this project's other
    # MAG integration tests already document and guard against.
    await set_tenant_context(db_session, tenant_id)
    history = await repository.find_history(user_id, "language_preference", tenant_id)
    assert len(history) == 1
    assert history[0].fact_value == "prefers Python"
    assert history[0].operation == "refine"
    await graph.close()


async def test_evolve_memory_dispatches_update_via_real_ollama_for_a_direct_contradiction(
    db_session, qdrant_url, embedding_model, neo4j_url
):
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repository = PostgresSemanticMemoryRepository(db_session)
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()
    graph = await _memory_graph_repository(neo4j_url)
    record = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        embedding_model=embedding_model,
        memory_graph_repository=graph,
    )
    chat_model = OllamaChatModel(client=ollama.AsyncClient(), model_id=_MODEL_ID)
    evolve = EvolveMemory(
        classify_fact_evolution=ClassifyFactEvolution(
            semantic_memory_repository=repository, chat_model=chat_model
        ),
        update_memory=UpdateMemory(
            semantic_memory_repository=repository, record_semantic_fact=record
        ),
        invalidate_memory=InvalidateMemory(
            semantic_memory_repository=repository,
            semantic_memory_index=index,
            memory_graph_repository=graph,
        ),
        refine_memory=RefineMemory(
            semantic_memory_repository=repository,
            record_semantic_fact=record,
            chat_model=chat_model,
        ),
    )

    await set_tenant_context(db_session, tenant_id)
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="location", fact_value="lives in New York"
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    classification, result = await evolve.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        fact_key="location",
        new_information="moved to Berlin last week",
    )
    await db_session.commit()

    print(
        f"\nEvolveMemory classification (real Ollama, {_MODEL_ID}): "
        f"{classification.operation!r} ({classification.reasoning!r})"
    )
    assert classification.operation == "update"
    assert result is not None
    assert "berlin" in result.fact_value.lower()
    await graph.close()
