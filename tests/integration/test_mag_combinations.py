"""Live verification for MAG Batch G's four combination archetypes -- #19/#78
(Living Agent), #22/#79 (Context Wizard), #24/#80 (Self-Improving Agent), and
#26/#81 (Relationship-Aware Agent). Each test reproduces its own archetype's
worked example from MAG.md structurally, composing already-built primitives
through the real command layer against real Postgres, Qdrant, Neo4j, and (for
the LLM-driven paths) a live Ollama model -- see
docs/superpowers/specs/2026-08-29-mag-combinations-design.md for why this
batch follows MAG's own established narrative-verification convention rather
than RAG Batch E's scenario-script/judge-scoring one.
"""
import uuid

import ollama
from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.commands.capture_episode import CaptureEpisode
from src.mag.application.commands.consolidate_episodes import ConsolidateEpisodes
from src.mag.application.commands.consolidate_procedures import ConsolidateProcedures
from src.mag.application.gating.gate_memories import GateMemories
from src.mag.application.queries.retrieve_by_spreading_activation import (
    SpreadingActivationRetrieval,
)
from src.mag.domain.entities import SemanticMemory
from src.mag.infrastructure.neo4j_memory_graph_repository import Neo4jMemoryGraphRepository
from src.mag.infrastructure.postgres_episodic_memory_repository import (
    PostgresEpisodicMemoryRepository,
)
from src.mag.infrastructure.postgres_procedural_memory_repository import (
    PostgresProceduralMemoryRepository,
)
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.mag.infrastructure.qdrant_episodic_memory_index import QdrantEpisodicMemoryIndex
from src.mag.infrastructure.qdrant_semantic_memory_index import QdrantSemanticMemoryIndex
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
_MODEL_ID = "qwen3.5"


async def _create_user_and_session(db_session, tenant_id: uuid.UUID) -> tuple[uuid.UUID, uuid.UUID]:
    await set_tenant_context(db_session, tenant_id)
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


async def _create_session(db_session, tenant_id: uuid.UUID, user_id: uuid.UUID) -> uuid.UUID:
    await set_tenant_context(db_session, tenant_id)
    session_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO sessions (id, user_id, tenant_id, title) "
            "VALUES (:id, :user_id, :tenant_id, :title)"
        ),
        {"id": session_id, "user_id": user_id, "tenant_id": tenant_id, "title": "t2"},
    )
    await db_session.commit()
    return session_id


async def _memory_graph_repository(neo4j_url) -> Neo4jMemoryGraphRepository:
    url, username, password = neo4j_url
    repository = Neo4jMemoryGraphRepository(url, auth=(username, password))
    await repository.ensure_schema()
    return repository


async def test_living_agent_semantic_knowledge_compounds_across_sessions(
    db_session, qdrant_url, embedding_model, neo4j_url
):
    # #19/#78: capture episodes across two SEPARATE sessions, consolidating
    # after each -- proves the archetype's own sharper claim, that
    # knowledge compounds ACROSS sessions rather than each one starting
    # cold, not just that one consolidation pass extracts a fact.
    tenant_id = uuid.uuid4()
    user_id, session_1 = await _create_user_and_session(db_session, tenant_id)
    episode_repo = PostgresEpisodicMemoryRepository(db_session)
    episode_index = QdrantEpisodicMemoryIndex(qdrant_url)
    await episode_index.ensure_collection()
    fact_repo = PostgresSemanticMemoryRepository(db_session)
    fact_index = QdrantSemanticMemoryIndex(qdrant_url)
    await fact_index.ensure_collection()
    graph = await _memory_graph_repository(neo4j_url)
    chat_model = OllamaChatModel(client=ollama.AsyncClient(), model_id=_MODEL_ID)
    capture = CaptureEpisode(episode_repo, episode_index, embedding_model, chat_model, graph)
    consolidate = ConsolidateEpisodes(
        episode_repo, fact_repo, fact_index, embedding_model, chat_model, graph
    )

    await set_tenant_context(db_session, tenant_id)
    # "input" is what the USER said, "output" the assistant's reply --
    # the preference has to be self-reported by the user for a
    # consolidation prompt asking for "durable facts about the user's
    # preferences" to correctly attribute it to them. An earlier version
    # of this fixture had the user ASKING the assistant its favorite
    # language instead of stating their own, which the live model
    # correctly declined to consolidate into a user-preference fact.
    await capture.execute(
        tenant_id, user_id, session_1,
        {
            "input": "My favorite programming language is Python, especially for data work.",
            "output": "Good to know -- I'll keep that in mind.",
        },
    )
    await capture.execute(
        tenant_id, user_id, session_1,
        {
            "input": "Yeah, Python is the language I reach for first, every time.",
            "output": "Noted.",
        },
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    facts_from_session_1 = await consolidate.execute(tenant_id, user_id, session_1)
    await db_session.commit()
    session_1_summary = [(f.fact_key, f.fact_value) for f in facts_from_session_1]
    print(f"\nLiving Agent, session 1 facts (real Ollama, {_MODEL_ID}): {session_1_summary}")
    assert len(facts_from_session_1) >= 1

    session_2 = await _create_session(db_session, tenant_id, user_id)
    await set_tenant_context(db_session, tenant_id)
    await capture.execute(
        tenant_id, user_id, session_2,
        {
            "input": "I need help structuring a REST API.",
            "output": "FastAPI is a solid choice for that.",
        },
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    facts_from_session_2 = await consolidate.execute(tenant_id, user_id, session_2)
    await db_session.commit()
    session_2_summary = [(f.fact_key, f.fact_value) for f in facts_from_session_2]
    print(f"Living Agent, session 2 facts (real Ollama, {_MODEL_ID}): {session_2_summary}")

    # The loop's own claim: knowledge from session 1 must still be
    # retrievable after session 2's own, entirely separate consolidation
    # pass has run -- not overwritten, not lost, genuinely compounding.
    await set_tenant_context(db_session, tenant_id)
    all_facts = await fact_repo.search_by_similarity(
        embedding_model.embed("What programming language does the user prefer?"),
        user_id, tenant_id, top_k=10,
    )
    assert any("python" in f.fact.fact_value.lower() for f in all_facts)
    await graph.close()


async def test_context_wizard_gating_curates_facts_ahead_of_episodes_end_to_end(
    db_session, qdrant_url, embedding_model, neo4j_url
):
    # #22/#79: retrieval across the episodic and semantic tiers (Memory
    # Hierarchy -- the Foundation batch's own store split) into one
    # candidate pool, gated by GateMemories -- reproducing MAG.md's own
    # "places user preferences and task-critical facts first" worked
    # example through the FULL retrieval-to-gating chain (real
    # CaptureEpisode writes, real search_by_similarity reads), not just
    # GateMemories in isolation the way Batch E's own integration test
    # already covers. Doesn't try to force a specific relationship
    # between the two candidates' raw similarity scores -- an earlier
    # version of this test attempted that (episode content set to the
    # query text verbatim) and got it wrong: CaptureEpisode embeds
    # json.dumps(content, sort_keys=True), the whole content dict, not a
    # bare string the way RecordSemanticFact embeds fact_value alone, so
    # the same "identical text maximizes cosine similarity" trick Batch
    # F's own review-driven fix used for facts doesn't transfer to
    # episodes. HierarchicalAssembly's fact-before-episode ordering is
    # unconditional regardless of either candidate's raw score (already
    # proven at the unit and Batch E/F integration level), so this test
    # doesn't need to re-litigate that -- it demonstrates the
    # COMBINATION works against real retrieved data.
    tenant_id = uuid.uuid4()
    user_id, session_id = await _create_user_and_session(db_session, tenant_id)
    episode_repo = PostgresEpisodicMemoryRepository(db_session)
    episode_index = QdrantEpisodicMemoryIndex(qdrant_url)
    await episode_index.ensure_collection()
    fact_repo = PostgresSemanticMemoryRepository(db_session)
    fact_index = QdrantSemanticMemoryIndex(qdrant_url)
    await fact_index.ensure_collection()
    graph = await _memory_graph_repository(neo4j_url)
    chat_model = OllamaChatModel(client=ollama.AsyncClient(), model_id=_MODEL_ID)
    capture = CaptureEpisode(episode_repo, episode_index, embedding_model, chat_model, graph)

    await set_tenant_context(db_session, tenant_id)
    await capture.execute(
        tenant_id, user_id, session_id,
        {
            "input": "What does the user know about Python?",
            "output": "They mentioned using it for data analysis scripts.",
        },
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    fact_value = "The user's preferred programming language is Python."
    fact = SemanticMemory(
        id=uuid.uuid5(uuid.NAMESPACE_OID, f"semantic_memory:{user_id}:language_preference"),
        user_id=user_id, fact_key="language_preference", fact_value=fact_value,
        embedding=embedding_model.embed(fact_value),
    )
    await fact_repo.save(fact, tenant_id)
    await fact_index.upsert(fact, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    query_embedding = embedding_model.embed("What does the user know about Python?")
    scored_episodes = await episode_repo.search_by_similarity(
        query_embedding, tenant_id, top_k=10
    )
    scored_facts = await fact_repo.search_by_similarity(
        query_embedding, user_id, tenant_id, top_k=10
    )
    assert len(scored_episodes) == 1
    assert len(scored_facts) == 1
    print(
        f"\nContext Wizard, raw scores -- episode: {scored_episodes[0].score:.4f}, "
        f"fact: {scored_facts[0].score:.4f}"
    )

    result = await GateMemories().execute(
        episodes=scored_episodes, facts=scored_facts, graph_nodes=[], token_budget=10_000,
    )

    assert [c.source_type for c in result] == ["fact", "episode"]
    await graph.close()


async def test_self_improving_agent_extracts_a_procedure_from_a_repeated_task_pattern(
    db_session, qdrant_url, embedding_model, neo4j_url
):
    # #24/#80: two episodes recording the SAME task pattern succeeding via
    # the SAME workflow -- MAG.md's own example (five FastAPI deployments
    # distilling into one proven three-step procedure), reproduced with
    # two occurrences (the minimum a "repeated pattern" can mean) rather
    # than five, to keep the live-model call count reasonable while still
    # exercising the real "is this actually repeated" judgment.
    tenant_id = uuid.uuid4()
    user_id, session_id = await _create_user_and_session(db_session, tenant_id)
    episode_repo = PostgresEpisodicMemoryRepository(db_session)
    episode_index = QdrantEpisodicMemoryIndex(qdrant_url)
    await episode_index.ensure_collection()
    procedures_repo = PostgresProceduralMemoryRepository(db_session)
    graph = await _memory_graph_repository(neo4j_url)
    chat_model = OllamaChatModel(client=ollama.AsyncClient(), model_id=_MODEL_ID)
    capture = CaptureEpisode(episode_repo, episode_index, embedding_model, chat_model, graph)
    consolidate_procedures = ConsolidateProcedures(procedures_repo, chat_model)

    await set_tenant_context(db_session, tenant_id)
    await capture.execute(
        tenant_id, user_id, session_id,
        {
            "input": "Deploy my FastAPI app to production",
            "output": "Deployed successfully.",
            "outcome": "success",
            "workflow": ["containerize with Docker", "run under Gunicorn", "front with Nginx"],
        },
    )
    await capture.execute(
        tenant_id, user_id, session_id,
        {
            "input": "Deploy the new FastAPI service the same way as last time",
            "output": "Deployed successfully using the same approach.",
            "outcome": "success",
            "workflow": ["containerize with Docker", "run under Gunicorn", "front with Nginx"],
        },
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    episodes = await episode_repo.get_unconsolidated_by_session(session_id, tenant_id, limit=10)
    assert len(episodes) == 2

    procedures = await consolidate_procedures.execute(tenant_id, user_id, episodes)
    await db_session.commit()

    procedures_summary = [(p.task_pattern, p.workflow) for p in procedures]
    print(f"\nSelf-Improving Agent procedures (real Ollama, {_MODEL_ID}): {procedures_summary}")
    assert len(procedures) >= 1
    procedure = procedures[0]
    assert "deploy" in procedure.task_pattern.lower()
    assert procedure.workflow  # non-empty -- real steps were extracted, not a blank object
    await graph.close()


async def test_relationship_aware_agent_answers_via_relationship_hopping(
    db_session, qdrant_url, embedding_model, neo4j_url
):
    # #26/#81: MAG.md's own "what did I book last month" worked example,
    # through the real command layer. Deliberately does NOT depend on
    # ConsolidateEpisodes/fact extraction succeeding: a one-off booking
    # is exactly the kind of INSTANCE-SPECIFIC content MAG.md's own
    # episodic-memory definition says belongs in episodic memory, not
    # semantic memory (`docs/architecture/MAG.md`'s five-required-
    # properties section) -- an LLM correctly declining to "generalize"
    # a single booking into a timeless fact is a valid, expected outcome,
    # not a test failure. The mechanism this test actually verifies is
    # deterministic and doesn't need an LLM judgment call at all:
    # CaptureEpisode's link_mentions always creates the MENTIONS edge
    # from content["entities"], so the episode containing the booking
    # details is reachable from the entity alone via one hop of graph
    # traversal -- which is what would let a caller answer "what did I
    # book" without ever searching for "booked" or "Paris" as text.
    tenant_id = uuid.uuid4()
    user_id, session_id = await _create_user_and_session(db_session, tenant_id)
    episode_repo = PostgresEpisodicMemoryRepository(db_session)
    episode_index = QdrantEpisodicMemoryIndex(qdrant_url)
    await episode_index.ensure_collection()
    graph = await _memory_graph_repository(neo4j_url)
    chat_model = OllamaChatModel(client=ollama.AsyncClient(), model_id=_MODEL_ID)
    capture = CaptureEpisode(episode_repo, episode_index, embedding_model, chat_model, graph)

    await set_tenant_context(db_session, tenant_id)
    episode = await capture.execute(
        tenant_id, user_id, session_id,
        {
            "input": "I just booked a flight to Paris for next month.",
            "output": "Got it, noted your upcoming trip.",
            "entities": ["Paris"],
        },
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    retrieval = SpreadingActivationRetrieval(graph)
    activated = await retrieval.execute(tenant_id=tenant_id, start_entity_names=["Paris"])

    reached_summary = [(n.node_type, n.hops) for n in activated]
    print(f"\nRelationship-Aware Agent, nodes reached from 'Paris': {reached_summary}")
    episode_nodes = [n for n in activated if n.node_type == "Episode"]
    assert len(episode_nodes) == 1
    reached_episode = episode_nodes[0]
    assert reached_episode.node_id == str(episode.id)
    assert reached_episode.hops == 1
    # The reached node's own content is what actually answers "what did
    # I book" -- proving the traversal reached something USEFUL, not
    # just a node of the right label. properties["content"] round-trips
    # back to a real dict (Neo4jMemoryGraphRepository parses the stored
    # JSON string back for exactly this reason), not a raw string.
    assert "Paris" in reached_episode.properties["content"]["input"]
    await graph.close()
