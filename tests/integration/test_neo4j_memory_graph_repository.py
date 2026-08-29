import uuid
from datetime import UTC, datetime

import pytest
from neo4j import AsyncGraphDatabase

from src.mag.domain.entities import EpisodicMemory, SemanticMemory
from src.mag.infrastructure.neo4j_memory_graph_repository import Neo4jMemoryGraphRepository


def _episode(content: dict | None = None) -> EpisodicMemory:
    return EpisodicMemory(
        id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        content=content or {"input": "hi"},
        embedding=[0.1] * 384,
        timestamp=datetime.now(UTC),
    )


def _fact() -> SemanticMemory:
    return SemanticMemory(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        fact_key="favorite_color",
        fact_value="blue",
        embedding=[0.1] * 384,
    )


@pytest.fixture
async def repo(neo4j_url):
    url, username, password = neo4j_url
    repository = Neo4jMemoryGraphRepository(url, auth=(username, password))
    await repository.ensure_schema()
    yield repository
    await repository.close()


async def _raw_query(neo4j_url: tuple[str, str, str], cypher: str, **params: object) -> int:
    # Verifies real graph state independently of Neo4jMemoryGraphRepository
    # itself -- opens its own driver connection off the same neo4j_url
    # fixture the `repo` fixture uses, rather than reaching into `repo`'s
    # internals, matching how the Postgres integration tests verify state
    # through their own db_session rather than a repository's private
    # connection.
    url, username, password = neo4j_url
    driver = AsyncGraphDatabase.driver(url, auth=(username, password))
    try:
        async with driver.session() as session:
            result = await session.run(cypher, **params)
            record = await result.single()
            return record["c"]
    finally:
        await driver.close()


async def test_upsert_episode_node_is_idempotent(repo, neo4j_url):
    tenant_id = uuid.uuid4()
    episode = _episode({"input": "hello", "entities": ["alice"]})

    await repo.upsert_episode_node(episode, tenant_id)
    await repo.upsert_episode_node(episode, tenant_id)

    count = await _raw_query(
        neo4j_url, "MATCH (n:Episode {id: $id}) RETURN count(n) AS c", id=str(episode.id)
    )
    assert count == 1


async def test_upsert_fact_node_is_idempotent(repo, neo4j_url):
    tenant_id = uuid.uuid4()
    fact = _fact()

    await repo.upsert_fact_node(fact, tenant_id)
    await repo.upsert_fact_node(fact, tenant_id)

    count = await _raw_query(
        neo4j_url, "MATCH (n:Fact {id: $id}) RETURN count(n) AS c", id=str(fact.id)
    )
    assert count == 1


async def test_link_participated_in_creates_user_session_and_edge(repo, neo4j_url):
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()

    await repo.link_participated_in(user_id, session_id, tenant_id)

    count = await _raw_query(
        neo4j_url,
        "MATCH (u:User {id: $user_id})-[:PARTICIPATED_IN]->(s:Session {id: $session_id}) "
        "RETURN count(*) AS c",
        user_id=str(user_id),
        session_id=str(session_id),
    )
    assert count == 1


async def test_link_temporally_follows_points_from_later_to_earlier(repo, neo4j_url):
    tenant_id = uuid.uuid4()
    earlier = _episode()
    later = _episode()
    await repo.upsert_episode_node(earlier, tenant_id)
    await repo.upsert_episode_node(later, tenant_id)

    await repo.link_temporally_follows(earlier.id, later.id, tenant_id)

    count = await _raw_query(
        neo4j_url,
        "MATCH (later:Episode {id: $later_id})-[:TEMPORALLY_FOLLOWS]->"
        "(earlier:Episode {id: $earlier_id}) RETURN count(*) AS c",
        later_id=str(later.id),
        earlier_id=str(earlier.id),
    )
    assert count == 1


async def test_link_mentions_creates_entity_and_edge(repo, neo4j_url):
    tenant_id = uuid.uuid4()
    episode = _episode()
    await repo.upsert_episode_node(episode, tenant_id)

    await repo.link_mentions(episode.id, "Paris", tenant_id)

    count = await _raw_query(
        neo4j_url,
        "MATCH (:Episode {id: $episode_id})-[:MENTIONS]->(ent:Entity {name: $name}) "
        "RETURN count(*) AS c",
        episode_id=str(episode.id),
        name="Paris",
    )
    assert count == 1


async def test_link_mentions_the_same_entity_twice_does_not_duplicate_the_node(repo, neo4j_url):
    tenant_id = uuid.uuid4()
    episode_a = _episode()
    episode_b = _episode()
    await repo.upsert_episode_node(episode_a, tenant_id)
    await repo.upsert_episode_node(episode_b, tenant_id)

    await repo.link_mentions(episode_a.id, "Paris", tenant_id)
    await repo.link_mentions(episode_b.id, "Paris", tenant_id)

    count = await _raw_query(
        neo4j_url,
        "MATCH (ent:Entity {name: $name, tenant_id: $tenant_id}) RETURN count(ent) AS c",
        name="Paris",
        tenant_id=str(tenant_id),
    )
    assert count == 1


async def test_link_abstracts_to_creates_the_edge(repo, neo4j_url):
    tenant_id = uuid.uuid4()
    episode = _episode()
    fact = _fact()
    await repo.upsert_episode_node(episode, tenant_id)
    await repo.upsert_fact_node(fact, tenant_id)

    await repo.link_abstracts_to(episode.id, fact.id, tenant_id)

    count = await _raw_query(
        neo4j_url,
        "MATCH (:Episode {id: $episode_id})-[:ABSTRACTS_TO]->(:Fact {id: $fact_id}) "
        "RETURN count(*) AS c",
        episode_id=str(episode.id),
        fact_id=str(fact.id),
    )
    assert count == 1


async def test_nodes_never_leak_across_tenants(repo, neo4j_url):
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    episode = _episode()
    await repo.upsert_episode_node(episode, tenant_a)

    count = await _raw_query(
        neo4j_url,
        "MATCH (e:Episode {id: $id, tenant_id: $tenant_id}) RETURN count(e) AS c",
        id=str(episode.id),
        tenant_id=str(tenant_b),
    )
    assert count == 0


async def test_spread_activation_reaches_a_fact_two_hops_from_a_mentioned_entity(repo):
    # Reproduces MAG.md's own worked example structurally: a query vector
    # search for the literal word "restaurant" would miss this fact
    # entirely, but graph traversal from the mentioned entity reaches it.
    tenant_id = uuid.uuid4()
    episode = _episode({"input": "I asked about vegan food in Paris"})
    fact = SemanticMemory(
        id=uuid.uuid4(), user_id=uuid.uuid4(),
        fact_key="restaurant_recommendation", fact_value="Le Potager du Marais",
        embedding=[0.1] * 384,
    )
    await repo.upsert_episode_node(episode, tenant_id)
    await repo.upsert_fact_node(fact, tenant_id)
    await repo.link_mentions(episode.id, "Paris", tenant_id)
    await repo.link_abstracts_to(episode.id, fact.id, tenant_id)

    result = await repo.spread_activation(
        tenant_id=tenant_id,
        start_entity_names=["Paris"],
        max_hops=3,
        decay_factor=0.5,
        activation_threshold=0.05,
    )

    by_id = {node.node_id: node for node in result}
    assert "Paris" in by_id
    assert by_id["Paris"].hops == 0
    assert by_id["Paris"].activation == pytest.approx(1.0)
    assert str(episode.id) in by_id
    assert by_id[str(episode.id)].hops == 1
    assert by_id[str(episode.id)].activation == pytest.approx(0.5)
    assert str(fact.id) in by_id
    assert by_id[str(fact.id)].hops == 2
    assert by_id[str(fact.id)].activation == pytest.approx(0.25)
    assert by_id[str(fact.id)].properties["fact_value"] == "Le Potager du Marais"


async def test_spread_activation_excludes_nodes_below_the_threshold(repo):
    tenant_id = uuid.uuid4()
    episode = _episode()
    await repo.upsert_episode_node(episode, tenant_id)
    await repo.link_mentions(episode.id, "Paris", tenant_id)

    # decay_factor=0.5, 1 hop -> activation 0.5; threshold above that
    # excludes the episode, leaving only the start entity itself.
    result = await repo.spread_activation(
        tenant_id=tenant_id,
        start_entity_names=["Paris"],
        max_hops=3,
        decay_factor=0.5,
        activation_threshold=0.6,
    )

    by_id = {node.node_id: node for node in result}
    assert "Paris" in by_id
    assert str(episode.id) not in by_id


async def test_spread_activation_takes_the_max_activation_across_multiple_paths(repo):
    # The fact is reachable both directly (1 hop from the entity via a
    # (hypothetical) mention -- simulated here via two distinct 2-hop
    # episodes) and via a longer path; the node's activation must reflect
    # the SHORTEST path (max activation), not be summed across both.
    tenant_id = uuid.uuid4()
    close_episode = _episode()
    far_episode = _episode()
    fact = _fact()
    await repo.upsert_episode_node(close_episode, tenant_id)
    await repo.upsert_episode_node(far_episode, tenant_id)
    await repo.upsert_fact_node(fact, tenant_id)
    await repo.link_mentions(close_episode.id, "Bob", tenant_id)
    await repo.link_mentions(far_episode.id, "Bob", tenant_id)
    await repo.link_abstracts_to(close_episode.id, fact.id, tenant_id)
    # far_episode reaches the fact via one extra hop through close_episode.
    await repo.link_temporally_follows(close_episode.id, far_episode.id, tenant_id)

    result = await repo.spread_activation(
        tenant_id=tenant_id,
        start_entity_names=["Bob"],
        max_hops=4,
        decay_factor=0.5,
        activation_threshold=0.01,
    )

    by_id = {node.node_id: node for node in result}
    # Reachable in 2 hops (Bob -> close_episode -> fact), so activation
    # should be 0.25, not diluted by the longer 3-hop path through
    # far_episode, and not appearing twice in the result.
    assert sum(1 for n in result if n.node_id == str(fact.id)) == 1
    assert by_id[str(fact.id)].hops == 2
    assert by_id[str(fact.id)].activation == pytest.approx(0.25)


async def test_spread_activation_never_reaches_across_tenants(repo):
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    episode_a = _episode()
    episode_b = _episode()
    await repo.upsert_episode_node(episode_a, tenant_a)
    await repo.upsert_episode_node(episode_b, tenant_b)
    await repo.link_mentions(episode_a.id, "SharedName", tenant_a)
    await repo.link_mentions(episode_b.id, "SharedName", tenant_b)

    result = await repo.spread_activation(
        tenant_id=tenant_a,
        start_entity_names=["SharedName"],
        max_hops=3,
        decay_factor=0.5,
        activation_threshold=0.05,
    )

    ids = {node.node_id for node in result}
    assert str(episode_a.id) in ids
    assert str(episode_b.id) not in ids


async def test_upsert_fact_node_never_leaks_across_tenants(repo, neo4j_url):
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    fact = _fact()
    await repo.upsert_fact_node(fact, tenant_a)

    # Positive case first -- a review caught that without this, a version
    # of upsert_fact_node that dropped tenant_id from its MERGE key
    # entirely (leaving the node with no tenant_id property at all) would
    # ALSO return 0 for the tenant_b query below, passing this test
    # vacuously: "not found under the wrong tenant" proves nothing if it's
    # not found under the RIGHT tenant either.
    found_under_correct_tenant = await _raw_query(
        neo4j_url,
        "MATCH (f:Fact {id: $id, tenant_id: $tenant_id}) RETURN count(f) AS c",
        id=str(fact.id),
        tenant_id=str(tenant_a),
    )
    assert found_under_correct_tenant == 1

    count = await _raw_query(
        neo4j_url,
        "MATCH (f:Fact {id: $id, tenant_id: $tenant_id}) RETURN count(f) AS c",
        id=str(fact.id),
        tenant_id=str(tenant_b),
    )
    assert count == 0


async def test_link_participated_in_never_leaks_across_tenants(repo, neo4j_url):
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await repo.link_participated_in(user_id, session_id, tenant_a)

    # Positive case first -- same reasoning as
    # test_upsert_fact_node_never_leaks_across_tenants: without this, a
    # version that dropped tenant_id from the User MERGE key entirely
    # would also return 0 under tenant_b, passing vacuously.
    found_under_correct_tenant = await _raw_query(
        neo4j_url,
        "MATCH (u:User {id: $user_id, tenant_id: $tenant_id}) RETURN count(u) AS c",
        user_id=str(user_id),
        tenant_id=str(tenant_a),
    )
    assert found_under_correct_tenant == 1

    count = await _raw_query(
        neo4j_url,
        "MATCH (u:User {id: $user_id, tenant_id: $tenant_id}) RETURN count(u) AS c",
        user_id=str(user_id),
        tenant_id=str(tenant_b),
    )
    assert count == 0


async def test_link_temporally_follows_never_links_across_tenants(repo, neo4j_url):
    # Both episodes exist, but under different tenants -- both MATCH
    # clauses are tenant-scoped, so this must be a clean no-op, not a
    # cross-tenant edge.
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    earlier = _episode()
    later = _episode()
    await repo.upsert_episode_node(earlier, tenant_a)
    await repo.upsert_episode_node(later, tenant_b)

    await repo.link_temporally_follows(earlier.id, later.id, tenant_a)

    count = await _raw_query(
        neo4j_url,
        "MATCH (:Episode {id: $later_id})-[:TEMPORALLY_FOLLOWS]->(:Episode {id: $earlier_id}) "
        "RETURN count(*) AS c",
        later_id=str(later.id),
        earlier_id=str(earlier.id),
    )
    assert count == 0


async def test_link_abstracts_to_never_links_across_tenants(repo, neo4j_url):
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    episode = _episode()
    fact = _fact()
    await repo.upsert_episode_node(episode, tenant_a)
    await repo.upsert_fact_node(fact, tenant_b)

    await repo.link_abstracts_to(episode.id, fact.id, tenant_a)

    count = await _raw_query(
        neo4j_url,
        "MATCH (:Episode {id: $episode_id})-[:ABSTRACTS_TO]->(:Fact {id: $fact_id}) "
        "RETURN count(*) AS c",
        episode_id=str(episode.id),
        fact_id=str(fact.id),
    )
    assert count == 0


async def test_link_mentions_does_not_create_a_dangling_entity_when_the_episode_is_missing(
    repo, neo4j_url
):
    # Regression test: an earlier version MERGEd the Entity node BEFORE
    # matching the Episode, so a call for an episode that was never (or not
    # yet) upserted still created a persisted, edge-less Entity node -- a
    # review caught this. The episode is deliberately never saved here.
    tenant_id = uuid.uuid4()
    never_saved_episode_id = uuid.uuid4()

    await repo.link_mentions(never_saved_episode_id, "GhostEntity", tenant_id)

    count = await _raw_query(
        neo4j_url,
        "MATCH (ent:Entity {name: $name, tenant_id: $tenant_id}) RETURN count(ent) AS c",
        name="GhostEntity",
        tenant_id=str(tenant_id),
    )
    assert count == 0


async def test_link_mentions_never_leaks_the_entity_node_across_tenants(repo, neo4j_url):
    # A scoped re-review caught that the four cross-tenant tests added
    # alongside this one's sibling (dangling-entity) fix missed a sixth
    # write method: link_mentions also MERGEs a tenant-scoped node
    # (Entity), the same class of risk upsert_fact_node/
    # link_participated_in/link_temporally_follows/link_abstracts_to were
    # given tests for, and nothing here had covered it.
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    episode = _episode()
    await repo.upsert_episode_node(episode, tenant_a)

    await repo.link_mentions(episode.id, "SameNameAcrossTenants", tenant_a)

    found_under_correct_tenant = await _raw_query(
        neo4j_url,
        "MATCH (:Episode {id: $episode_id})-[:MENTIONS]->"
        "(ent:Entity {name: $name, tenant_id: $tenant_id}) RETURN count(*) AS c",
        episode_id=str(episode.id),
        name="SameNameAcrossTenants",
        tenant_id=str(tenant_a),
    )
    assert found_under_correct_tenant == 1

    count = await _raw_query(
        neo4j_url,
        "MATCH (ent:Entity {name: $name, tenant_id: $tenant_id}) RETURN count(ent) AS c",
        name="SameNameAcrossTenants",
        tenant_id=str(tenant_b),
    )
    assert count == 0


async def test_spread_activation_rejects_max_hops_out_of_range(repo):
    with pytest.raises(ValueError, match="max_hops"):
        await repo.spread_activation(
            tenant_id=uuid.uuid4(),
            start_entity_names=["x"],
            max_hops=0,
            decay_factor=0.5,
            activation_threshold=0.05,
        )
    with pytest.raises(ValueError, match="max_hops"):
        await repo.spread_activation(
            tenant_id=uuid.uuid4(),
            start_entity_names=["x"],
            max_hops=11,
            decay_factor=0.5,
            activation_threshold=0.05,
        )


async def test_spread_activation_accepts_max_hops_at_the_upper_boundary(repo):
    # A scoped re-review caught that only invalid values (0, 11) were ever
    # tested -- max_hops=10, the documented ceiling itself, was never
    # exercised as a value that should NOT raise. Without this, an
    # off-by-one in the validation (e.g. "< _MAX_HOPS" instead of
    # "<= _MAX_HOPS", silently shrinking the real valid range to [1, 9])
    # would pass every other test in this file while incorrectly rejecting
    # every real caller who requests the documented maximum.
    result = await repo.spread_activation(
        tenant_id=uuid.uuid4(),
        start_entity_names=["x"],
        max_hops=10,
        decay_factor=0.5,
        activation_threshold=0.05,
    )
    assert result == []  # no matching entity -- just confirms no ValueError


async def test_spread_activation_rejects_decay_factor_out_of_range(repo):
    for bad_decay in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="decay_factor"):
            await repo.spread_activation(
                tenant_id=uuid.uuid4(),
                start_entity_names=["x"],
                max_hops=3,
                decay_factor=bad_decay,
                activation_threshold=0.05,
            )


async def test_spread_activation_never_excludes_a_start_entity_regardless_of_threshold(repo):
    tenant_id = uuid.uuid4()
    episode = _episode()
    await repo.upsert_episode_node(episode, tenant_id)
    await repo.link_mentions(episode.id, "Paris", tenant_id)

    # A threshold of 1.0 would exclude every non-start node (their
    # activation is always < 1.0 for decay_factor < 1.0), but the start
    # entity itself (activation exactly 1.0) must still come back.
    result = await repo.spread_activation(
        tenant_id=tenant_id,
        start_entity_names=["Paris"],
        max_hops=1,
        decay_factor=0.5,
        activation_threshold=1.0,
    )

    by_id = {node.node_id: node for node in result}
    assert "Paris" in by_id
    assert by_id["Paris"].activation == pytest.approx(1.0)
