import json
import uuid
from datetime import datetime
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from src.mag.domain.entities import ActivatedNode, EpisodicMemory, SemanticMemory
from src.mag.domain.ports import MemoryGraphRepository

# The six node types and five edge types this repository writes/traverses --
# see docs/database/DATABASE.md for the schema this implements, and the
# Batch D design spec for why per-node properties beyond what's documented
# there were chosen the way they were.
_NODE_LABELS = ("User", "Session", "Entity", "Concept", "Episode", "Fact")

# A hard ceiling on spread_activation's max_hops -- not just "must be
# positive": an unbounded value lets a caller request a traversal across
# the whole graph regardless of decay, which is expensive and defeats the
# point of a distance-decayed relevance signal. 10 is generous relative to
# the design spec's own default of 3 and MAG.md's worked examples, which
# never traverse more than a handful of hops.
_MAX_HOPS = 10


class Neo4jMemoryGraphRepository(MemoryGraphRepository):
    def __init__(self, url: str, auth: tuple[str, str]) -> None:
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(url, auth=auth)

    async def close(self) -> None:
        await self._driver.close()

    async def ensure_schema(self) -> None:
        # Deliberately NOT part of the abstract port -- matches
        # QdrantEpisodicMemoryIndex.ensure_collection()'s established
        # convention (a store-provisioning concern, concrete-only). One
        # uniqueness constraint per id-identified node type (which also
        # creates a backing index, so MERGE-by-id below isn't a full scan),
        # plus the two Entity indexes DATABASE.md names explicitly.
        statements = [
            "CREATE CONSTRAINT episode_id IF NOT EXISTS "
            "FOR (e:Episode) REQUIRE (e.id, e.tenant_id) IS UNIQUE",
            "CREATE CONSTRAINT fact_id IF NOT EXISTS "
            "FOR (f:Fact) REQUIRE (f.id, f.tenant_id) IS UNIQUE",
            "CREATE CONSTRAINT user_id IF NOT EXISTS "
            "FOR (u:User) REQUIRE (u.id, u.tenant_id) IS UNIQUE",
            "CREATE CONSTRAINT session_id IF NOT EXISTS "
            "FOR (s:Session) REQUIRE (s.id, s.tenant_id) IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS "
            "FOR (n:Entity) REQUIRE (n.name, n.tenant_id) IS UNIQUE",
            "CREATE INDEX entity_embedding IF NOT EXISTS FOR (n:Entity) ON (n.embedding)",
        ]
        async with self._driver.session() as session:
            for statement in statements:
                await session.run(statement)

    async def upsert_episode_node(self, episode: EpisodicMemory, tenant_id: uuid.UUID) -> None:
        async with self._driver.session() as session:
            await session.run(
                "MERGE (e:Episode {id: $id, tenant_id: $tenant_id}) "
                "SET e.content = $content, e.timestamp = $timestamp",
                id=str(episode.id),
                tenant_id=str(tenant_id),
                # Neo4j node properties can't hold a nested map -- content is
                # arbitrary JSON-shaped data (see EpisodicMemory.content's own
                # docstring), so it's serialized here the same way
                # QdrantEpisodicMemoryIndex's payload already handles
                # datetime (isoformat string, not a native temporal type) --
                # matching an established convention, not inventing a new one.
                content=json.dumps(episode.content),
                timestamp=episode.timestamp.isoformat(),
            )

    async def upsert_fact_node(self, fact: SemanticMemory, tenant_id: uuid.UUID) -> None:
        async with self._driver.session() as session:
            await session.run(
                "MERGE (f:Fact {id: $id, tenant_id: $tenant_id}) "
                "SET f.fact_key = $fact_key, f.fact_value = $fact_value, "
                "    f.confidence = $confidence",
                id=str(fact.id),
                tenant_id=str(tenant_id),
                fact_key=fact.fact_key,
                fact_value=fact.fact_value,
                confidence=fact.confidence,
            )

    async def link_participated_in(
        self, user_id: uuid.UUID, session_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> None:
        async with self._driver.session() as db_session:
            await db_session.run(
                "MERGE (u:User {id: $user_id, tenant_id: $tenant_id}) "
                "MERGE (s:Session {id: $session_id, tenant_id: $tenant_id}) "
                "MERGE (u)-[:PARTICIPATED_IN]->(s)",
                user_id=str(user_id),
                session_id=str(session_id),
                tenant_id=str(tenant_id),
            )

    async def link_temporally_follows(
        self, earlier_episode_id: uuid.UUID, later_episode_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> None:
        # (later)-[:TEMPORALLY_FOLLOWS]->(earlier): "later follows earlier"
        # reads correctly in English, and matches DATABASE.md's own framing
        # of what this edge is for ("what happened after X" walks
        # backward from a later episode to find what it followed).
        async with self._driver.session() as session:
            await session.run(
                "MATCH (earlier:Episode {id: $earlier_id, tenant_id: $tenant_id}) "
                "MATCH (later:Episode {id: $later_id, tenant_id: $tenant_id}) "
                "MERGE (later)-[:TEMPORALLY_FOLLOWS]->(earlier)",
                earlier_id=str(earlier_episode_id),
                later_id=str(later_episode_id),
                tenant_id=str(tenant_id),
            )

    async def link_mentions(
        self, episode_id: uuid.UUID, entity_name: str, tenant_id: uuid.UUID
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                # Episode MATCHed first, Entity MERGEd second -- a review
                # caught the reverse order creating a dangling Entity node
                # with no edge when the episode didn't exist (e.g. its own
                # upsert_episode_node call failed earlier): MERGE with no
                # preceding MATCH runs and persists unconditionally, so
                # putting it before the Episode MATCH meant a missing
                # episode silently left an orphaned Entity behind instead
                # of this call being a clean no-op like its siblings
                # (link_temporally_follows, link_abstracts_to) already are.
                "MATCH (ep:Episode {id: $episode_id, tenant_id: $tenant_id}) "
                "MERGE (ent:Entity {name: $entity_name, tenant_id: $tenant_id}) "
                "MERGE (ep)-[:MENTIONS]->(ent)",
                entity_name=entity_name,
                episode_id=str(episode_id),
                tenant_id=str(tenant_id),
            )

    async def link_abstracts_to(
        self, episode_id: uuid.UUID, fact_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                "MATCH (ep:Episode {id: $episode_id, tenant_id: $tenant_id}) "
                "MATCH (f:Fact {id: $fact_id, tenant_id: $tenant_id}) "
                "MERGE (ep)-[:ABSTRACTS_TO]->(f)",
                episode_id=str(episode_id),
                fact_id=str(fact_id),
                tenant_id=str(tenant_id),
            )

    async def spread_activation(
        self,
        tenant_id: uuid.UUID,
        start_entity_names: list[str],
        max_hops: int,
        decay_factor: float,
        activation_threshold: float,
    ) -> list[ActivatedNode]:
        # Validated here, at the entry point, before any Cypher is built --
        # a review caught that neither this value nor decay_factor was
        # bounded anywhere in the call chain. max_hops<1 or >_MAX_HOPS
        # would either fail Cypher parsing (a negative bound) or let an
        # unbounded traversal run against the whole graph; decay_factor
        # outside (0, 1) either doesn't decay at all (1.0) or makes
        # activation GROW with distance instead of decaying (>1), inverting
        # "most activated" from nearest to farthest.
        if not (1 <= max_hops <= _MAX_HOPS):
            raise ValueError(f"max_hops must be between 1 and {_MAX_HOPS}, got {max_hops}")
        if not (0.0 < decay_factor < 1.0):
            raise ValueError(f"decay_factor must be in (0.0, 1.0), got {decay_factor}")
        # max_hops is interpolated directly into the query text, not bound
        # as a parameter: Cypher's variable-length relationship syntax
        # ([*1..N]) requires N to be a literal at parse time, not a runtime
        # parameter -- a real Cypher limitation, not an oversight. Safe here
        # specifically because max_hops is now validated as a bounded int
        # immediately above, and int() below is a defensive cast on an
        # already-validated value, not a trust boundary -- every other
        # value in this query (tenant_id, start_entity_names) is still a
        # real bound parameter.
        hops = int(max_hops)
        query = (
            "MATCH (start:Entity {tenant_id: $tenant_id}) "
            "WHERE start.name IN $start_entity_names "
            "RETURN start AS node, 0 AS hops "
            "UNION "
            "MATCH (start:Entity {tenant_id: $tenant_id}) "
            "WHERE start.name IN $start_entity_names "
            "WITH collect(DISTINCT start) AS starts "
            "UNWIND starts AS s "
            f"MATCH path = (s)-[*1..{hops}]-(reached) "
            "WHERE ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id) "
            "  AND NOT reached IN starts "
            "WITH reached, min(length(path)) AS hops "
            "RETURN reached AS node, hops AS hops"
        )
        async with self._driver.session() as session:
            result = await session.run(
                query,
                tenant_id=str(tenant_id),
                start_entity_names=start_entity_names,
            )
            records = [record async for record in result]

        activated: dict[str, ActivatedNode] = {}
        for record in records:
            node = record["node"]
            node_hops = record["hops"]
            activation = decay_factor**node_hops
            # A start entity (hops == 0, activation always exactly 1.0) is
            # exempt from the threshold -- it's the thing the caller
            # explicitly asked about, not a decayed-relevance discovery, so
            # a caller picking a high threshold (even >= 1.0, "only the
            # strongest matches") shouldn't silently drop the entity their
            # own query was anchored on. A review caught this excluding
            # start entities whenever activation_threshold >= 1.0.
            if node_hops > 0 and activation <= activation_threshold:
                continue
            node_id = _node_identifier(node)
            existing = activated.get(node_id)
            # A node can appear more than once (reached via >1 start entity,
            # or the same UNION branch producing it twice) -- keep the
            # highest activation seen, i.e. the fewest hops, matching
            # spread_activation's own MAX-not-sum contract.
            if existing is not None and existing.activation >= activation:
                continue
            activated[node_id] = ActivatedNode(
                node_id=node_id,
                node_type=_node_type(node),
                properties=_node_properties(node),
                activation=activation,
                hops=node_hops,
            )
        return list(activated.values())


def _node_identifier(node: Any) -> str:
    # Entity/Concept have no id property (name is their natural key); every
    # other node type here does. Falls back to Neo4j's own internal element
    # id only if neither is present, which shouldn't happen for any node
    # this repository itself ever writes.
    props = dict(node.items())
    if "id" in props:
        return str(props["id"])
    if "name" in props:
        return str(props["name"])
    return str(node.element_id)


def _node_type(node: Any) -> str:
    for label in node.labels:
        if label in _NODE_LABELS:
            return str(label)
    # Shouldn't happen for any node this repository writes, but a node from
    # outside this schema (or a future label this repository doesn't know
    # about yet) shouldn't crash a traversal result -- report it honestly
    # instead of silently mislabeling it as one of the six known types.
    return next(iter(node.labels), "Unknown")


def _node_properties(node: Any) -> dict[str, Any]:
    props = dict(node.items())
    if "content" in props and isinstance(props["content"], str):
        # Episode.content round-trips back to a dict here -- see
        # upsert_episode_node's docstring for why it's stored as a JSON
        # string in the first place. A caller inspecting
        # ActivatedNode.properties["content"] wants the dict this system
        # uses everywhere else, not an escaped JSON string.
        props["content"] = json.loads(props["content"])
    if "timestamp" in props and isinstance(props["timestamp"], str):
        props["timestamp"] = datetime.fromisoformat(props["timestamp"])
    return props
