import json
from datetime import datetime
from typing import Any

from src.mag.domain.entities import ActivatedNode, GatingCandidate, ScoredEpisode, ScoredFact


def from_scored_episode(scored: ScoredEpisode) -> GatingCandidate:
    episode = scored.episode
    return GatingCandidate(
        # sort_keys=True matches CaptureEpisode's own embedding-input
        # convention for the same field -- a stable text form regardless
        # of dict insertion order.
        content_text=json.dumps(episode.content, sort_keys=True),
        score=scored.score,
        salience=episode.salience_score,
        timestamp=episode.timestamp,
        source_type="episode",
        origin=episode,
        embedding=episode.embedding,
    )


def from_scored_fact(scored: ScoredFact) -> GatingCandidate:
    fact = scored.fact
    return GatingCandidate(
        content_text=fact.fact_value,
        # confidence is the closest analogous secondary signal a
        # SemanticMemory carries -- a deliberate substitution for
        # salience_score, which facts have no equivalent of, not an
        # oversight (see the design spec's candidate-type section).
        salience=fact.confidence,
        score=scored.score,
        timestamp=None,
        source_type="fact",
        origin=fact,
        embedding=fact.embedding,
    )


def from_activated_node(node: ActivatedNode) -> GatingCandidate:
    timestamp = _extract_timestamp(node.properties)
    return GatingCandidate(
        # default=str, not the bare json.dumps other adapters use above:
        # an Episode-typed node's properties carry a real `datetime`
        # object (Neo4jMemoryGraphRepository._node_properties
        # deserializes it from its stored ISO string), which plain
        # json.dumps can't serialize -- str() gives a stable, readable
        # text form for token-counting purposes without needing this
        # function to special-case every possible non-JSON-native
        # property value a future node type might carry.
        content_text=json.dumps(node.properties, sort_keys=True, default=str),
        score=node.activation,
        salience=0.0,
        timestamp=timestamp,
        source_type="graph_node",
        origin=node,
        # No node type this batch produces via spread_activation carries a
        # real embedding in its properties dict (Neo4jMemoryGraphRepository
        # never writes one into any node's properties -- Entity.embedding
        # is documented as unpopulated, see Batch D's design spec).
        embedding=[],
    )


def _extract_timestamp(properties: dict[str, Any]) -> datetime | None:
    # Opportunistic, not required: an Episode-typed ActivatedNode's
    # properties carry a real datetime already deserialized by
    # Neo4jMemoryGraphRepository._node_properties (isoformat string ->
    # datetime, same as this project's other stores); every other node
    # type has no timestamp property at all.
    value = properties.get("timestamp")
    return value if isinstance(value, datetime) else None
