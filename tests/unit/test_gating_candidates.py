import json
import uuid
from datetime import UTC, datetime

from src.mag.application.gating._candidates import (
    from_activated_node,
    from_scored_episode,
    from_scored_fact,
)
from src.mag.domain.entities import (
    ActivatedNode,
    EpisodicMemory,
    ScoredEpisode,
    ScoredFact,
    SemanticMemory,
)


def test_from_scored_episode_flattens_the_expected_fields():
    timestamp = datetime.now(UTC)
    episode = EpisodicMemory(
        id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        content={"b": "second", "a": "first"},
        embedding=[0.1] * 384,
        timestamp=timestamp,
        salience_score=0.7,
    )
    scored = ScoredEpisode(episode=episode, score=0.9)

    candidate = from_scored_episode(scored)

    assert candidate.content_text == json.dumps(episode.content, sort_keys=True)
    assert candidate.score == 0.9
    assert candidate.salience == 0.7
    assert candidate.timestamp == timestamp
    assert candidate.source_type == "episode"
    assert candidate.origin is episode
    assert candidate.embedding == episode.embedding


def test_from_scored_fact_uses_confidence_as_salience_and_has_no_timestamp():
    fact = SemanticMemory(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        fact_key="favorite_color",
        fact_value="blue",
        embedding=[0.2] * 384,
        confidence=0.6,
    )
    scored = ScoredFact(fact=fact, score=0.8)

    candidate = from_scored_fact(scored)

    assert candidate.content_text == "blue"
    assert candidate.score == 0.8
    assert candidate.salience == 0.6
    assert candidate.timestamp is None
    assert candidate.source_type == "fact"
    assert candidate.origin is fact
    assert candidate.embedding == fact.embedding


def test_from_activated_node_has_zero_salience_and_no_embedding():
    node = ActivatedNode(
        node_id="Paris",
        node_type="Entity",
        properties={"name": "Paris"},
        activation=0.5,
        hops=1,
    )

    candidate = from_activated_node(node)

    assert candidate.content_text == json.dumps(node.properties, sort_keys=True)
    assert candidate.score == 0.5
    assert candidate.salience == 0.0
    assert candidate.timestamp is None
    assert candidate.source_type == "graph_node"
    assert candidate.origin is node
    assert candidate.embedding == []


def test_from_activated_node_opportunistically_extracts_a_real_timestamp():
    timestamp = datetime.now(UTC)
    node = ActivatedNode(
        node_id=str(uuid.uuid4()),
        node_type="Episode",
        properties={"content": {"input": "hi"}, "timestamp": timestamp},
        activation=0.25,
        hops=2,
    )

    candidate = from_activated_node(node)

    assert candidate.timestamp == timestamp


def test_from_activated_node_ignores_a_non_datetime_timestamp_value():
    node = ActivatedNode(
        node_id="x",
        node_type="Entity",
        properties={"timestamp": "not-a-datetime-instance"},
        activation=0.5,
        hops=1,
    )

    candidate = from_activated_node(node)

    assert candidate.timestamp is None
