import uuid
from datetime import UTC, datetime

from src.mag.application.gating.gate_memories import GateMemories
from src.mag.domain.entities import (
    ActivatedNode,
    EpisodicMemory,
    ScoredEpisode,
    ScoredFact,
    SemanticMemory,
)


def _episode(score: float, timestamp: datetime, content: dict | None = None) -> ScoredEpisode:
    episode = EpisodicMemory(
        id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        content=content or {"input": "hi"},
        embedding=[0.1] * 384,
        timestamp=timestamp,
    )
    return ScoredEpisode(episode=episode, score=score)


def _fact(score: float, fact_value: str = "blue") -> ScoredFact:
    fact = SemanticMemory(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        fact_key="favorite_color",
        fact_value=fact_value,
        embedding=[0.2] * 384,
    )
    return ScoredFact(fact=fact, score=score)


def _node(activation: float) -> ActivatedNode:
    return ActivatedNode(
        node_id="Paris", node_type="Entity", properties={"name": "Paris"},
        activation=activation, hops=1,
    )


async def test_execute_combines_all_three_source_types_into_one_pool():
    now = datetime.now(UTC)
    result = await GateMemories().execute(
        episodes=[_episode(0.5, now)],
        facts=[_fact(0.5)],
        graph_nodes=[_node(0.5)],
        token_budget=10_000,
    )

    source_types = {c.source_type for c in result}
    assert source_types == {"episode", "fact", "graph_node"}


async def test_execute_respects_the_token_budget():
    now = datetime.now(UTC)
    episodes = [_episode(0.9, now, content={"input": "x" * 2000})]
    facts = [_fact(0.8, fact_value="y" * 2000)]

    result = await GateMemories().execute(
        episodes=episodes, facts=facts, graph_nodes=[], token_budget=50,
    )

    from src.shared.tokenization import count_tokens

    assert sum(count_tokens(c.content_text) for c in result) <= 50


async def test_execute_orders_facts_before_episodes_via_hierarchical_assembly():
    now = datetime.now(UTC)
    # Episode scores higher than the fact, but hierarchical assembly must
    # still place the fact first -- the whole point of this pipeline stage
    # running last.
    result = await GateMemories().execute(
        episodes=[_episode(0.99, now)],
        facts=[_fact(0.01)],
        graph_nodes=[],
        token_budget=10_000,
    )

    assert [c.source_type for c in result] == ["fact", "episode"]


async def test_execute_skips_dynamic_reranking_when_no_query_embedding_is_given():
    now = datetime.now(UTC)
    episode = _episode(0.5, now)

    result = await GateMemories().execute(
        episodes=[episode], facts=[], graph_nodes=[], token_budget=10_000,
    )

    # No query_embedding -> DynamicReranking never runs, so the only thing
    # that could have changed the original 0.5 score is recency decay
    # (age ~0 here, so decay ~1.0) -- score should still be ~0.5, not
    # replaced by a cosine-similarity value against nothing.
    assert result[0].score > 0.49


async def test_execute_applies_dynamic_reranking_when_a_query_embedding_is_given():
    now = datetime.now(UTC)
    unit_vector = [1.0] + [0.0] * 383
    episode = EpisodicMemory(
        id=uuid.uuid4(), session_id=uuid.uuid4(), content={"input": "hi"},
        embedding=unit_vector, timestamp=now,
    )
    matching = ScoredEpisode(episode=episode, score=0.1)  # low original score

    result = await GateMemories().execute(
        episodes=[matching], facts=[], graph_nodes=[], token_budget=10_000,
        query_embedding=unit_vector,
    )

    # Re-ranked against an identical query embedding -> similarity ~1.0,
    # overriding the deliberately low original score of 0.1.
    assert result[0].score > 0.9


async def test_execute_skips_task_specific_filtering_when_no_allowed_types_given():
    now = datetime.now(UTC)
    result = await GateMemories().execute(
        episodes=[_episode(0.5, now)], facts=[_fact(0.5)], graph_nodes=[_node(0.5)],
        token_budget=10_000,
    )

    assert {c.source_type for c in result} == {"episode", "fact", "graph_node"}


async def test_execute_applies_task_specific_filtering_when_allowed_types_given():
    now = datetime.now(UTC)
    result = await GateMemories().execute(
        episodes=[_episode(0.5, now)], facts=[_fact(0.5)], graph_nodes=[_node(0.5)],
        token_budget=10_000,
        allowed_source_types={"fact"},
    )

    assert {c.source_type for c in result} == {"fact"}


async def test_execute_returns_an_empty_list_for_empty_input():
    result = await GateMemories().execute(
        episodes=[], facts=[], graph_nodes=[], token_budget=10_000,
    )

    assert result == []
