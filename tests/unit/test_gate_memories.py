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

    total_tokens = sum(count_tokens(c.content_text) for c in result)
    assert total_tokens <= 50
    # The budget invariant alone is satisfied trivially by an empty result
    # -- both candidates are ~500+ tokens each, individually far too big
    # for a 50-token budget, so TokenBudgetAllocation must skip both and
    # this must come back empty. An assertion that only checks "<= 50"
    # would pass identically whether the pipeline actually ran the walk or
    # short-circuited to [] for the wrong reason.
    assert result == []


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

    # now=now pins the pipeline's own clock to the same instant the
    # episode's age is computed against below, so decay is an exact,
    # reproducible value instead of whatever a few microseconds of real
    # wall-clock drift between fixture construction and pipeline execution
    # happen to produce.
    result = await GateMemories().execute(
        episodes=[episode], facts=[], graph_nodes=[], token_budget=10_000, now=now,
    )

    # No query_embedding -> DynamicReranking never runs, so the only thing
    # that could have changed the original 0.5 score is recency decay --
    # age is exactly 0 (now == episode.timestamp), so decay is exactly
    # 1.0. The result must equal 0.5 exactly, not merely be "close to it",
    # or a cosine-similarity value slipping in unnoticed near 0.5 would
    # pass just as easily as the correct behavior.
    assert result[0].score == 0.5


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
