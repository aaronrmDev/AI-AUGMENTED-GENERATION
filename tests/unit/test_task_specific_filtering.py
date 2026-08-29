from src.mag.application.gating.task_specific_filtering import TaskSpecificFiltering
from src.mag.domain.entities import GatingCandidate


def _candidate(source_type: str) -> GatingCandidate:
    return GatingCandidate(
        content_text="x",
        score=0.5,
        salience=0.0,
        timestamp=None,
        source_type=source_type,
        origin=None,  # type: ignore[arg-type]
        embedding=[],
    )


async def test_filters_down_to_a_single_allowed_source_type():
    fact_a = _candidate("fact")
    episode = _candidate("episode")
    fact_b = _candidate("fact")
    graph_node = _candidate("graph_node")
    candidates = [fact_a, episode, fact_b, graph_node]

    result = await TaskSpecificFiltering().execute(candidates, {"fact"})

    assert result == [fact_a, fact_b]


async def test_filters_to_two_allowed_source_types_excludes_the_third():
    episode = _candidate("episode")
    fact = _candidate("fact")
    graph_node = _candidate("graph_node")
    candidates = [episode, fact, graph_node]

    result = await TaskSpecificFiltering().execute(candidates, {"episode", "graph_node"})

    assert result == [episode, graph_node]


async def test_preserves_original_relative_order_of_survivors():
    first_episode = _candidate("episode")
    fact = _candidate("fact")
    second_episode = _candidate("episode")
    graph_node = _candidate("graph_node")
    candidates = [first_episode, fact, second_episode, graph_node]

    result = await TaskSpecificFiltering().execute(candidates, {"episode"})

    assert result == [first_episode, second_episode]


async def test_empty_allowed_source_types_returns_empty_list():
    candidates = [_candidate("episode"), _candidate("fact"), _candidate("graph_node")]

    result = await TaskSpecificFiltering().execute(candidates, set())

    assert result == []


async def test_all_known_source_types_allowed_returns_every_candidate_unchanged():
    episode = _candidate("episode")
    fact = _candidate("fact")
    graph_node = _candidate("graph_node")
    candidates = [episode, fact, graph_node]

    result = await TaskSpecificFiltering().execute(
        candidates, {"episode", "fact", "graph_node"}
    )

    assert result == candidates


async def test_empty_candidate_list_returns_empty_list():
    result = await TaskSpecificFiltering().execute([], {"episode", "fact"})

    assert result == []
