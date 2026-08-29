from src.mag.application.gating.task_specific_filtering import TaskSpecificFiltering
from src.mag.domain.entities import GatingCandidate


def _candidate(source_type: str, content_text: str = "x") -> GatingCandidate:
    return GatingCandidate(
        content_text=content_text,
        score=0.5,
        salience=0.0,
        timestamp=None,
        source_type=source_type,
        origin=None,  # type: ignore[arg-type]
        embedding=[],
    )


async def test_filters_down_to_a_single_allowed_source_type():
    # Distinguished by content_text, not just source_type -- two
    # field-identical "fact" candidates would make dataclass equality
    # accept ANY relative order as "correct" (fact_a == fact_b already
    # holds), proving nothing about whether filtering preserved order.
    fact_a = _candidate("fact", content_text="fact_a")
    episode = _candidate("episode", content_text="episode")
    fact_b = _candidate("fact", content_text="fact_b")
    graph_node = _candidate("graph_node", content_text="graph_node")
    candidates = [fact_a, episode, fact_b, graph_node]

    result = await TaskSpecificFiltering().execute(candidates, {"fact"})

    assert [c.content_text for c in result] == ["fact_a", "fact_b"]


async def test_filters_to_two_allowed_source_types_excludes_the_third():
    episode = _candidate("episode")
    fact = _candidate("fact")
    graph_node = _candidate("graph_node")
    candidates = [episode, fact, graph_node]

    result = await TaskSpecificFiltering().execute(candidates, {"episode", "graph_node"})

    assert result == [episode, graph_node]


async def test_preserves_original_relative_order_of_survivors():
    # Same field-identity trap as above -- distinguish the two episodes by
    # content_text so the assertion actually proves order, not just count.
    first_episode = _candidate("episode", content_text="first_episode")
    fact = _candidate("fact", content_text="fact")
    second_episode = _candidate("episode", content_text="second_episode")
    graph_node = _candidate("graph_node", content_text="graph_node")
    candidates = [first_episode, fact, second_episode, graph_node]

    result = await TaskSpecificFiltering().execute(candidates, {"episode"})

    assert [c.content_text for c in result] == ["first_episode", "second_episode"]


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
