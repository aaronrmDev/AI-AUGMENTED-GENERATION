from src.mag.application.gating.hierarchical_assembly import HierarchicalAssembly
from src.mag.domain.entities import GatingCandidate


def _candidate(source_type: str, score: float) -> GatingCandidate:
    return GatingCandidate(
        content_text="x",
        score=score,
        salience=0.0,
        timestamp=None,
        source_type=source_type,
        origin=None,  # type: ignore[arg-type]
        embedding=[],
    )


async def test_facts_sort_before_episodes_before_graph_nodes_regardless_of_score():
    node = _candidate("graph_node", 0.99)
    episode = _candidate("episode", 0.5)
    fact = _candidate("fact", 0.01)

    result = await HierarchicalAssembly().execute([node, episode, fact])

    assert result == [fact, episode, node]


async def test_within_the_same_source_type_higher_score_sorts_first():
    low = _candidate("fact", 0.2)
    high = _candidate("fact", 0.8)
    mid = _candidate("fact", 0.5)

    result = await HierarchicalAssembly().execute([low, high, mid])

    assert result == [high, mid, low]


async def test_output_has_the_same_length_and_set_as_the_input():
    candidates = [
        _candidate("graph_node", 0.1),
        _candidate("fact", 0.9),
        _candidate("episode", 0.3),
        _candidate("fact", 0.4),
    ]

    result = await HierarchicalAssembly().execute(candidates)

    assert len(result) == len(candidates)
    assert {id(c) for c in result} == {id(c) for c in candidates}


async def test_empty_list_in_empty_list_out():
    result = await HierarchicalAssembly().execute([])

    assert result == []


async def test_a_tie_in_source_type_and_score_preserves_original_relative_order():
    first = _candidate("episode", 0.5)
    second = _candidate("episode", 0.5)
    third = _candidate("episode", 0.5)

    result = await HierarchicalAssembly().execute([first, second, third])

    assert [id(c) for c in result] == [id(first), id(second), id(third)]


async def test_unrecognized_source_type_sorts_after_graph_node_without_crashing():
    fact = _candidate("fact", 0.1)
    episode = _candidate("episode", 0.1)
    node = _candidate("graph_node", 0.1)
    unknown = _candidate("unknown_type", 0.99)

    result = await HierarchicalAssembly().execute([unknown, node, episode, fact])

    assert result == [fact, episode, node, unknown]
