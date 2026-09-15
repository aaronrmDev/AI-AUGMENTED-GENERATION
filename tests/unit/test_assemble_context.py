from src.orchestration.application.assemble_context import assemble_context
from src.orchestration.domain.entities import BudgetAllocation, ContextItem, Paradigm
from src.shared.tokenization import count_tokens

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG
_LARGE = "alpha " * 60


def _budget(cag=1_000, mag=1_000, rag=1_000):
    return BudgetAllocation(cag=cag, mag=mag, rag=rag, query=0, reserve=0)


def test_sections_render_cheapest_first_with_paradigm_labels_regardless_of_input_order():
    items = [
        ContextItem(RAG, "retrieved text", 0.9),
        ContextItem(CAG, "cached text", 0.8),
        ContextItem(MAG, "remembered text", 0.7),
    ]
    text = assemble_context(items, _budget()).text
    assert text.index("(CAG)") < text.index("cached text") < text.index("(MAG)")
    assert text.index("remembered text") < text.index("(RAG)") < text.index("retrieved text")


def test_items_pack_highest_score_first_and_an_oversized_item_does_not_block_a_smaller_one():
    big = ContextItem(RAG, _LARGE, 0.9)
    small = ContextItem(RAG, "small fact", 0.5)
    assembled = assemble_context([small, big], _budget(rag=10))
    assert assembled.included == [small]
    assert assembled.dropped[RAG] == 1
    assert assembled.tokens_used[RAG] == count_tokens("small fact")


def test_higher_scoring_items_win_the_budget_when_both_cannot_fit():
    first = ContextItem(MAG, "one two three", 0.9)
    second = ContextItem(MAG, "four five six", 0.4)
    budget = count_tokens("one two three")
    assembled = assemble_context([second, first], _budget(mag=budget))
    assert assembled.included == [first]
    assert assembled.dropped[MAG] == 1


def test_a_paradigm_with_a_zero_slice_drops_everything_it_returned():
    assembled = assemble_context([ContextItem(CAG, "cached", 0.9)], _budget(cag=0))
    assert assembled.included == []
    assert assembled.dropped[CAG] == 1
    assert "(CAG)" not in assembled.text


def test_no_items_assemble_to_an_empty_context():
    assembled = assemble_context([], _budget())
    assert assembled.text == ""
    assert assembled.dropped == {CAG: 0, MAG: 0, RAG: 0}
