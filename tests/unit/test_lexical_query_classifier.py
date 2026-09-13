import pytest

from src.orchestration.domain.entities import Paradigm
from src.orchestration.infrastructure.lexical_query_classifier import LexicalQueryClassifier

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG


async def test_a_query_with_no_cues_scores_zero_everywhere():
    scores = await LexicalQueryClassifier().score("what is two plus two", [])
    assert scores == {CAG: 0.0, MAG: 0.0, RAG: 0.0}


async def test_each_matching_cue_raises_its_paradigm_score():
    scores = await LexicalQueryClassifier().score("What changed in the policy today?", [])
    assert scores[RAG] == pytest.approx(0.91)  # "changed", "today"
    assert scores[CAG] == pytest.approx(0.7)  # "policy"
    assert scores[MAG] == 0.0


async def test_cues_match_whole_words_case_insensitively():
    classifier = LexicalQueryClassifier()
    assert (await classifier.score("the NEWSPAPER archive", []))[RAG] == 0.0
    assert (await classifier.score("any NEWS about it", []))[RAG] == pytest.approx(0.7)


async def test_custom_cues_replace_the_defaults():
    classifier = LexicalQueryClassifier({CAG: ("zebra",), MAG: (), RAG: ()})
    scores = await classifier.score("zebra today", [])
    assert scores == {CAG: pytest.approx(0.7), MAG: 0.0, RAG: 0.0}
