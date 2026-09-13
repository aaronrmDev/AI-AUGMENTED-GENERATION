import pytest

from src.orchestration.domain.entities import Paradigm
from src.orchestration.domain.errors import ClassificationFailed
from src.orchestration.infrastructure.llm_query_classifier import LlmQueryClassifier
from tests.unit.rag_fakes import FakeChatModel

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG


async def test_a_clean_json_response_becomes_the_scores():
    chat_model = FakeChatModel('{"cag": 0.1, "mag": 0.0, "rag": 0.9}')
    classifier = LlmQueryClassifier(chat_model)
    assert await classifier.score("What changed today?", []) == {CAG: 0.1, MAG: 0.0, RAG: 0.9}
    assert "What changed today?" in chat_model.last_prompt
    assert classifier.parse_failures == 0


async def test_prose_around_the_json_object_is_tolerated():
    reply = 'Sure. {"cag": 1, "mag": 0.25, "rag": 0} Hope that helps.'
    scores = await LlmQueryClassifier(FakeChatModel(reply)).score("q", [])
    assert scores == {CAG: 1.0, MAG: 0.25, RAG: 0.0}


@pytest.mark.parametrize(
    "reply",
    [
        "no json here",
        '{"cag": 0.1, "mag": 0.2}',
        '{"cag": 1.5, "mag": 0.0, "rag": 0.0}',
        '{"cag": true, "mag": 0.0, "rag": 0.0}',
        '{"cag": "high", "mag": 0.0, "rag": 0.0}',
        "[0.1, 0.2, 0.3]",
        '{"cag": 0.1, "mag": 0.2, "rag": 0.3',
    ],
)
async def test_an_unusable_reply_is_an_explicit_classification_failure(reply):
    # Raised, not disguised as a score: a magic 0.5 only lands in the uncertainty
    # band at one particular threshold. The use case maps this to the fallback route.
    classifier = LlmQueryClassifier(FakeChatModel(reply))
    with pytest.raises(ClassificationFailed):
        await classifier.score("q", [])
    assert classifier.parse_failures == 1


async def test_a_query_containing_braces_does_not_break_the_prompt():
    chat_model = FakeChatModel('{"cag": 0.0, "mag": 0.0, "rag": 1.0}')
    await LlmQueryClassifier(chat_model).score("parse {this} json", [])
    assert "parse {this} json" in chat_model.last_prompt
