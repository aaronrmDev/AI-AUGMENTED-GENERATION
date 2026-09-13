"""Live checks against the local Ollama qwen3.5 model. Skips, with the reason,
wherever Ollama or the model is unavailable -- the same shape as the
vLLM-dependent CAG tests."""
import ollama
import pytest

from src.orchestration.application.unified_answer_question import UnifiedAnswerQuestion
from src.orchestration.domain.entities import PARADIGM_ORDER, Paradigm
from src.orchestration.infrastructure.llm_query_classifier import LlmQueryClassifier
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel
from tests.integration.orchestration_env import (
    FRESHNESS_QUERY,
    FixedScoresClassifier,
    build_env,
)

_MODEL_ID = "qwen3.5"


@pytest.fixture(scope="module")
def qwen_available() -> None:
    try:
        names = [model.model or "" for model in ollama.Client().list().models]
    except Exception as exc:  # any connection failure means "not available here"
        pytest.skip(f"Ollama is not reachable at its default host: {exc}")
    if not any(name.split(":")[0] == _MODEL_ID for name in names):
        pytest.skip(f"{_MODEL_ID} is not pulled into the local Ollama")


async def test_the_real_model_returns_parseable_routing_scores(qwen_available):
    classifier = LlmQueryClassifier(OllamaChatModel(ollama.AsyncClient(), _MODEL_ID))
    for query in (
        "What's our refund policy?",
        "What changed in the policy today?",
        "What did I ask you to remember?",
    ):
        scores = await classifier.score(query, [])
        print(f"{query!r} -> {scores}")
        assert set(scores) == set(PARADIGM_ORDER)
    assert classifier.parse_failures == 0


async def test_the_unified_pipeline_answers_a_freshness_question_from_the_current_document(
    qwen_available,
    db_session,
    qdrant_url,
    embedding_model,
    distilgpt2_tokenizer,
    distilgpt2_model,
):
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )
    use_case = UnifiedAnswerQuestion(
        embedding_model,
        FixedScoresClassifier({Paradigm.CAG: 0.0, Paradigm.MAG: 0.0, Paradigm.RAG: 1.0}),
        env.cascade(),
        OllamaChatModel(ollama.AsyncClient(), _MODEL_ID),
    )
    result = await use_case.execute(env.tenant_id, env.user_id, env.session_id, FRESHNESS_QUERY)
    print(f"answer: {result.answer!r}")
    # What the pipeline controls: only the current document reached the model.
    assert result.sources
    assert all(source.paradigm is Paradigm.RAG for source in result.sources)
    assert any("forty-five days" in source.content for source in result.sources)
    # What the model then did with it: no superseded value, and the current one stated.
    answer = result.answer.lower()
    assert "thirty" not in answer and "30" not in answer
    assert "45" in answer or "forty-five" in answer
