import uuid

from src.rag.application.answer_question import AnswerQuestion
from src.rag.application.search_documents import SearchDocuments
from src.rag.domain.entities import SearchResult
from tests.unit.rag_fakes import FakeChatModel, FakeEmbeddingModel, FakeVectorStore


async def test_answer_question_grounds_the_answer_in_retrieved_sources():
    vector_store = FakeVectorStore()
    sources = [
        SearchResult(
            document_id=uuid.uuid4(),
            chunk_id=uuid.uuid4(),
            content="FastAPI is a Python web framework.",
            score=0.95,
        ),
        SearchResult(
            document_id=uuid.uuid4(),
            chunk_id=uuid.uuid4(),
            content="It has automatic docs.",
            score=0.87,
        ),
    ]
    vector_store.set_search_results(sources)
    search = SearchDocuments(embedding_model=FakeEmbeddingModel(), vector_store=vector_store)
    chat_model = FakeChatModel(response="FastAPI is a web framework with automatic docs.")

    use_case = AnswerQuestion(search_documents=search, chat_model=chat_model, top_k=5)
    result = await use_case.execute(tenant_id=uuid.uuid4(), question="What is FastAPI?")

    assert result.answer == "FastAPI is a web framework with automatic docs."
    assert result.sources == sources
    assert "FastAPI is a Python web framework." in chat_model.last_context
    assert "It has automatic docs." in chat_model.last_context
    assert chat_model.last_question == "What is FastAPI?"
