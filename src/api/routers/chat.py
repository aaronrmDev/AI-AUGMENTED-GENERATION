import uuid

from fastapi import APIRouter, Depends

from src.api.dependencies import get_chat_model, get_current_user_claims, get_embedding_model, get_vector_store
from src.api.schemas.chat import ChatRequest, ChatResponse, ChatSourceSchema
from src.rag.application.answer_question import AnswerQuestion
from src.rag.application.search_documents import SearchDocuments

router = APIRouter(prefix="/chat", tags=["chat"])

_TOP_K = 5


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    claims: dict = Depends(get_current_user_claims),
) -> ChatResponse:
    tenant_id = uuid.UUID(claims["tenant_id"])
    search = SearchDocuments(embedding_model=get_embedding_model(), vector_store=get_vector_store())
    use_case = AnswerQuestion(search_documents=search, chat_model=get_chat_model(), top_k=_TOP_K)
    result = await use_case.execute(tenant_id=tenant_id, question=payload.question)

    return ChatResponse(
        answer=result.answer,
        sources=[
            ChatSourceSchema(document_id=s.document_id, chunk_id=s.chunk_id, content=s.content)
            for s in result.sources
        ],
    )
