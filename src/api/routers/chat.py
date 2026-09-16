from fastapi import APIRouter, Depends, Request, Response

from src.api.caller import Caller
from src.api.dependencies import (
    get_caller,
    get_chat_model,
    get_embedding_model,
    get_rate_limiter,
    get_vector_store,
)
from src.api.rate_limit import (
    GLOBAL_CHAT_KEY,
    chat_rate_limit,
    enforce_rate_limit,
    global_chat_limit,
)
from src.api.schemas.chat import ChatRequest, ChatResponse, ChatSourceSchema
from src.rag.application.answer_question import AnswerQuestion
from src.rag.application.search_documents import SearchDocuments

router = APIRouter(prefix="/chat", tags=["chat"])

_TOP_K = 5


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    response: Response,
    caller: Caller = Depends(get_caller),
) -> ChatResponse:
    # Global quota checked first, per-user budget second: both write their
    # X-RateLimit-* headers onto the same `response`, and the second call's
    # values are what the client ends up seeing on a 200. Checking the
    # per-user limit last keeps those headers the ones callers can act on
    # (their own remaining budget) rather than the shared global counter's.
    await enforce_rate_limit(
        request,
        response,
        limiter=get_rate_limiter(),
        key=GLOBAL_CHAT_KEY,
        limit=global_chat_limit(),
        window_seconds=3600,
    )
    # The same per-user budget answering in a session draws from, under the same key.
    await enforce_rate_limit(
        request,
        response,
        limiter=get_rate_limiter(),
        key=f"chat:{caller.user_id}",
        limit=chat_rate_limit(),
    )
    search = SearchDocuments(embedding_model=get_embedding_model(), vector_store=get_vector_store())
    use_case = AnswerQuestion(search_documents=search, chat_model=get_chat_model(), top_k=_TOP_K)
    result = await use_case.execute(tenant_id=caller.tenant_id, question=payload.question)

    return ChatResponse(
        answer=result.answer,
        sources=[
            ChatSourceSchema(document_id=s.document_id, chunk_id=s.chunk_id, content=s.content)
            for s in result.sources
        ],
    )
