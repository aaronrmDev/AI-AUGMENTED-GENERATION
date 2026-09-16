import uuid

from fastapi import APIRouter, Depends, Query, Request, Response

from src.api.caller import Caller
from src.api.dependencies import (
    get_answer_in_session,
    get_caller,
    get_chat_session_repository,
    get_rate_limiter,
)
from src.api.rate_limit import (
    GLOBAL_CHAT_KEY,
    GLOBAL_CHAT_WINDOW_SECONDS,
    SESSION_CREATE_LIMIT,
    chat_rate_limit,
    enforce_rate_limit,
    global_chat_limit,
)
from src.api.schemas.sessions import (
    AnswerRequest,
    AnswerResponse,
    CreateSessionRequest,
    SessionListResponse,
    SessionResponse,
    answer_response,
)
from src.identity.application.list_chat_sessions import MAX_SESSIONS_PER_PAGE, ListChatSessions
from src.identity.application.start_chat_session import StartChatSession
from src.identity.domain.ports import ChatSessionRepository
from src.orchestration.application.answer_in_session import AnswerInSession

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    payload: CreateSessionRequest,
    request: Request,
    response: Response,
    caller: Caller = Depends(get_caller),
    sessions: ChatSessionRepository = Depends(get_chat_session_repository),
) -> SessionResponse:
    await enforce_rate_limit(
        request,
        response,
        limiter=get_rate_limiter(),
        key=f"sessions:{caller.user_id}",
        limit=SESSION_CREATE_LIMIT,
    )
    session = await StartChatSession(sessions).execute(
        caller.tenant_id, caller.user_id, payload.title
    )
    return SessionResponse.of(session)


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    caller: Caller = Depends(get_caller),
    sessions: ChatSessionRepository = Depends(get_chat_session_repository),
    limit: int = Query(default=20, ge=1, le=MAX_SESSIONS_PER_PAGE),
) -> SessionListResponse:
    found = await ListChatSessions(sessions).execute(caller.tenant_id, caller.user_id, limit)
    return SessionListResponse(sessions=[SessionResponse.of(s) for s in found])


@router.post("/{session_id}/answers", response_model=AnswerResponse)
async def answer(
    session_id: uuid.UUID,
    payload: AnswerRequest,
    request: Request,
    response: Response,
    # Declared before the pipeline, so a request without a valid token is refused
    # before anything builds the pipeline.
    caller: Caller = Depends(get_caller),
    answer_in_session: AnswerInSession = Depends(get_answer_in_session),
) -> AnswerResponse:
    # Per-user budget checked first, global quota second: a single account
    # already over its own limit is rejected here, before it can ever touch
    # (and burn down) the shared global counter -- checking the other order
    # would let one abusive account exhaust everyone else's global budget via
    # requests that were always going to be rejected anyway.
    await enforce_rate_limit(
        request,
        response,
        limiter=get_rate_limiter(),
        key=f"chat:{caller.user_id}",
        limit=chat_rate_limit(),
    )
    await enforce_rate_limit(
        request,
        response,
        limiter=get_rate_limiter(),
        key=GLOBAL_CHAT_KEY,
        limit=global_chat_limit(),
        window_seconds=GLOBAL_CHAT_WINDOW_SECONDS,
    )
    result = await answer_in_session.execute(
        caller.tenant_id, caller.user_id, session_id, payload.question
    )
    return answer_response(result)
