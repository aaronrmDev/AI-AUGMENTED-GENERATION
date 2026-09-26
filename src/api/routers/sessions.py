import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import StreamingResponse

from src.api.caller import Caller
from src.api.dependencies import (
    get_answer_in_session,
    get_caller,
    get_chat_session_repository,
    get_rate_limiter,
    get_stream_concurrency_limiter,
)
from src.api.rate_limit import (
    GLOBAL_CHAT_KEY,
    GLOBAL_CHAT_WINDOW_SECONDS,
    SESSION_CREATE_LIMIT,
    StreamConcurrencyLimitExceeded,
    chat_rate_limit,
    enforce_rate_limit,
    global_chat_limit,
    stream_concurrency_limit,
    stream_slot_ttl_seconds,
)
from src.api.schemas.sessions import (
    AnswerRequest,
    AnswerResponse,
    CreateSessionRequest,
    SessionListResponse,
    SessionResponse,
    answer_response,
)
from src.api.sse import encode_sse
from src.identity.application.list_chat_sessions import MAX_SESSIONS_PER_PAGE, ListChatSessions
from src.identity.application.start_chat_session import StartChatSession
from src.identity.domain.ports import ChatSessionRepository
from src.orchestration.application.answer_in_session import AnswerInSession
from src.orchestration.application.unified_answer_question import UnifiedAnswerEvent

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


@router.post(
    "/{session_id}/answers/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def answer_stream(
    session_id: uuid.UUID,
    payload: AnswerRequest,
    request: Request,
    response: Response,
    caller: Caller = Depends(get_caller),
    answer_in_session: AnswerInSession = Depends(get_answer_in_session),
) -> StreamingResponse:
    # Same two checks, same order, same keys and limits as answer() above --
    # this is the same resource's budget, not a second one.
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
    # Raises SessionNotFound/QueryExceedsBudget HERE, before StreamingResponse
    # is ever constructed -- see AnswerInSession.stream()'s docstring for why
    # that's what keeps a denial a plain 404/422 instead of a stream that
    # opens and then errors.
    events = await answer_in_session.stream(
        caller.tenant_id, caller.user_id, session_id, payload.question
    )

    # Acquired only now that generation is actually about to start, never
    # earlier: every rejection above (rate limit, SessionNotFound,
    # QueryExceedsBudget) has to stay a plain, cheap failure that never so
    # much as touches a concurrency slot.
    limiter = get_stream_concurrency_limiter()
    slot_key = f"stream:{caller.user_id}"
    slot_token = await limiter.acquire(
        key=slot_key, limit=stream_concurrency_limit(), ttl_seconds=stream_slot_ttl_seconds()
    )
    if slot_token is None:
        raise StreamConcurrencyLimitExceeded(limit=stream_concurrency_limit(), key=slot_key)

    async def _release_slot_when_done(
        events: AsyncIterator[UnifiedAnswerEvent],
    ) -> AsyncIterator[UnifiedAnswerEvent]:
        try:
            async for event in events:
                yield event
        finally:
            await limiter.release(key=slot_key, slot_token=slot_token)

    return StreamingResponse(
        encode_sse(_release_slot_when_done(events)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"},
    )
