from datetime import timedelta

from fastapi import APIRouter, Depends, Request, Response

from src.api.caller import Caller
from src.api.dependencies import get_caller, get_ingestion_job_dispatcher, get_rate_limiter
from src.api.rate_limit import INGESTION_RATE_LIMIT, enforce_rate_limit
from src.api.schemas.data_sources import (
    IngestDataSourceAcceptedResponse,
    IngestDataSourceRequest,
    JobStatusResponse,
)
from src.orchestration.domain.entities import DataSourceProfile, SourceScope
from src.orchestration.domain.errors import IngestionJobNotFound
from src.orchestration.domain.ports import IngestionJobDispatcher

router = APIRouter(prefix="/data-sources", tags=["data-sources"])


@router.post("", response_model=IngestDataSourceAcceptedResponse, status_code=202)
async def ingest(
    payload: IngestDataSourceRequest,
    request: Request,
    response: Response,
    caller: Caller = Depends(get_caller),
    dispatcher: IngestionJobDispatcher = Depends(get_ingestion_job_dispatcher),
) -> IngestDataSourceAcceptedResponse:
    await enforce_rate_limit(
        request,
        response,
        limiter=get_rate_limiter(),
        key=f"data-sources:{caller.user_id}",
        limit=INGESTION_RATE_LIMIT,
    )
    profile = DataSourceProfile(
        payload.source_key,
        SourceScope(payload.scope),
        timedelta(seconds=payload.expected_change_interval_seconds),
    )
    task_id = await dispatcher.dispatch(
        tenant_id=caller.tenant_id,
        profile=profile,
        content=payload.content,
        user_id=caller.user_id if payload.scope == "user" else None,
    )
    return IngestDataSourceAcceptedResponse(task_id=task_id)


@router.get("/jobs/{task_id}", response_model=JobStatusResponse)
async def job_status(
    task_id: str,
    caller: Caller = Depends(get_caller),
    dispatcher: IngestionJobDispatcher = Depends(get_ingestion_job_dispatcher),
) -> JobStatusResponse:
    status = await dispatcher.status(task_id, tenant_id=caller.tenant_id, user_id=caller.user_id)
    if status is None:
        raise IngestionJobNotFound(task_id)
    return JobStatusResponse.of(status)
