import asyncio
import uuid

import redis.asyncio as redis
from celery import Celery
from celery.result import AsyncResult

from src.orchestration.domain.entities import (
    DataSourceProfile,
    IngestionResult,
    IngestionRoute,
    JobState,
    JobStatus,
)
from src.orchestration.domain.ports import IngestionJobDispatcher
from src.workers.celery_app import JOB_TTL_SECONDS

_OWNERSHIP_KEY_PREFIX = "ingestion_job:"


class CeleryIngestionJobDispatcher(IngestionJobDispatcher):
    """The real IngestionJobDispatcher: Celery for the work, Redis for who owns it.

    The ownership record is checked in status() *before* Celery's result backend is
    ever read -- a task id belonging to another tenant or user reports back exactly
    as if it never existed, the same "missing or not yours" 404 shape
    AnswerInSession already uses for sessions.
    """

    def __init__(self, celery_app: Celery, redis_client: redis.Redis) -> None:
        self._celery_app = celery_app
        self._redis = redis_client

    async def dispatch(
        self,
        *,
        tenant_id: uuid.UUID,
        profile: DataSourceProfile,
        content: str,
        user_id: uuid.UUID | None,
    ) -> str:
        def _send() -> str:
            result = self._celery_app.send_task(
                "ingest_data_source",
                kwargs={
                    "tenant_id": str(tenant_id),
                    "source_key": profile.source_key,
                    "scope": profile.scope.value,
                    "expected_change_interval_seconds": int(
                        profile.expected_change_interval.total_seconds()
                    ),
                    "content": content,
                    "user_id": str(user_id) if user_id is not None else None,
                },
            )
            return str(result.id)

        task_id = await asyncio.to_thread(_send)
        owner = f"{tenant_id}:{user_id if user_id is not None else ''}"
        await self._redis.set(f"{_OWNERSHIP_KEY_PREFIX}{task_id}", owner, ex=JOB_TTL_SECONDS)
        return task_id

    async def status(
        self, task_id: str, *, tenant_id: uuid.UUID, user_id: uuid.UUID
    ) -> JobStatus | None:
        owner = await self._redis.get(f"{_OWNERSHIP_KEY_PREFIX}{task_id}")
        if owner is None:
            return None
        owner_tenant_str, _, owner_user_str = owner.decode().partition(":")
        if uuid.UUID(owner_tenant_str) != tenant_id:
            return None
        if owner_user_str and uuid.UUID(owner_user_str) != user_id:
            return None

        def _read() -> JobStatus:
            async_result: AsyncResult = self._celery_app.AsyncResult(task_id)
            if async_result.state in ("PENDING", "STARTED", "RETRY"):
                return JobStatus(JobState.PENDING, None, None)
            if async_result.state == "SUCCESS":
                raw = async_result.result
                return JobStatus(
                    JobState.SUCCESS,
                    IngestionResult(
                        source_id=uuid.UUID(raw["source_id"]),
                        route=IngestionRoute(raw["route"]),
                        changed=raw["changed"],
                    ),
                    None,
                )
            # FAILURE, or any other terminal-but-not-SUCCESS state.
            return JobStatus(JobState.FAILURE, None, str(async_result.result))

        return await asyncio.to_thread(_read)
