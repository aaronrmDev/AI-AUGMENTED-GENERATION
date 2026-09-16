from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.orchestration.domain.entities import IngestionResult, JobStatus

MAX_CONTENT_CHARS = 2_000_000  # generous text ceiling; MaxBodySizeMiddleware's 11 MiB
                                 # override on this route is the real backstop


class IngestDataSourceRequest(BaseModel):
    # Unknown fields are refused, not ignored: tenant_id/user_id can't be named
    # here -- both come only from the verified Caller, never the request body.
    model_config = ConfigDict(extra="forbid")

    source_key: str = Field(..., min_length=1)
    scope: Literal["tenant", "user"]
    expected_change_interval_seconds: int = Field(..., gt=0)
    content: str = Field(..., min_length=1, max_length=MAX_CONTENT_CHARS)

    @field_validator("source_key", "content")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class IngestDataSourceAcceptedResponse(BaseModel):
    task_id: str


class IngestionResultSchema(BaseModel):
    source_id: uuid.UUID
    route: str
    changed: bool

    @classmethod
    def of(cls, result: IngestionResult) -> IngestionResultSchema:
        return cls(source_id=result.source_id, route=result.route.value, changed=result.changed)


class JobStatusResponse(BaseModel):
    state: Literal["pending", "success", "failure"]
    result: IngestionResultSchema | None
    error: str | None

    @classmethod
    def of(cls, status: JobStatus) -> JobStatusResponse:
        return cls(
            state=status.state.value,
            result=None if status.result is None else IngestionResultSchema.of(status.result),
            error=status.error,
        )
