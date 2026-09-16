import uuid

import pytest
from pydantic import ValidationError

from src.orchestration.domain.entities import IngestionResult, IngestionRoute, JobState, JobStatus
from src.api.schemas.data_sources import (
    IngestDataSourceRequest,
    IngestionResultSchema,
    JobStatusResponse,
)


def test_a_valid_request_parses():
    request = IngestDataSourceRequest(
        source_key="return-policy",
        scope="tenant",
        expected_change_interval_seconds=90 * 24 * 60 * 60,
        content="Our return policy allows returns within 90 days.",
    )
    assert request.scope == "tenant"


def test_a_non_positive_interval_is_rejected():
    with pytest.raises(ValidationError):
        IngestDataSourceRequest(
            source_key="return-policy", scope="tenant",
            expected_change_interval_seconds=0, content="text",
        )


def test_blank_content_is_rejected():
    with pytest.raises(ValidationError):
        IngestDataSourceRequest(
            source_key="return-policy", scope="tenant",
            expected_change_interval_seconds=3600, content="   ",
        )


def test_an_unknown_field_is_refused_not_ignored():
    with pytest.raises(ValidationError):
        IngestDataSourceRequest(
            source_key="return-policy", scope="tenant",
            expected_change_interval_seconds=3600, content="text",
            tenant_id=str(uuid.uuid4()),
        )


def test_job_status_response_of_a_success_carries_the_result():
    result = IngestionResult(source_id=uuid.uuid4(), route=IngestionRoute.RAG_ONLY, changed=True)
    response = JobStatusResponse.of(JobStatus(JobState.SUCCESS, result, None))
    assert response.state == "success"
    assert response.result is not None
    assert response.result.route == "rag_only"
    assert response.error is None


def test_job_status_response_of_a_failure_carries_the_error():
    response = JobStatusResponse.of(JobStatus(JobState.FAILURE, None, "a tenant-scoped source cannot route to MAG"))
    assert response.state == "failure"
    assert response.result is None
    assert response.error == "a tenant-scoped source cannot route to MAG"
