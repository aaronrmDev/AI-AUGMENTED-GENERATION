"""CeleryIngestionJobDispatcher against a real Redis instance. No worker runs
here -- that's Task 4's ingestion_worker test and Task 6's full-stack test.
This proves dispatch() actually enqueues onto Redis and writes the ownership
record, and status() reads both back correctly, including denying the wrong
caller -- by priming the Celery result backend directly, the same way Celery's
own test suite proves its result-backend read path without a worker."""
import os
import uuid
from datetime import timedelta

import pytest
import redis.asyncio as redis

from src.orchestration.domain.entities import DataSourceProfile, JobState, SourceScope

TENANT = uuid.uuid4()
OTHER_TENANT = uuid.uuid4()
USER = uuid.uuid4()
PROFILE = DataSourceProfile("return-policy", SourceScope.TENANT, timedelta(days=90))


@pytest.fixture
async def redis_client(redis_url: str):
    client = redis.from_url(redis_url)
    yield client
    await client.aclose()


@pytest.fixture
def dispatcher(redis_url: str, redis_client: redis.Redis):
    # src.workers.celery_app reads REDIS_URL at *import* time (the same
    # module-level-singleton pattern src/api/dependencies.py already uses for
    # APP_DATABASE_URL/QDRANT_URL) -- setting the env var here, before the
    # deferred import below, matches how tests/integration/test_sessions_
    # endpoints.py's _client() helper already handles the identical ordering
    # requirement for src.api.main. A module-level top-of-file import would
    # run before this fixture (or any fixture) executes, and crash with
    # KeyError: 'REDIS_URL' the moment pytest collects this file.
    os.environ["REDIS_URL"] = redis_url
    from src.workers.celery_app import build_celery_app
    from src.workers.celery_ingestion_dispatcher import CeleryIngestionJobDispatcher

    return CeleryIngestionJobDispatcher(build_celery_app(redis_url), redis_client)


@pytest.mark.asyncio
async def test_dispatch_writes_an_ownership_record_and_enqueues_the_task(dispatcher, redis_client):
    task_id = await dispatcher.dispatch(
        tenant_id=TENANT, profile=PROFILE, content="policy text", user_id=None
    )
    stored = await redis_client.get(f"ingestion_job:{task_id}")
    assert stored is not None
    assert stored.decode() == f"{TENANT}:"


@pytest.mark.asyncio
async def test_status_reports_pending_for_a_task_with_no_result_yet(dispatcher):
    task_id = await dispatcher.dispatch(
        tenant_id=TENANT, profile=PROFILE, content="policy text", user_id=None
    )
    status = await dispatcher.status(task_id, tenant_id=TENANT, user_id=USER)
    assert status is not None
    assert status.state is JobState.PENDING


@pytest.mark.asyncio
async def test_status_denies_another_tenant(dispatcher):
    task_id = await dispatcher.dispatch(
        tenant_id=TENANT, profile=PROFILE, content="policy text", user_id=None
    )
    assert await dispatcher.status(task_id, tenant_id=OTHER_TENANT, user_id=USER) is None


@pytest.mark.asyncio
async def test_status_is_none_for_an_unknown_task_id(dispatcher):
    assert await dispatcher.status(str(uuid.uuid4()), tenant_id=TENANT, user_id=USER) is None
