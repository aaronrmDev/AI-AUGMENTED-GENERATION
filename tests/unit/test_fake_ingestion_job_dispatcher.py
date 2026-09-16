import uuid
from datetime import timedelta

import pytest

from src.orchestration.domain.entities import DataSourceProfile, JobState, SourceScope
from tests.unit.fakes import FakeIngestionJobDispatcher

TENANT = uuid.uuid4()
OTHER_TENANT = uuid.uuid4()
USER = uuid.uuid4()
OTHER_USER = uuid.uuid4()
PROFILE = DataSourceProfile("return-policy", SourceScope.TENANT, timedelta(days=90))


@pytest.mark.asyncio
async def test_dispatch_then_status_reports_success_with_the_real_result():
    dispatcher = FakeIngestionJobDispatcher()
    task_id = await dispatcher.dispatch(
        tenant_id=TENANT, profile=PROFILE, content="policy text", user_id=None
    )
    status = await dispatcher.status(task_id, tenant_id=TENANT, user_id=USER)
    assert status is not None
    assert status.state is JobState.SUCCESS
    assert status.result is not None
    assert status.result.changed is True


@pytest.mark.asyncio
async def test_status_denies_another_tenant():
    dispatcher = FakeIngestionJobDispatcher()
    task_id = await dispatcher.dispatch(
        tenant_id=TENANT, profile=PROFILE, content="policy text", user_id=None
    )
    assert await dispatcher.status(task_id, tenant_id=OTHER_TENANT, user_id=USER) is None


@pytest.mark.asyncio
async def test_status_denies_another_user_for_a_user_scoped_job():
    dispatcher = FakeIngestionJobDispatcher()
    user_profile = DataSourceProfile("shoe-size", SourceScope.USER, timedelta(days=30))
    task_id = await dispatcher.dispatch(
        tenant_id=TENANT, profile=user_profile, content="size 10", user_id=USER
    )
    assert await dispatcher.status(task_id, tenant_id=TENANT, user_id=OTHER_USER) is None


@pytest.mark.asyncio
async def test_status_is_none_for_an_unknown_task_id():
    dispatcher = FakeIngestionJobDispatcher()
    assert await dispatcher.status("no-such-task", tenant_id=TENANT, user_id=USER) is None
