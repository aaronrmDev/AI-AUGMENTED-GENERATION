import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.identity.infrastructure.db import set_tenant_context
from src.mag.domain.entities import ProceduralMemory
from src.mag.infrastructure.postgres_procedural_memory_repository import (
    PostgresProceduralMemoryRepository,
)

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"


async def _create_user(db_session: AsyncSession, tenant_id: uuid.UUID) -> uuid.UUID:
    # set_tenant_context's setting is transaction-LOCAL (see its own comment
    # in src/identity/infrastructure/db.py) and this helper commits -- so
    # every caller must re-set the context after calling this, right before
    # its own RLS-sensitive operation. Mirrors
    # test_postgres_semantic_memory_repository.py's identical pattern.
    await set_tenant_context(db_session, tenant_id)
    user_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id) "
            "VALUES (:id, :email, :hashed_password, :tenant_id)"
        ),
        {
            "id": user_id,
            "email": f"{user_id}@example.com",
            "hashed_password": VALID_HASH,
            "tenant_id": tenant_id,
        },
    )
    await db_session.commit()
    return user_id


def _procedure(
    user_id: uuid.UUID, task_pattern: str, workflow: dict, success_rate: float = 0.0
) -> ProceduralMemory:
    return ProceduralMemory(
        id=uuid.uuid4(),
        user_id=user_id,
        task_pattern=task_pattern,
        workflow=workflow,
        success_rate=success_rate,
    )


async def test_save_then_find_by_task_pattern_round_trips(db_session):
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repo = PostgresProceduralMemoryRepository(db_session)

    await set_tenant_context(db_session, tenant_id)
    workflow = {"steps": ["build", "test", "push"], "tool_sequence": ["git", "docker"]}
    procedure = _procedure(user_id, "deploy_service", workflow, success_rate=0.75)
    await repo.save(procedure, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    found = await repo.find_by_task_pattern(user_id, "deploy_service", tenant_id)

    assert found is not None
    assert found.id == procedure.id
    assert found.user_id == user_id
    assert found.task_pattern == "deploy_service"
    assert found.workflow == workflow
    assert found.success_rate == 0.75
    assert found.last_used is None


async def test_saving_the_same_task_pattern_twice_updates_instead_of_duplicating(db_session):
    # Regression test: migration 0004's
    # uq_procedural_memory_user_id_task_pattern constraint plus save()'s
    # ON CONFLICT DO UPDATE must mean a second RecordProcedure for the same
    # (user_id, task_pattern) replaces the first row rather than creating a
    # second one resolved nondeterministically.
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repo = PostgresProceduralMemoryRepository(db_session)

    await set_tenant_context(db_session, tenant_id)
    await repo.save(
        _procedure(user_id, "deploy_service", {"steps": ["build"]}, success_rate=0.5), tenant_id
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    await repo.save(
        _procedure(
            user_id, "deploy_service", {"steps": ["build", "test", "push"]}, success_rate=0.9
        ),
        tenant_id,
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    count_result = await db_session.execute(
        text(
            "SELECT count(*) FROM procedural_memory "
            "WHERE user_id = :user_id AND task_pattern = :task_pattern"
        ),
        {"user_id": user_id, "task_pattern": "deploy_service"},
    )
    assert count_result.scalar_one() == 1

    found = await repo.find_by_task_pattern(user_id, "deploy_service", tenant_id)
    assert found is not None
    assert found.workflow == {"steps": ["build", "test", "push"]}
    assert found.success_rate == 0.9


async def test_find_by_task_pattern_returns_none_for_an_unknown_pattern(db_session):
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repo = PostgresProceduralMemoryRepository(db_session)

    await set_tenant_context(db_session, tenant_id)
    assert await repo.find_by_task_pattern(user_id, "unknown_pattern", tenant_id) is None


async def test_a_different_users_procedure_never_leaks_into_find_by_task_pattern(db_session):
    tenant_id = uuid.uuid4()
    user_a = await _create_user(db_session, tenant_id)
    user_b = await _create_user(db_session, tenant_id)
    repo = PostgresProceduralMemoryRepository(db_session)

    await set_tenant_context(db_session, tenant_id)
    await repo.save(
        _procedure(user_a, "deploy_service", {"steps": ["build"]}), tenant_id
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    assert await repo.find_by_task_pattern(user_b, "deploy_service", tenant_id) is None


async def test_a_different_tenants_procedure_never_leaks_into_find_by_task_pattern(db_session):
    # user_id alone used to be the only scoping key; a procedure recorded
    # for the same user_id under a different tenant_id must not be visible
    # either -- migration 0004's tenant_id/RLS closes exactly this gap.
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_a)
    repo = PostgresProceduralMemoryRepository(db_session)

    await set_tenant_context(db_session, tenant_a)
    await repo.save(
        _procedure(user_id, "deploy_service", {"steps": ["build"]}), tenant_a
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_a)
    assert await repo.find_by_task_pattern(user_id, "deploy_service", tenant_b) is None
