import uuid

from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.commands.record_procedure import RecordProcedure
from src.mag.infrastructure.postgres_procedural_memory_repository import (
    PostgresProceduralMemoryRepository,
)

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"


async def _create_user(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
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


async def test_execute_writes_to_real_postgres(db_session):
    # This is the real seam RecordProcedure's own unit tests can't reach --
    # they inject a fake repository, so a bug in the real ON CONFLICT SQL
    # (a typo in a column name, a mismatched CAST) is invisible at the unit
    # level. This test constructs the real command against real
    # infrastructure end to end, mirroring
    # test_record_semantic_fact_command.py's identical purpose.
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repository = PostgresProceduralMemoryRepository(db_session)
    command = RecordProcedure(procedural_memory_repository=repository)

    await set_tenant_context(db_session, tenant_id)
    workflow = {"steps": ["build", "test", "push"]}
    procedure = await command.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        task_pattern="deploy_service",
        workflow=workflow,
        success_rate=0.8,
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    found = await repository.find_by_task_pattern(user_id, "deploy_service", tenant_id)
    assert found is not None
    assert found.id == procedure.id
    assert found.workflow == workflow
    assert found.success_rate == 0.8


async def test_recording_the_same_task_pattern_twice_does_not_orphan_a_stale_row(db_session):
    # Regression test mirroring
    # test_recording_the_same_key_twice_does_not_orphan_a_stale_qdrant_point:
    # there's no Qdrant point to orphan here (ProceduralMemory has no
    # embedding), but the deterministic-id invariant still matters for
    # Postgres itself -- recording the same (user_id, task_pattern) twice
    # through the REAL command against REAL Postgres must end with exactly
    # one row carrying the newer data, not two rows resolved
    # nondeterministically.
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repository = PostgresProceduralMemoryRepository(db_session)
    command = RecordProcedure(procedural_memory_repository=repository)

    await set_tenant_context(db_session, tenant_id)
    await command.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        task_pattern="deploy_service",
        workflow={"steps": ["build"]},
        success_rate=0.3,
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    await command.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        task_pattern="deploy_service",
        workflow={"steps": ["build", "test", "push"]},
        success_rate=0.95,
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

    found = await repository.find_by_task_pattern(user_id, "deploy_service", tenant_id)
    assert found is not None
    assert found.workflow == {"steps": ["build", "test", "push"]}
    assert found.success_rate == 0.95
