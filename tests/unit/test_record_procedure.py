import uuid

from src.mag.application.commands.record_procedure import RecordProcedure
from tests.unit.mag_fakes import FakeProceduralMemoryRepository


async def test_execute_saves_to_the_repository():
    repository = FakeProceduralMemoryRepository()
    command = RecordProcedure(procedural_memory_repository=repository)

    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    procedure = await command.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        task_pattern="deploy_service",
        workflow={"steps": ["build", "test", "push"]},
    )

    assert repository.saved == [(procedure, tenant_id)]


async def test_execute_derives_the_same_id_for_the_same_user_and_task_pattern():
    # The id is deterministic (uuid5 of user_id+task_pattern), not random --
    # this is the invariant that keeps a re-recorded procedure from
    # duplicating instead of overwriting, matching RecordSemanticFact's
    # identical corrected shape (see record_procedure.py's comment and
    # test_execute_derives_the_same_id_for_the_same_user_and_fact_key in
    # test_record_semantic_fact.py). A unit test asserting only "two
    # DIFFERENT patterns get different ids" can't tell a correct
    # deterministic scheme apart from a reverted uuid4() -- this test is the
    # fast, Docker-free guard for the specific property that matters.
    command = RecordProcedure(procedural_memory_repository=FakeProceduralMemoryRepository())
    user_id = uuid.uuid4()

    first = await command.execute(
        tenant_id=uuid.uuid4(),
        user_id=user_id,
        task_pattern="deploy_service",
        workflow={"steps": ["a"]},
    )
    second = await command.execute(
        tenant_id=uuid.uuid4(),
        user_id=user_id,
        task_pattern="deploy_service",
        workflow={"steps": ["a", "b"]},
    )
    different_pattern = await command.execute(
        tenant_id=uuid.uuid4(),
        user_id=user_id,
        task_pattern="rollback_service",
        workflow={"steps": ["a"]},
    )
    different_user = await command.execute(
        tenant_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        task_pattern="deploy_service",
        workflow={"steps": ["a"]},
    )

    assert isinstance(first.id, uuid.UUID)
    assert first.id == second.id
    assert first.id != different_pattern.id
    assert first.id != different_user.id


async def test_execute_defaults_success_rate_to_zero():
    command = RecordProcedure(procedural_memory_repository=FakeProceduralMemoryRepository())

    procedure = await command.execute(
        tenant_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        task_pattern="deploy_service",
        workflow={"steps": ["a"]},
    )

    assert procedure.success_rate == 0.0


async def test_execute_stores_the_given_user_id_task_pattern_and_workflow():
    command = RecordProcedure(procedural_memory_repository=FakeProceduralMemoryRepository())

    user_id = uuid.uuid4()
    workflow = {"steps": ["build", "test", "push"], "tool_sequence": ["git", "docker"]}
    procedure = await command.execute(
        tenant_id=uuid.uuid4(),
        user_id=user_id,
        task_pattern="deploy_service",
        workflow=workflow,
        success_rate=0.85,
    )

    assert procedure.user_id == user_id
    assert procedure.task_pattern == "deploy_service"
    assert procedure.workflow == workflow
    assert procedure.success_rate == 0.85


async def test_execute_recording_the_same_task_pattern_twice_updates_the_fake_not_duplicates():
    # Mirrors the real repository's ON CONFLICT (user_id, task_pattern) DO
    # UPDATE -- this fake's upsert semantics (see mag_fakes.py) must agree
    # with it, or a unit test here proves nothing about the real integration
    # behavior.
    repository = FakeProceduralMemoryRepository()
    command = RecordProcedure(procedural_memory_repository=repository)
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()

    await command.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        task_pattern="deploy_service",
        workflow={"steps": ["a"]},
    )
    await command.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        task_pattern="deploy_service",
        workflow={"steps": ["a", "b"]},
    )

    found = await repository.find_by_task_pattern(user_id, "deploy_service", tenant_id)
    assert found is not None
    assert found.workflow == {"steps": ["a", "b"]}
