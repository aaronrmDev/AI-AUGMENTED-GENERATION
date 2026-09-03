import uuid

from src.mag.application.queries.find_procedure import FindProcedure
from src.mag.domain.entities import ProceduralMemory
from tests.unit.mag_fakes import FakeProceduralMemoryRepository


def _procedure(user_id: uuid.UUID, task_pattern: str = "deploy_service") -> ProceduralMemory:
    return ProceduralMemory(
        id=uuid.uuid4(),
        user_id=user_id,
        task_pattern=task_pattern,
        workflow={"steps": ["build", "test", "push"]},
    )


async def test_by_task_pattern_delegates_to_find_by_task_pattern():
    repository = FakeProceduralMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    procedure = _procedure(user_id)
    await repository.save(procedure, tenant_id)

    query = FindProcedure(procedural_memory_repository=repository)
    result = await query.by_task_pattern(user_id, "deploy_service", tenant_id)

    assert result == procedure


async def test_by_task_pattern_returns_none_for_an_unknown_pattern():
    query = FindProcedure(procedural_memory_repository=FakeProceduralMemoryRepository())

    result = await query.by_task_pattern(uuid.uuid4(), "unknown_pattern", uuid.uuid4())

    assert result is None
