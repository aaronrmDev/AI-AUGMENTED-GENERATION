import uuid
from datetime import UTC, datetime

from sqlalchemy import text

from src.identity.infrastructure.db import get_sessionmaker
from src.identity.infrastructure.postgres_chat_session_repository import (
    PostgresChatSessionRepository,
)
from src.orchestration.domain.budget_allocator import allocate
from src.orchestration.domain.entities import Paradigm
from src.orchestration.infrastructure.postgres_session_budget_recorder import (
    PostgresSessionBudgetRecorder,
)

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"


async def _user(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
    now, user_id = datetime.now(UTC), uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id, created_at, updated_at) "
            "VALUES (:id, :email, :hashed_password, :tenant_id, :created_at, :updated_at)"
        ),
        {
            "id": user_id, "email": f"{user_id}@example.com", "hashed_password": VALID_HASH,
            "tenant_id": tenant_id, "created_at": now, "updated_at": now,
        },
    )
    await db_session.commit()
    return user_id


def _repository(db_session) -> PostgresChatSessionRepository:
    # The same engine as db_session, through the repository's own sessions: the
    # production shape, where each call owns its transaction.
    return PostgresChatSessionRepository(get_sessionmaker(db_session.bind))


async def test_a_created_session_is_found_by_its_owner_with_server_generated_fields(db_session):
    tenant_id = uuid.uuid4()
    user_id = await _user(db_session, tenant_id)
    repository = _repository(db_session)

    created = await repository.create(tenant_id, user_id, "Returns")

    assert (created.tenant_id, created.user_id, created.title) == (tenant_id, user_id, "Returns")
    assert created.context_budget is None
    assert created.created_at.tzinfo is not None
    assert await repository.find_owned(tenant_id, user_id, created.id) == created


async def test_another_user_in_the_same_tenant_finds_nothing(db_session):
    tenant_id = uuid.uuid4()
    owner, other = await _user(db_session, tenant_id), await _user(db_session, tenant_id)
    repository = _repository(db_session)
    created = await repository.create(tenant_id, owner, None)

    assert await repository.find_owned(tenant_id, other, created.id) is None


async def test_the_owner_under_another_tenant_finds_nothing(db_session):
    tenant_id = uuid.uuid4()
    owner = await _user(db_session, tenant_id)
    repository = _repository(db_session)
    created = await repository.create(tenant_id, owner, None)

    assert await repository.find_owned(uuid.uuid4(), owner, created.id) is None


async def test_listing_returns_the_owners_sessions_newest_first_up_to_the_limit(db_session):
    tenant_id = uuid.uuid4()
    owner = await _user(db_session, tenant_id)
    repository = _repository(db_session)
    created = [await repository.create(tenant_id, owner, f"s{i}") for i in range(3)]

    assert await repository.list_owned(tenant_id, owner, 2) == [created[2], created[1]]


async def test_listing_never_includes_another_users_or_another_tenants_sessions(db_session):
    tenant_id, other_tenant = uuid.uuid4(), uuid.uuid4()
    owner, colleague = await _user(db_session, tenant_id), await _user(db_session, tenant_id)
    stranger = await _user(db_session, other_tenant)
    repository = _repository(db_session)
    mine = await repository.create(tenant_id, owner, "mine")
    await repository.create(tenant_id, colleague, "a colleague's")
    await repository.create(other_tenant, stranger, "another tenant's")

    assert await repository.list_owned(tenant_id, owner, 100) == [mine]
    assert await repository.list_owned(other_tenant, owner, 100) == []


async def test_a_recorded_budget_is_read_back_as_a_dict(db_session):
    tenant_id = uuid.uuid4()
    owner = await _user(db_session, tenant_id)
    repository = _repository(db_session)
    created = await repository.create(tenant_id, owner, None)
    contributing = frozenset({Paradigm.RAG})
    await PostgresSessionBudgetRecorder(get_sessionmaker(db_session.bind)).record(
        tenant_id, owner, created.id, allocate(128_000, contributing), contributing
    )

    found = await repository.find_owned(tenant_id, owner, created.id)

    assert found is not None and found.context_budget is not None
    assert found.context_budget["total"] == 128_000
    assert found.context_budget["contributing"] == ["rag"]
