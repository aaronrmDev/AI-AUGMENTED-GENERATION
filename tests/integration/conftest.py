import os

import pytest
import pytest_asyncio
from alembic.config import Config
from sqlalchemy.engine import make_url
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

from alembic import command
from src.identity.infrastructure.db import get_engine, get_sessionmaker

# Never used outside a throwaway TestContainers instance.
_APP_DB_PASSWORD = "test-only-app-user-password"


@pytest.fixture(scope="session")
def postgres_container():
    with PostgresContainer("pgvector/pgvector:pg16") as container:
        yield container


@pytest.fixture(scope="session")
def redis_container():
    with RedisContainer("redis:7") as container:
        yield container


@pytest.fixture(scope="session")
def database_url(postgres_container: PostgresContainer) -> str:
    url = postgres_container.get_connection_url()
    return url.replace("postgresql+psycopg2", "postgresql+asyncpg")


@pytest.fixture(scope="session")
def redis_url(redis_container: RedisContainer) -> str:
    # Use 127.0.0.1 explicitly instead of get_container_host_ip() which returns "localhost"
    # On Windows, localhost resolves to ::1 (IPv6) which doesn't work with TestContainers
    port = redis_container.get_exposed_port(6379)
    return f"redis://127.0.0.1:{port}/0"


@pytest.fixture(scope="session", autouse=True)
def run_migrations(database_url: str):
    os.environ["DATABASE_URL"] = database_url
    os.environ["APP_DB_PASSWORD"] = _APP_DB_PASSWORD
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    yield


@pytest.fixture(scope="session")
def app_database_url(database_url: str, run_migrations) -> str:
    # Depends on run_migrations (not just database_url) because app_user
    # doesn't exist until the migration creates it.
    #
    # str(url) is deliberately NOT used here: SQLAlchemy's URL.__str__ calls
    # render_as_string(hide_password=True) by default, which replaces the
    # password with the literal string "***" — that's meant for safe logging,
    # not for building a connection string, and using it here silently
    # produces a URL asyncpg can never authenticate with.
    url = make_url(database_url)
    app_url = url.set(username="app_user", password=_APP_DB_PASSWORD)
    return app_url.render_as_string(hide_password=False)


@pytest_asyncio.fixture
async def db_session(app_database_url: str):
    engine = get_engine(app_database_url)
    sessionmaker = get_sessionmaker(engine)
    async with sessionmaker() as session:
        yield session
        await session.rollback()
    await engine.dispose()


from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder


@pytest.fixture(scope="session")
def embedding_model() -> SentenceTransformersEmbedder:
    return SentenceTransformersEmbedder()


from testcontainers.qdrant import QdrantContainer


@pytest.fixture(scope="session")
def qdrant_container():
    with QdrantContainer() as container:
        yield container


@pytest.fixture(scope="session")
def qdrant_url(qdrant_container: QdrantContainer) -> str:
    # Use 127.0.0.1 explicitly instead of get_client().rest_uri / the
    # container's own rest_host_address, for two separate reasons:
    #
    # 1. The installed qdrant-client (1.19.0) / testcontainers (4.15.0)
    #    don't expose a `rest_uri` attribute on the `QdrantClient` that
    #    `get_client()` returns — it exists only on the internal
    #    `QdrantRemote` transport object, not the public wrapper — so
    #    `qdrant_container.get_client().rest_uri` raises AttributeError.
    # 2. `rest_host_address` builds its URL from get_container_host_ip(),
    #    which returns "localhost" and hits the same Windows/IPv6 problem
    #    documented on `redis_url` above.
    port = qdrant_container.get_exposed_port(6333)
    return f"http://127.0.0.1:{port}"
