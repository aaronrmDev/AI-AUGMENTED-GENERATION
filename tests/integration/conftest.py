import os

import pytest
import pytest_asyncio
import redis.asyncio as redis
from alembic.config import Config
from sqlalchemy.engine import make_url
from testcontainers.neo4j import Neo4jContainer
from testcontainers.postgres import PostgresContainer
from testcontainers.qdrant import QdrantContainer
from testcontainers.redis import RedisContainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from alembic import command
from src.identity.infrastructure.db import get_engine, get_sessionmaker
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder

_DISTILGPT2_MODEL_ID = "distilgpt2"

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


@pytest_asyncio.fixture(autouse=True)
async def _clean_redis_between_all_integration_tests(redis_url: str):
    # The rate limiter and the refresh-token store both key off state that
    # lives in this session-scoped Redis container, so one test's writes
    # (most visibly test_sixth_request_in_a_window_is_rate_limited in
    # test_auth_endpoints.py, which deliberately exhausts the /auth/login
    # rate limit) otherwise leak into whichever test runs next -- and with
    # Redis shared across the whole session rather than scoped per file, that
    # means *any* later file that calls /auth/login, not just later tests
    # within test_auth_endpoints.py itself. Task 14's
    # test_documents_endpoints.py became the first such file (via its own
    # _register_and_login helper), which is what surfaced this. This fixture
    # replaces what used to be a module-local, same-purpose fixture defined
    # only inside test_auth_endpoints.py -- generalizing it here, rather than
    # duplicating it into every new test file, covers every integration test
    # uniformly with a single flush per test.
    client = redis.from_url(redis_url)
    await client.flushdb()
    yield
    await client.aclose()


@pytest_asyncio.fixture(autouse=True)
async def _dispose_api_engine_after_each_test():
    # src/api/dependencies.py builds its Postgres AsyncEngine (and asyncpg
    # connection pool) exactly once, at first import, as a process-wide
    # module-level singleton -- deliberately, to avoid reopening a pool per
    # request. But pytest-asyncio gives each test module its own event loop
    # (loop_scope="module" in test_auth_endpoints.py and
    # test_documents_endpoints.py; the function-scoped default elsewhere),
    # and an asyncpg connection is bound to whichever loop was running when it
    # was opened. A connection opened while servicing one module's loop, then
    # left idle in the pool, is unusable once that loop closes and a
    # different module's loop starts -- surfacing deep inside asyncpg as
    # "Event loop is closed" / "'NoneType' object has no attribute 'send'".
    # Disposing the pool after every test (while its own loop is still the
    # active one, so the close can actually complete) guarantees the next
    # test always opens fresh connections under whatever loop is current then,
    # so no connection ever has to survive past the loop it was born on.
    # Import is deferred and looked up via sys.modules (not a top-level
    # import) so tests that never touch src.api.dependencies at all -- e.g.
    # test_qdrant_vector_store.py -- don't pay for importing or constructing
    # it just for this fixture's sake.
    yield
    import sys

    deps = sys.modules.get("src.api.dependencies")
    if deps is not None:
        await deps._engine.dispose()


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


@pytest.fixture(scope="session")
def embedding_model() -> SentenceTransformersEmbedder:
    return SentenceTransformersEmbedder()


@pytest.fixture(scope="session")
def distilgpt2_tokenizer():
    return AutoTokenizer.from_pretrained(_DISTILGPT2_MODEL_ID)


@pytest.fixture(scope="session")
def distilgpt2_model():
    model = AutoModelForCausalLM.from_pretrained(_DISTILGPT2_MODEL_ID)
    model.eval()
    return model


@pytest.fixture(scope="session")
def qdrant_container():
    # Pinned explicitly, matching postgres_container/redis_container above and
    # the qdrant service in docker/docker-compose.yml. Left unpinned, this
    # fixture takes whatever image testcontainers currently defaults to, so a
    # routine dependency bump could silently move the integration suite off the
    # version compose runs -- which is exactly how the query_points-404-against
    # -an-old-server bug found in Task 15 would come back unnoticed.
    with QdrantContainer("qdrant/qdrant:v1.16.2") as container:
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


@pytest.fixture(scope="session")
def neo4j_container():
    # Pinned to the 5.x line, matching docker/docker-compose.yml's neo4j
    # service and OVERVIEW.md's "neo4j >=5.20.0" driver-version floor -- same
    # reasoning as qdrant_container's explicit pin above (an unpinned image
    # tag drifts to whatever testcontainers currently defaults to, silently
    # moving the integration suite off the version compose actually runs).
    with Neo4jContainer("neo4j:5-community") as container:
        yield container


@pytest.fixture(scope="session")
def neo4j_url(neo4j_container: Neo4jContainer) -> tuple[str, str, str]:
    # (bolt_url, username, password) -- 127.0.0.1 explicit, not
    # get_container_host_ip() (returns "localhost", the same Windows/IPv6
    # problem documented on redis_url/qdrant_url above).
    port = neo4j_container.get_exposed_port(neo4j_container.port)
    url = f"bolt://127.0.0.1:{port}"
    return url, neo4j_container.username, neo4j_container.password


@pytest.fixture(scope="session", autouse=True)
def _default_qdrant_url_env(qdrant_url: str):
    # src/api/dependencies.py (Task 14) reads os.environ["QDRANT_URL"] at
    # *import* time, module-level, mirroring the existing _engine/_sessionmaker
    # singleton pattern -- so importing src.api.main or src.api.dependencies
    # now unconditionally requires QDRANT_URL to already be set. Test files
    # from before Task 14 (test_app_dependencies.py, test_auth_endpoints.py)
    # were written when that requirement didn't exist and never set it
    # themselves. This autouse, session-scoped fixture (mirroring
    # run_migrations above) guarantees QDRANT_URL is populated before the
    # first test of the session runs, regardless of which test file that is
    # or whether it touches RAG functionality at all. setdefault, not a plain
    # assignment, so a test that deliberately sets a different QDRANT_URL
    # value in its own _client() helper still wins.
    os.environ.setdefault("QDRANT_URL", qdrant_url)
