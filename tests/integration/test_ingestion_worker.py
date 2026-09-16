"""ingest_data_source_task against real Postgres and Qdrant, run synchronously via
Celery's own .apply() (no broker, no worker process) -- this proves the task's
composition root wires IngestDataSource correctly, independent of Task 6's
full HTTP-to-real-worker proof."""
import uuid

import pytest


@pytest.mark.asyncio
async def test_a_new_tenant_scoped_source_is_ingested_and_returns_its_route(
    app_database_url, qdrant_url, redis_url, neo4j_url, monkeypatch
):
    monkeypatch.setenv("APP_DATABASE_URL", app_database_url)
    monkeypatch.setenv("QDRANT_URL", qdrant_url)
    # ingestion_worker imports src.workers.celery_app, which reads REDIS_URL at
    # *import* time -- set before the deferred import below, same reasoning as
    # Task 3's dispatcher fixture and tests/integration/test_sessions_
    # endpoints.py's _client() helper for APP_DATABASE_URL/QDRANT_URL. A
    # module-top-level `from src.workers.ingestion_worker import
    # ingest_data_source_task` would run at collection time, before this test
    # function (or any fixture) ever executes, and crash with KeyError.
    monkeypatch.setenv("REDIS_URL", redis_url)
    url, username, password = neo4j_url
    monkeypatch.setenv("NEO4J_URL", url)
    monkeypatch.setenv("NEO4J_USERNAME", username)
    monkeypatch.setenv("NEO4J_PASSWORD", password)

    from src.workers.ingestion_worker import ingest_data_source_task

    tenant_id = uuid.uuid4()
    result = ingest_data_source_task.apply(
        kwargs={
            "tenant_id": str(tenant_id),
            "source_key": "return-policy",
            "scope": "tenant",
            "expected_change_interval_seconds": 90 * 24 * 60 * 60,
            "content": "Our return policy allows returns within 90 days.",
            "user_id": None,
        }
    ).get()

    assert result["changed"] is True
    assert result["route"] in ("rag_only", "cag_with_rag_backup")
    uuid.UUID(result["source_id"])  # does not raise
