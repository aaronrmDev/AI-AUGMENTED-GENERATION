import os

from celery import Celery

# Both the Celery result backend and the ownership record CeleryIngestionJobDispatcher
# writes to Redis expire on this same schedule, so neither ever outlives the other.
JOB_TTL_SECONDS = 24 * 60 * 60


def build_celery_app(redis_url: str) -> Celery:
    app = Celery(
        "unified_ai_workers",
        broker=redis_url,
        backend=redis_url,
        # The worker process's own entry point imports this module by name
        # (`celery -A src.workers.celery_app:celery_app worker`) and needs the
        # task registered; the API process, which only ever calls send_task()
        # and AsyncResult() by name, never imports ingestion_worker at all.
        # Keeping that import here rather than at this module's top level is
        # what keeps Postgres/Qdrant/Neo4j infrastructure imports out of the
        # API process's import graph entirely.
        include=["src.workers.ingestion_worker"],
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        result_expires=JOB_TTL_SECONDS,
        task_track_started=True,
    )
    return app


# The module-level instance both the API process and the `celery` CLI import.
# REDIS_URL is read once, at import time, matching the existing
# _engine/_sessionmaker singleton pattern in src/api/dependencies.py.
celery_app = build_celery_app(os.environ["REDIS_URL"])
