import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


async def best_effort_graph_write(write: Coroutine[Any, Any, None], description: str) -> None:
    # Neo4j writes are a side effect of CaptureEpisode/RecordSemanticFact/
    # ConsolidateEpisodes, not their primary job -- per DATABASE.md's own
    # "three separate writes to three separate systems, not one atomic
    # transaction" model, a graph write failing must not roll back or block
    # the Postgres/Qdrant writes that already succeeded (this project has
    # no outbox/saga mechanism to reconcile a partial failure, and building
    # one is out of scope for MAG Batch D's two issues). Catches bare
    # Exception deliberately: a graph write can fail in many
    # driver-specific ways (connection loss, transient errors, constraint
    # violations from a schema drift), and enumerating every neo4j
    # exception type here would be exactly the kind of speculative
    # completeness this project's own conventions avoid -- what matters is
    # that NO failure from this call propagates past it.
    try:
        await write
    except Exception:
        logger.warning(
            "Memory graph write failed (%s); continuing without it.", description, exc_info=True
        )
