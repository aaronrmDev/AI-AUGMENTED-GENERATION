import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


async def best_effort_index_write(write: Coroutine[Any, Any, None], description: str) -> None:
    # Same non-blocking rationale as _graph_write_safety.py's
    # best_effort_graph_write, applied to a Qdrant STATUS SYNC
    # (InvalidateMemory/ArchiveMemory's set_valid_until/set_archived_at)
    # rather than a Neo4j write. If a fact's Qdrant point is missing --
    # e.g. an earlier RecordSemanticFact call's Qdrant upsert already
    # failed, a scenario DATABASE.md explicitly anticipates as a
    # consequence of writing to three separate stores with no shared
    # transaction -- set_payload raises rather than silently no-opping
    # (confirmed against the actually-installed qdrant-client: a missing
    # point raises UnexpectedResponse 404, not a quiet skip). By the time
    # this call runs, the Postgres UPDATE it wraps around has already
    # committed, so letting an uncaught exception here abort the whole
    # command would misrepresent a partially-successful, recoverable
    # state as a hard failure. Deliberately NOT applied to
    # RecordSemanticFact's own Qdrant upsert -- that call IS the primary,
    # required write for a fact's embedding; there is no analogous
    # "already succeeded elsewhere" state to fall back on if it fails.
    try:
        await write
    except Exception:
        logger.warning(
            "Qdrant status sync failed (%s); continuing without it.", description, exc_info=True
        )
