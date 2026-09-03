import hashlib
import uuid

from src.orchestration.domain.entities import CacheHit, SyncConflict


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def reconcile(
    cached_hit: CacheHit | None, authoritative_content: str, document_id: uuid.UUID
) -> SyncConflict | None:
    """The sync mixer's RAG-vs-CAG tiebreak: RAG, as the external source, wins.

    Pure comparison -- no I/O, no eviction. The caller owns the FrozenCache
    port and is responsible for actually evicting on a real conflict, the
    same domain/application split every other real side effect in this
    codebase already follows.
    """
    if cached_hit is None:
        return None
    authoritative_hash = content_hash(authoritative_content)
    if cached_hit.content_hash == authoritative_hash:
        return None
    return SyncConflict(
        document_id=document_id,
        cached_content_hash=cached_hit.content_hash,
        authoritative_content_hash=authoritative_hash,
    )
