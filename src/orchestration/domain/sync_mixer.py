import hashlib
import uuid

from src.orchestration.domain.entities import SyncConflict


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def reconcile(
    cached_content_hash: str | None, authoritative_content: str, document_id: uuid.UUID
) -> SyncConflict | None:
    """The sync mixer's tiebreak: whichever content the caller passes as
    `authoritative_content` wins. Shared, unmodified, by all three
    cross-paradigm Sync Mixer mechanisms -- SyncCycle (RAG-vs-CAG: RAG
    wins), MagSyncCycle (RAG-vs-MAG: RAG wins), and CagMagSyncCycle
    (CAG-vs-MAG: MAG wins, since CAG's hot-tier entry is a cached copy of
    a specific MAG record in that pairing, not an independent paradigm
    holding a competing fact -- see CagMagSyncCycle's own docstring for
    the full reasoning). This function itself is deliberately paradigm-
    agnostic and doesn't know or enforce which paradigm "wins" in any
    particular case; the comparison only needs a hash to compare against,
    and each caller decides for itself what "authoritative" means for its
    own pairing before calling this.

    Takes a plain content hash rather than a paradigm-specific cache-hit
    entity (this function originally took CacheHit | None; generalized
    here so RAG+MAG's MagSyncCycle can reuse it too, rather than
    duplicating this same comparison for a WarmEntry instead of a
    CacheHit -- the comparison itself never depended on anything CAG-
    specific, only on having a hash to compare). Pure comparison -- no
    I/O, no eviction. The caller owns whichever fast-store port it's
    reconciling against and is responsible for actually evicting/demoting
    on a real conflict, the same domain/application split every other
    real side effect in this codebase already follows.
    """
    if cached_content_hash is None:
        return None
    authoritative_hash = content_hash(authoritative_content)
    if cached_content_hash == authoritative_hash:
        return None
    return SyncConflict(
        document_id=document_id,
        cached_content_hash=cached_content_hash,
        authoritative_content_hash=authoritative_hash,
    )
