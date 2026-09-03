import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class CacheHit:
    # kv_cache is an opaque handle -- only the FrozenCache implementation
    # that produced it knows its real shape (a transformers DynamicCache in
    # this batch's one implementation). The domain layer never inspects it,
    # only passes it through, the same "opaque payload, typed at the
    # infrastructure boundary" shape CAG Batch B's CompressedKV established.
    content_hash: str
    kv_cache: Any


@dataclass(frozen=True)
class SyncConflict:
    # A real, detected disagreement between what the hot tier's cached
    # entry for document_id holds and what the paradigm-specific
    # authoritative source currently says -- RAG for SyncCycle and
    # MagSyncCycle, MAG for CagMagSyncCycle (see domain/sync_mixer.py,
    # which doesn't itself know or care which side is authoritative).
    # Always produced already-resolved (by eviction), never a pending
    # conflict a caller still has to act on.
    document_id: uuid.UUID
    cached_content_hash: str
    authoritative_content_hash: str


class TierDecision(Enum):
    PROMOTED = "promoted"
    DEMOTED = "demoted"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class WarmEntry:
    # MAG's warm-tier analogue of CacheHit. Unlike CacheHit.kv_cache (an
    # opaque tensor handle only infrastructure understands), `content` is a
    # real string here: MAG's warm tier is a semantic fact, not a KV
    # cache, so there's no opaque payload to hide -- the domain layer can
    # see and use the real text (State-Aware RAG's ranking boost and
    # query enrichment both need to read it).
    content_hash: str
    content: str
