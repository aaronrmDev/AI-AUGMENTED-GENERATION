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
    # A real, detected disagreement between what CAG's cache holds for
    # document_id and what RAG's index currently says -- always produced
    # already-resolved (by eviction; see domain/sync_mixer.py), never a
    # pending conflict a caller still has to act on.
    document_id: uuid.UUID
    cached_content_hash: str
    authoritative_content_hash: str


class TierDecision(Enum):
    PROMOTED = "promoted"
    DEMOTED = "demoted"
    UNCHANGED = "unchanged"
