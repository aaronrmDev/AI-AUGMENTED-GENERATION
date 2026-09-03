import hashlib
import uuid

from src.orchestration.domain.entities import CacheHit
from src.orchestration.domain.ports import FrozenCache
from src.orchestration.domain.sync_mixer import content_hash
from src.rag.domain.ports import EmbeddingModel


class FakeFrozenCache(FrozenCache):
    def __init__(self) -> None:
        self._entries: dict[tuple[uuid.UUID, uuid.UUID], str] = {}
        self.preload_calls: list[tuple[uuid.UUID, uuid.UUID, str]] = []
        self.evict_calls: list[tuple[uuid.UUID, uuid.UUID]] = []

    def preload(self, tenant_id: uuid.UUID, document_id: uuid.UUID, content: str) -> None:
        self.preload_calls.append((tenant_id, document_id, content))
        self._entries[(tenant_id, document_id)] = content

    def lookup(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> CacheHit | None:
        content = self._entries.get((tenant_id, document_id))
        if content is None:
            return None
        return CacheHit(content_hash=content_hash(content), kv_cache=None)

    def evict(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> None:
        self.evict_calls.append((tenant_id, document_id))
        self._entries.pop((tenant_id, document_id), None)

    def contains(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        return (tenant_id, document_id) in self._entries


class FakeBagOfWordsEmbeddingModel(EmbeddingModel):
    """Deterministic, cheap stand-in that actually varies by direction
    instead of collapsing to one (unlike rag_fakes.FakeEmbeddingModel,
    whose length-derived vectors are always parallel and would make every
    cosine similarity comparison trivially 1.0) -- needed here because
    CacheWarmedRetrieve's hit/miss decision genuinely depends on real
    directional similarity between two embeddings, not just their
    presence. A plain hashed bag-of-words vector, normalized: shared words
    push cosine similarity up, disjoint words push it down, deterministic
    across runs (hashlib, not Python's randomized built-in hash()).
    """

    _DIM = 64

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._DIM
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._DIM
            vector[index] += 1.0
        norm = sum(v * v for v in vector) ** 0.5
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]
