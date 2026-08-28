import json
import uuid
from datetime import UTC, datetime
from typing import Any

from src.mag.domain.entities import EpisodicMemory
from src.mag.domain.ports import EpisodicMemoryRepository
from src.mag.infrastructure.qdrant_episodic_memory_index import QdrantEpisodicMemoryIndex
from src.rag.domain.ports import EmbeddingModel


class CaptureEpisode:
    # episodic_memory_index is typed against the concrete
    # QdrantEpisodicMemoryIndex, not a src.mag.domain.ports abstraction --
    # ports.py (this batch's already-written, shared contract) defines only
    # EpisodicMemoryRepository/SemanticMemoryRepository/WorkingMemoryStore,
    # with no fast-lookup-index port. The design spec treats this index as a
    # second concrete write target for save(), matching how UploadDocument
    # (RAG) writes to both DocumentRepository and VectorStore, not as a port
    # this batch is scoped to introduce.
    def __init__(
        self,
        episodic_memory_repository: EpisodicMemoryRepository,
        episodic_memory_index: QdrantEpisodicMemoryIndex,
        embedding_model: EmbeddingModel,
    ) -> None:
        self._episodes = episodic_memory_repository
        self._index = episodic_memory_index
        self._embedder = embedding_model

    async def execute(
        self, tenant_id: uuid.UUID, session_id: uuid.UUID, content: dict[str, Any]
    ) -> EpisodicMemory:
        # Embeds the whole event (json.dumps, keys sorted for a stable
        # embedding regardless of dict insertion order) rather than one
        # designated field -- content's shape (input/reasoning/tool_calls/
        # output/actors/entities) has no single field guaranteed to carry the
        # episode's meaning, and search_by_similarity has to be able to match
        # on any of them.
        embedding = self._embedder.embed(json.dumps(content, sort_keys=True))
        episode = EpisodicMemory(
            id=uuid.uuid4(),
            session_id=session_id,
            content=content,
            embedding=embedding,
            timestamp=datetime.now(UTC),
        )
        await self._episodes.save(episode, tenant_id)
        await self._index.upsert(episode, tenant_id)
        return episode
