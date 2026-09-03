import re
import uuid

from rank_bm25 import BM25Plus

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import DocumentRepository, Retriever

_TOKEN = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25KeywordSearch(Retriever):
    def __init__(self, document_repository: DocumentRepository) -> None:
        self._documents = document_repository

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        chunks = await self._documents.get_chunks_for_tenant(tenant_id)
        if not chunks:
            return []

        corpus = [_tokenize(chunk.content) for chunk in chunks]
        # BM25Plus, not the plain BM25Okapi the plan's snippet named: Okapi's IDF term is
        # log((N - n + 0.5) / (n + 0.5)), which is exactly 0 whenever a query term appears in
        # precisely half of a 2-document corpus (n=1, N=2) -- the corpus shape this class's own
        # first unit test uses. That collapses every candidate's score to 0.0 and the "more
        # relevant" chunk stops winning the sort on ties. BM25+ adds a fixed delta per matched
        # term specifically to keep scores strictly positive and rank-differentiating in exactly
        # this small-corpus/common-term regime, without changing the tokenization, corpus
        # construction, or ranking/truncation logic below.
        bm25 = BM25Plus(corpus)
        scores = bm25.get_scores(_tokenize(query))

        ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            SearchResult(
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                content=chunk.content,
                score=float(score),
            )
            for chunk, score in ranked[:top_k]
        ]
