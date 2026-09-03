import uuid

from src.rag.domain.entities import SearchResult

_DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]], top_k: int, k_rrf: int = _DEFAULT_RRF_K
) -> list[SearchResult]:
    rrf_scores: dict[uuid.UUID, float] = {}
    by_id: dict[uuid.UUID, SearchResult] = {}
    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            rrf_scores[result.chunk_id] = rrf_scores.get(
                result.chunk_id, 0.0
            ) + 1.0 / (k_rrf + rank + 1)
            by_id[result.chunk_id] = result

    merged_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    return [
        SearchResult(
            document_id=by_id[cid].document_id, chunk_id=cid,
            content=by_id[cid].content, score=rrf_scores[cid],
        )
        for cid in merged_ids[:top_k]
    ]
