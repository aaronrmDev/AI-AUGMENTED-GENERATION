import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure._result_fusion import reciprocal_rank_fusion


def _result(chunk_id: uuid.UUID, score: float) -> SearchResult:
    return SearchResult(document_id=uuid.uuid4(), chunk_id=chunk_id, content=f"chunk {chunk_id}", score=score)


def test_a_chunk_ranked_second_in_both_lists_beats_a_chunk_ranked_first_in_only_one():
    shared_id = uuid.uuid4()
    vector_only_id = uuid.uuid4()
    keyword_only_id = uuid.uuid4()
    vector = [_result(vector_only_id, 0.9), _result(shared_id, 0.5)]
    keyword = [_result(keyword_only_id, 12.0), _result(shared_id, 3.0)]

    results = reciprocal_rank_fusion([vector, keyword], top_k=3)

    assert results[0].chunk_id == shared_id


def test_a_chunk_found_in_only_one_list_still_appears():
    only_id = uuid.uuid4()
    results = reciprocal_rank_fusion([[_result(only_id, 0.9)], [_result(uuid.uuid4(), 5.0)]], top_k=5)
    assert any(r.chunk_id == only_id for r in results)


def test_respects_top_k():
    lists = [[_result(uuid.uuid4(), 1.0) for _ in range(5)] for _ in range(2)]
    results = reciprocal_rank_fusion(lists, top_k=3)
    assert len(results) == 3


def test_works_with_more_than_two_lists():
    shared_id = uuid.uuid4()
    lists = [[_result(shared_id, 1.0), _result(uuid.uuid4(), 0.5)] for _ in range(4)]
    results = reciprocal_rank_fusion(lists, top_k=1)
    # shared_id is rank 0 in all 4 lists -- must win outright
    assert results[0].chunk_id == shared_id
