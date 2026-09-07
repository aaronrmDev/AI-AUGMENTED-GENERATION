import pytest

from src.cag.domain.batching_metrics import prefill_tokens_required, unbatched_prefill_tokens
from src.cag.domain.entities import BatchRequest
from src.cag.infrastructure.fifo_batch_scheduler import FIFOBatchScheduler
from src.cag.infrastructure.prefix_aware_batch_scheduler import PrefixAwareBatchScheduler

_BLOCK = 16


def _request(request_id: str, prefix: tuple[int, ...], unique: int) -> BatchRequest:
    return BatchRequest(request_id=request_id, tokens=prefix + (unique,))


def _tokens(batches: list[list[BatchRequest]]) -> list[list[tuple[int, ...]]]:
    return [[request.tokens for request in batch] for batch in batches]


def test_fifo_batches_purely_in_arrival_order():
    pending = [BatchRequest(f"r{i}", (i,)) for i in range(5)]
    batches = FIFOBatchScheduler().form_batches(pending, max_batch_size=2)
    assert [[r.request_id for r in batch] for batch in batches] == [
        ["r0", "r1"],
        ["r2", "r3"],
        ["r4"],
    ]


def test_prefix_aware_groups_requests_sharing_a_block_aligned_prefix():
    system_a = tuple(range(100, 100 + _BLOCK))
    system_b = tuple(range(900, 900 + _BLOCK))
    # Arrivals deliberately interleave the two prefixes.
    pending = [
        _request("a1", system_a, 1),
        _request("b1", system_b, 2),
        _request("a2", system_a, 3),
        _request("b2", system_b, 4),
    ]
    batches = PrefixAwareBatchScheduler(block_size=_BLOCK).form_batches(
        pending, max_batch_size=4
    )
    grouped = sorted(sorted(r.request_id for r in batch) for batch in batches)
    assert grouped == [["a1", "a2"], ["b1", "b2"]]


def test_prefix_aware_preserves_arrival_order_inside_a_group():
    system = tuple(range(100, 100 + _BLOCK))
    pending = [_request(f"r{i}", system, i) for i in range(4)]
    batches = PrefixAwareBatchScheduler(block_size=_BLOCK).form_batches(
        pending, max_batch_size=4
    )
    assert [r.request_id for r in batches[0]] == ["r0", "r1", "r2", "r3"]


def test_prefix_aware_respects_the_maximum_batch_size():
    system = tuple(range(100, 100 + _BLOCK))
    pending = [_request(f"r{i}", system, i) for i in range(5)]
    batches = PrefixAwareBatchScheduler(block_size=_BLOCK).form_batches(
        pending, max_batch_size=2
    )
    assert [len(batch) for batch in batches] == [2, 2, 1]


def test_requests_shorter_than_one_block_group_together_with_no_shared_prefix():
    pending = [BatchRequest(f"r{i}", (i, i + 1)) for i in range(3)]
    batches = PrefixAwareBatchScheduler(block_size=_BLOCK).form_batches(
        pending, max_batch_size=8
    )
    # Nothing shareable, so they land in one group rather than being
    # spuriously split by signatures that mean nothing.
    assert len(batches) == 1
    assert len(batches[0]) == 3


def test_prefix_aware_grouping_costs_less_prefill_than_fifo_on_interleaved_arrivals():
    # CAG.md's scheduling claim, made falsifiable: the same requests,
    # the same batch size, differing only in how they are grouped.
    system_a = tuple(range(100, 100 + 4 * _BLOCK))
    system_b = tuple(range(900, 900 + 4 * _BLOCK))
    pending: list[BatchRequest] = []
    for i in range(4):
        pending.append(_request(f"a{i}", system_a, i))
        pending.append(_request(f"b{i}", system_b, 1000 + i))

    fifo = FIFOBatchScheduler().form_batches(pending, max_batch_size=2)
    aware = PrefixAwareBatchScheduler(block_size=_BLOCK, grouping_blocks=4).form_batches(
        pending, max_batch_size=2
    )

    fifo_cost = prefill_tokens_required(_tokens(fifo), _BLOCK)
    aware_cost = prefill_tokens_required(_tokens(aware), _BLOCK)
    baseline = unbatched_prefill_tokens([r.tokens for r in pending])

    # FIFO pairs every a with a b, so no pair shares anything and it
    # saves nothing at all over prefilling each request independently.
    assert fifo_cost == baseline
    assert aware_cost < fifo_cost


def test_rejects_a_non_positive_batch_size():
    with pytest.raises(ValueError):
        FIFOBatchScheduler().form_batches([], max_batch_size=0)
    with pytest.raises(ValueError):
        PrefixAwareBatchScheduler(block_size=_BLOCK).form_batches([], max_batch_size=0)
