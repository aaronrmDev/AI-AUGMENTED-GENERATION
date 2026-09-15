import pytest

from src.cag.domain.batching_metrics import (
    longest_common_block_prefix,
    prefill_tokens_required,
    unbatched_prefill_tokens,
)


def test_common_prefix_rounds_down_to_whole_blocks():
    # 20 tokens agree, but with 16-token blocks only the first whole
    # block is actually shareable.
    a = tuple(range(20)) + (99,)
    b = tuple(range(20)) + (77,)
    assert longest_common_block_prefix([a, b], block_size=16) == 16


def test_common_prefix_is_zero_when_agreement_is_shorter_than_one_block():
    a = (1, 2, 3, 50)
    b = (1, 2, 3, 60)
    assert longest_common_block_prefix([a, b], block_size=16) == 0


def test_common_prefix_of_identical_sequences_is_every_whole_block():
    a = tuple(range(32))
    assert longest_common_block_prefix([a, a], block_size=16) == 32


def test_common_prefix_of_an_empty_batch_is_zero():
    assert longest_common_block_prefix([], block_size=16) == 0


def test_common_prefix_rejects_a_non_positive_block_size():
    with pytest.raises(ValueError):
        longest_common_block_prefix([(1, 2)], block_size=0)


def test_prefill_cost_charges_a_shared_prefix_once_per_batch():
    shared = tuple(range(32))
    a = shared + (100, 101)
    b = shared + (200, 201)
    # Shared 32 counted once, plus 2 unique tokens each.
    assert prefill_tokens_required([[a, b]], block_size=16) == 36
    # Without any sharing both sequences cost their full length.
    assert unbatched_prefill_tokens([a, b]) == 68


def test_splitting_a_shared_prefix_across_batches_pays_for_it_twice():
    shared = tuple(range(32))
    a = shared + (100,)
    b = shared + (200,)
    together = prefill_tokens_required([[a, b]], block_size=16)
    apart = prefill_tokens_required([[a], [b]], block_size=16)
    assert apart > together
    assert apart - together == 32


def test_prefill_cost_ignores_empty_batches():
    a = tuple(range(16))
    assert prefill_tokens_required([[a], []], block_size=16) == 16
