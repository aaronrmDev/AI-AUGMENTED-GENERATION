from src.cag.domain.attention_compatibility import (
    CAGTechnique,
    is_compatible_with_alternative_attention,
)


def test_kv_cache_eviction_is_incompatible():
    assert is_compatible_with_alternative_attention(CAGTechnique.KV_CACHE_EVICTION) is False


def test_kv_cache_compression_is_incompatible():
    assert is_compatible_with_alternative_attention(CAGTechnique.KV_CACHE_COMPRESSION) is False


def test_paged_attention_is_incompatible():
    assert is_compatible_with_alternative_attention(CAGTechnique.PAGED_ATTENTION) is False


def test_speculative_decoding_is_incompatible():
    assert is_compatible_with_alternative_attention(CAGTechnique.SPECULATIVE_DECODING) is False


def test_prefix_caching_is_compatible():
    assert is_compatible_with_alternative_attention(CAGTechnique.PREFIX_CACHING) is True


def test_hybrid_offloading_is_compatible():
    assert is_compatible_with_alternative_attention(CAGTechnique.HYBRID_OFFLOADING) is True


def test_multi_turn_caching_is_compatible():
    assert is_compatible_with_alternative_attention(CAGTechnique.MULTI_TURN_CACHING) is True


def test_cache_aware_batching_is_compatible():
    assert is_compatible_with_alternative_attention(CAGTechnique.CACHE_AWARE_BATCHING) is True


def test_the_split_is_exactly_four_incompatible_and_four_compatible():
    # ADR-0003's own framing: "incompatible with four of the eight CAG
    # techniques... compatible with the remaining four" -- a literal,
    # checked invariant, not just eight individually-correct answers that
    # could still add up to the wrong split (e.g. 3/5) if one of the
    # per-technique tests above were itself wrong about which bucket a
    # technique belongs in.
    results = [is_compatible_with_alternative_attention(t) for t in CAGTechnique]
    assert results.count(True) == 4
    assert results.count(False) == 4


def test_every_technique_has_a_defined_compatibility():
    # Guards against a future technique added to CAGTechnique without its
    # compatibility being decided -- is_compatible_with_alternative_attention
    # must return a real bool for every member, not raise or silently
    # default.
    for technique in CAGTechnique:
        assert isinstance(is_compatible_with_alternative_attention(technique), bool)
