from enum import Enum


class CAGTechnique(Enum):
    # The 4 techniques ADR-0003 marks incompatible with alternative
    # attention -- each depends on manipulating a per-token KV cache that
    # alternative-attention architectures (Linear Attention, Mamba, and
    # similar) don't produce.
    KV_CACHE_EVICTION = "kv_cache_eviction"
    KV_CACHE_COMPRESSION = "kv_cache_compression"
    PAGED_ATTENTION = "paged_attention"
    SPECULATIVE_DECODING = "speculative_decoding"
    # The 4 techniques ADR-0003 marks compatible -- each operates above
    # the attention mechanism itself, at the scheduling, session, or
    # storage-tiering level, rather than reaching into per-token cache
    # internals.
    PREFIX_CACHING = "prefix_caching"
    HYBRID_OFFLOADING = "hybrid_offloading"
    MULTI_TURN_CACHING = "multi_turn_caching"
    CACHE_AWARE_BATCHING = "cache_aware_batching"


# ADR-0003's exact 4-incompatible/4-compatible split, restated as data --
# see docs/decisions/adr/0003-standard-attention-cache-optimization.md's
# Decision section for the reasoning behind each entry.
_COMPATIBILITY: dict[CAGTechnique, bool] = {
    CAGTechnique.KV_CACHE_EVICTION: False,
    CAGTechnique.KV_CACHE_COMPRESSION: False,
    CAGTechnique.PAGED_ATTENTION: False,
    CAGTechnique.SPECULATIVE_DECODING: False,
    CAGTechnique.PREFIX_CACHING: True,
    CAGTechnique.HYBRID_OFFLOADING: True,
    CAGTechnique.MULTI_TURN_CACHING: True,
    CAGTechnique.CACHE_AWARE_BATCHING: True,
}


def is_compatible_with_alternative_attention(technique: CAGTechnique) -> bool:
    return _COMPATIBILITY[technique]
