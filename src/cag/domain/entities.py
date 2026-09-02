from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CompressedKV:
    # payload is method-specific (quantized ints + scales for KIVI/KVQuant,
    # a latent matrix + reconstruction matrix for PALU, an interpolated
    # representation for MiniCache, a composite for ShadowKV) -- a single
    # shared dataclass shape across five genuinely different compressed
    # representations would force an artificial uniformity none of them
    # actually share, the same reasoning that keeps MiniCache off the
    # single-tensor KVCacheCompressor port entirely.
    method: str
    payload: dict[str, Any] = field(default_factory=dict)
    original_shape: tuple[int, int] = (0, 0)
