from typing import Any

import numpy as np

from src.cag.domain.entities import CompressedKV
from src.cag.domain.ports import KVCacheCompressor

_METHOD = "palu"


class PALUCompressor(KVCacheCompressor):
    # PALU's real technique (CAG.md) SVD-decomposes each KV *projection
    # weight matrix* W offline into A and B, caching only the compact
    # latent H = X @ A per forward pass and fusing B for cheap reuse
    # across every token that passes through that layer. This port
    # operates on the KV tensor itself (rows = tokens, columns =
    # channels), not on the upstream W this port never sees -- so,
    # disclosed simplification per the design spec's "does not attempt
    # bit-exact reproduction of any paper's setup", the same low-rank
    # mechanic is applied directly to the cached tensor: kv is SVD-
    # truncated and stored as its own compact factors H and B, rather
    # than as a projection matrix's factors.
    def __init__(self, rank: int = 4) -> None:
        if rank < 1:
            raise ValueError("rank must be at least 1")
        self._rank = rank

    def compress(self, kv: list[list[float]]) -> CompressedKV:
        if not kv:
            raise ValueError("kv must be non-empty")
        matrix = np.array(kv, dtype=np.float64)
        num_tokens, num_channels = matrix.shape
        # SVD can never produce more singular values/vectors than
        # min(rows, cols) -- a requested rank past that ceiling is
        # clamped to full rank rather than left to silently truncate to
        # whatever numpy happens to return.
        effective_rank = min(self._rank, num_tokens, num_channels)
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        u_k = u[:, :effective_rank]
        s_k = s[:effective_rank]
        v_k_t = vh[:effective_rank, :]
        # H = U_k @ diag(S_k): broadcasting u_k's columns against s_k is
        # the same result as multiplying by a diagonal matrix, cheaper.
        h = u_k * s_k
        b = v_k_t

        payload: dict[str, Any] = {
            "H": h.tolist(),
            "B": b.tolist(),
            "rank": effective_rank,
        }
        return CompressedKV(
            method=_METHOD, payload=payload, original_shape=(num_tokens, num_channels)
        )

    def decompress(self, compressed: CompressedKV) -> list[list[float]]:
        if compressed.method != _METHOD:
            raise ValueError(f"expected a '{_METHOD}' payload, got '{compressed.method}'")
        h = np.array(compressed.payload["H"], dtype=np.float64)
        b = np.array(compressed.payload["B"], dtype=np.float64)
        reconstructed: list[list[float]] = (h @ b).tolist()
        return reconstructed
