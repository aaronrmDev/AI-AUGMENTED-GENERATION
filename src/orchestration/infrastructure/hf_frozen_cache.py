import copy
import time
import uuid
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.orchestration.domain.entities import CacheHit
from src.orchestration.domain.ports import FrozenCache
from src.orchestration.domain.sync_mixer import content_hash

_DEFAULT_MODEL_ID = "distilgpt2"
_Key = tuple[uuid.UUID, uuid.UUID]  # (tenant_id, document_id)


class HFFrozenCache(FrozenCache):
    """A real, CPU-sized proxy for CAG's GPU-resident frozen KV cache.

    Loads its own real transformers causal LM by default, the same
    self-contained shape as SentenceTransformersEmbedder rather than
    HFTargetModel's constructor-injected model -- there's no fine indexing
    arithmetic here that benefits from a fast synthetic fake, only real
    cache-hit/miss and latency-direction behavior, so this class always
    talks to a real model. `tokenizer`/`model` can be passed in TOGETHER to
    reuse an already-loaded pair (this project's own session-scoped
    distilgpt2_tokenizer/distilgpt2_model fixtures) across many test
    functions instead of reloading from disk each time -- purely a speed
    optimization for callers that already have one loaded, not a fakery
    seam; production code just calls `HFFrozenCache()`. Passing only one of
    the two raises immediately (a review finding caught the first version
    silently pairing a caller-supplied component with a freshly-loaded
    default for the other, which could pair a tokenizer and a model from
    different vocabularies with no error until a downstream, hard-to-trace
    index/embedding failure).

    preload runs one real forward pass over a document's content and
    stores the resulting DynamicCache; lookup returns it so a caller can
    skip recomputing that prefix. This demonstrates the real mechanism
    behind CLAUDE.md's "near-zero TTFT on a hit" claim at a scale this
    project can actually run on CPU -- it is NOT a claim about production
    GPU VRAM capacity or eviction-under-concurrent-load behavior, which
    stay deferred to the GPU-dependent Prefix Caching CAG story per
    docs/testing/TESTING.md.

    tenant_id-scoped on every method, matching FrozenCache's own port
    contract -- a review finding caught the first version of this class
    keying its entries by document_id alone, which would have let one
    tenant's warmed content be looked up (and served) under another
    tenant's document_id if the two ever collided or were queried through
    a shared instance without a tenant filter.
    """

    def __init__(
        self, model_id: str = _DEFAULT_MODEL_ID, *, tokenizer: Any = None, model: Any = None
    ) -> None:
        if (tokenizer is None) != (model is None):
            raise ValueError(
                "tokenizer and model must be supplied together or not at all -- "
                "pairing a caller-supplied one with a freshly-loaded default risks "
                "a silent vocabulary mismatch"
            )
        self._tokenizer = (
            tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_id)
        )
        self._model = (
            model if model is not None else AutoModelForCausalLM.from_pretrained(model_id)
        )
        self._model.eval()
        self._entries: dict[_Key, tuple[str, Any]] = {}

    def preload(self, tenant_id: uuid.UUID, document_id: uuid.UUID, content: str) -> None:
        input_ids = self._tokenizer(content, return_tensors="pt").input_ids
        with torch.no_grad():
            output = self._model(input_ids, use_cache=True)
        self._entries[(tenant_id, document_id)] = (content_hash(content), output.past_key_values)

    def lookup(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> CacheHit | None:
        entry = self._entries.get((tenant_id, document_id))
        if entry is None:
            return None
        hash_, cache = entry
        return CacheHit(content_hash=hash_, kv_cache=cache)

    def evict(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> None:
        self._entries.pop((tenant_id, document_id), None)

    def contains(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        return (tenant_id, document_id) in self._entries

    def prefill_latency_ms(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, content: str
    ) -> tuple[float, float]:
        """Real wall-clock cold-vs-warm prefill timing for one document.

        cold_ms: time to run the full forward pass over `content` from
        scratch. warm_ms: time to reuse the DynamicCache already stored for
        document_id (deep-copied first, matching CAG Batch B's own
        established precedent for not letting cache mutation contaminate a
        later call) by feeding only the final token through against it --
        the real difference between reprocessing an entire prefix and
        reusing an already-computed one. Requires document_id to already
        be preloaded.
        """
        cached_hit = self.lookup(tenant_id, document_id)
        if cached_hit is None:
            raise ValueError(f"{document_id} is not preloaded -- nothing to reuse")

        input_ids = self._tokenizer(content, return_tensors="pt").input_ids

        cold_start = time.perf_counter()
        with torch.no_grad():
            self._model(input_ids, use_cache=True)
        cold_ms = (time.perf_counter() - cold_start) * 1000

        isolated_cache = copy.deepcopy(cached_hit.kv_cache)
        last_token = input_ids[:, -1:]
        warm_start = time.perf_counter()
        with torch.no_grad():
            self._model(last_token, past_key_values=isolated_cache, use_cache=True)
        warm_ms = (time.perf_counter() - warm_start) * 1000

        return cold_ms, warm_ms
