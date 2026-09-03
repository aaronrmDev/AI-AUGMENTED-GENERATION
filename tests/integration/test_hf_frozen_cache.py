"""Real verification of HFFrozenCache against a real distilgpt2 (CPU, no
GPU needed) -- the same model and CPU setup CAG's own Batch B/C integration
tests already proved out. Classified as integration, not unit, matching
this project's own precedent (test_sentence_transformers_embedder.py):
downloading and running a real model is integration work here regardless
of whether Docker/testcontainers is involved.
"""
import statistics
import uuid

import pytest

from src.orchestration.infrastructure.hf_frozen_cache import HFFrozenCache

_TENANT = uuid.uuid4()
_RETURN_POLICY = (
    "Our return policy allows customers to return unopened items within "
    "thirty days of purchase for a full refund. Opened items may be "
    "exchanged within fourteen days. Refunds are issued to the original "
    "payment method and typically process within five to seven business "
    "days after we receive the returned item at our warehouse."
)


def test_preload_then_lookup_returns_a_hit_with_the_correct_content_hash(
    distilgpt2_model, distilgpt2_tokenizer
):
    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)
    document_id = uuid.uuid4()

    cache.preload(_TENANT, document_id, _RETURN_POLICY)
    hit = cache.lookup(_TENANT, document_id)

    assert hit is not None
    assert hit.kv_cache is not None


def test_lookup_before_preload_is_a_miss(distilgpt2_model, distilgpt2_tokenizer):
    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)

    assert cache.lookup(_TENANT, uuid.uuid4()) is None
    assert cache.contains(_TENANT, uuid.uuid4()) is False


def test_evict_then_lookup_is_a_miss_again(distilgpt2_model, distilgpt2_tokenizer):
    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)
    document_id = uuid.uuid4()
    cache.preload(_TENANT, document_id, _RETURN_POLICY)

    cache.evict(_TENANT, document_id)

    assert cache.lookup(_TENANT, document_id) is None
    assert cache.contains(_TENANT, document_id) is False


def test_two_different_documents_do_not_collide(distilgpt2_model, distilgpt2_tokenizer):
    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)
    doc_a, doc_b = uuid.uuid4(), uuid.uuid4()

    cache.preload(_TENANT, doc_a, "The shipping policy covers domestic orders only.")
    cache.preload(_TENANT, doc_b, _RETURN_POLICY)

    hit_a = cache.lookup(_TENANT, doc_a)
    hit_b = cache.lookup(_TENANT, doc_b)
    assert hit_a is not None and hit_b is not None
    assert hit_a.content_hash != hit_b.content_hash
    assert cache.contains(_TENANT, doc_a) and cache.contains(_TENANT, doc_b)


def test_two_tenants_warming_the_same_document_id_do_not_collide(
    distilgpt2_model, distilgpt2_tokenizer
):
    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    document_id = uuid.uuid4()  # same document_id, two different tenants
    other_content = "The shipping policy covers domestic orders only."

    cache.preload(tenant_a, document_id, _RETURN_POLICY)
    cache.preload(tenant_b, document_id, other_content)

    hit_a = cache.lookup(tenant_a, document_id)
    hit_b = cache.lookup(tenant_b, document_id)
    assert hit_a is not None and hit_b is not None
    assert hit_a.content_hash != hit_b.content_hash

    cache.evict(tenant_a, document_id)
    assert cache.lookup(tenant_a, document_id) is None
    assert cache.lookup(tenant_b, document_id) is not None  # untouched


def test_reusing_a_warmed_cache_is_faster_than_recomputing_it(
    distilgpt2_model, distilgpt2_tokenizer
):
    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)
    document_id = uuid.uuid4()
    cache.preload(_TENANT, document_id, _RETURN_POLICY)

    trials = [
        cache.prefill_latency_ms(_TENANT, document_id, _RETURN_POLICY) for _ in range(3)
    ]
    cold_ms = [cold for cold, _ in trials]
    warm_ms = [warm for _, warm in trials]
    median_cold, median_warm = statistics.median(cold_ms), statistics.median(warm_ms)

    print(f"cold-vs-warm prefill latency (ms), 3 trials: {trials}")
    print(f"median cold: {median_cold:.1f}ms, median warm: {median_warm:.1f}ms")
    # A sign-only check (warm faster than cold) degrades to close to a
    # coin flip under a regression that closes the real timing gap without
    # reversing it (e.g. accidentally feeding the full sequence through
    # the "warm" path too) -- a review finding confirmed this empirically.
    # The real, reliably-reproducible effect measured across this batch's
    # own runs is a consistent 3-4x margin, so 1.5x leaves ample headroom
    # without courting environment-driven flakiness.
    assert median_cold > 1.5 * median_warm


def test_tokenizer_and_model_must_be_supplied_together_or_not_at_all(
    distilgpt2_model, distilgpt2_tokenizer
):
    with pytest.raises(ValueError, match="together"):
        HFFrozenCache(tokenizer=distilgpt2_tokenizer)
    with pytest.raises(ValueError, match="together"):
        HFFrozenCache(model=distilgpt2_model)
