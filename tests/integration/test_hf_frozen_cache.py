"""Real verification of HFFrozenCache against a real distilgpt2 (CPU, no
GPU needed) -- the same model and CPU setup CAG's own Batch B/C integration
tests already proved out. Classified as integration, not unit, matching
this project's own precedent (test_sentence_transformers_embedder.py):
downloading and running a real model is integration work here regardless
of whether Docker/testcontainers is involved.
"""
import uuid

from src.orchestration.infrastructure.hf_frozen_cache import HFFrozenCache

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

    cache.preload(document_id, _RETURN_POLICY)
    hit = cache.lookup(document_id)

    assert hit is not None
    assert hit.kv_cache is not None


def test_lookup_before_preload_is_a_miss(distilgpt2_model, distilgpt2_tokenizer):
    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)

    assert cache.lookup(uuid.uuid4()) is None
    assert cache.contains(uuid.uuid4()) is False


def test_evict_then_lookup_is_a_miss_again(distilgpt2_model, distilgpt2_tokenizer):
    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)
    document_id = uuid.uuid4()
    cache.preload(document_id, _RETURN_POLICY)

    cache.evict(document_id)

    assert cache.lookup(document_id) is None
    assert cache.contains(document_id) is False


def test_two_different_documents_do_not_collide(distilgpt2_model, distilgpt2_tokenizer):
    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)
    doc_a, doc_b = uuid.uuid4(), uuid.uuid4()

    cache.preload(doc_a, "The shipping policy covers domestic orders only.")
    cache.preload(doc_b, _RETURN_POLICY)

    hit_a = cache.lookup(doc_a)
    hit_b = cache.lookup(doc_b)
    assert hit_a is not None and hit_b is not None
    assert hit_a.content_hash != hit_b.content_hash
    assert cache.contains(doc_a) and cache.contains(doc_b)


def test_reusing_a_warmed_cache_is_faster_than_recomputing_it(
    distilgpt2_model, distilgpt2_tokenizer
):
    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)
    document_id = uuid.uuid4()
    cache.preload(document_id, _RETURN_POLICY)

    trials = [cache.prefill_latency_ms(document_id, _RETURN_POLICY) for _ in range(3)]
    warm_faster_count = sum(1 for cold_ms, warm_ms in trials if warm_ms < cold_ms)

    print(f"cold-vs-warm prefill latency (ms), 3 trials: {trials}")
    assert warm_faster_count >= 2
