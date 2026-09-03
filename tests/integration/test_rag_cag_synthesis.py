"""Real, live-measured validation of RAG+CAG synthesis -- the Combined
Capability Validation Story (#10/#27) plus the real numbers each of the
three mechanism Stories' (#2/#18, #3/#21, #8/#25) Definition of Done asks
for. Real Qdrant (testcontainers), real sentence-transformers embeddings,
real distilgpt2 (CPU) for the frozen cache -- no GPU or vLLM anywhere.

Each test mints its own fresh tenant_id and calls _seed_corpus with it,
rather than sharing one module-level tenant across the file -- a review
finding caught the first version of this file reusing one tenant_id and
never clearing the shared, session-scoped Qdrant collection between tests,
which made the hit-rate/hot-keys assertions silently order-dependent
(confirmed by reproduction: reordering the tests made the hit-rate test
fail). Minting a fresh tenant_id per test matches this project's own
established convention in test_qdrant_vector_store.py.

`authoritative_content` (used by the Sync Mixer test) is a plain Python
dict mirroring exactly what's really seeded into Qdrant, updated in lockstep
whenever the test changes a document's content. SyncCycle's own port only
needs a callable that returns a document's current true content; a real
production wiring would read that from Postgres or Qdrant directly, but the
mechanism under test here is sync_mixer.reconcile's real detection-and-evict
behavior, not the transport used to fetch the comparison value -- the same
reasoning CAG's own tests apply when a fake stands in for a port the test
isn't about.
"""
import time
import uuid
from datetime import datetime, timedelta

from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.application.sync_cycle import SyncCycle
from src.orchestration.application.tiering_policy import TieringPolicy
from src.orchestration.application.warm_cache import WarmCache
from src.orchestration.domain.entities import TierDecision
from src.orchestration.infrastructure.hf_frozen_cache import HFFrozenCache
from src.orchestration.infrastructure.in_memory_access_tracker import (
    InMemoryAccessFrequencyTracker,
)
from src.rag.application.search_documents import SearchDocuments
from src.rag.domain.entities import Chunk
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore

_WINDOW = timedelta(hours=1)
_NOW = datetime(2026, 9, 2, 12, 0, 0)

# Three "hot" documents (matching OVERVIEW.md's own worked example shape)
# and two "cold" ones -- traffic below is built to favor the hot three.
_DOCUMENTS = {
    "return_policy": (
        "Our return policy allows customers to return unopened items within "
        "thirty days of purchase for a full refund. Opened items may be "
        "exchanged within fourteen days."
    ),
    "shipping_policy": (
        "Standard shipping takes five to seven business days. Expedited "
        "shipping is available at checkout for an additional fee and "
        "arrives within two business days."
    ),
    "account_setup": (
        "To set up an account, provide your email address and choose a "
        "password of at least twelve characters. A verification email is "
        "sent immediately after registration."
    ),
    "investor_relations": (
        "Quarterly earnings calls are held on the second Tuesday of each "
        "quarter. Historical financial filings are available in the "
        "investor relations archive."
    ),
    "warehouse_safety": (
        "All warehouse personnel must complete safety certification before "
        "operating forklifts. Certification is renewed annually and "
        "requires a written and practical examination."
    ),
}

# 16 queries: 12 about the three hot topics, 4 about the two cold ones --
# an 80/20-shaped mix, but built from real query phrasings, not assumed.
_QUERY_MIX = [
    ("what is the return policy for unopened items", "return_policy"),
    ("how many days do I have to return something", "return_policy"),
    ("can I exchange an opened item", "return_policy"),
    ("what is the refund window", "return_policy"),
    ("how long does shipping take", "shipping_policy"),
    ("is expedited shipping available", "shipping_policy"),
    ("when will my order arrive", "shipping_policy"),
    ("what are the shipping options at checkout", "shipping_policy"),
    ("how do I create an account", "account_setup"),
    ("what are the password requirements", "account_setup"),
    ("do I get a verification email after signup", "account_setup"),
    ("how do I register for the first time", "account_setup"),
    ("when is the next earnings call", "investor_relations"),
    ("where can I find historical financial filings", "investor_relations"),
    ("what certification do forklift operators need", "warehouse_safety"),
    ("how often is safety certification renewed", "warehouse_safety"),
]


async def _seed_corpus(
    tenant_id: uuid.UUID, embedding_model, vector_store
) -> dict[str, uuid.UUID]:
    document_ids: dict[str, uuid.UUID] = {}
    for key, content in _DOCUMENTS.items():
        document_id = uuid.uuid4()
        document_ids[key] = document_id
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=document_id,
            content=content,
            embedding=embedding_model.embed(content),
        )
        await vector_store.upsert(chunk, tenant_id)
    return document_ids


async def test_cache_warmed_rag_shows_a_real_measured_hit_rate_and_latency_delta(
    qdrant_url, embedding_model, distilgpt2_model, distilgpt2_tokenizer
):
    tenant_id = uuid.uuid4()
    vector_store = QdrantVectorStore(qdrant_url)
    await vector_store.ensure_collection()
    document_ids = await _seed_corpus(tenant_id, embedding_model, vector_store)
    content_by_document_id = {
        document_ids[key]: content for key, content in _DOCUMENTS.items()
    }

    baseline = SearchDocuments(embedding_model, vector_store)

    # Step 1: real analytics -- record which document a plain baseline
    # retrieval actually returns for each query in the traffic mix.
    tracker = InMemoryAccessFrequencyTracker()
    for query, _expected_key in _QUERY_MIX:
        results = await baseline.execute(tenant_id, query, top_k=1)
        if results:
            tracker.record_access(tenant_id, results[0].document_id, _NOW)

    top_3 = tracker.most_accessed(tenant_id, 3, _WINDOW, _NOW)
    hot_keys = {key for key, doc_id in document_ids.items() if doc_id in top_3}
    print(f"real measured top-3 most-retrieved documents: {hot_keys}")
    # This project's own corpus really does show the skew OVERVIEW.md's
    # source assumes -- all three deliberately-hot documents rank above
    # both deliberately-cold ones, measured from real retrieval, not assumed.
    assert hot_keys == {"return_policy", "shipping_policy", "account_setup"}

    # Step 2: warm the real top-3 into a real HFFrozenCache.
    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)
    warmed = WarmCache(tracker, cache).execute(
        tenant_id, 3, _WINDOW, _NOW, lambda doc_id: content_by_document_id[doc_id]
    )
    retriever = CacheWarmedRetrieve(embedding_model, cache, baseline, similarity_threshold=0.3)
    for document_id in warmed:
        retriever.note_warmed(tenant_id, document_id, content_by_document_id[document_id])

    # Step 3: real hit-rate measurement, running the SAME traffic mix again.
    for query, _expected_key in _QUERY_MIX:
        await retriever.execute(tenant_id, query, top_k=1)
    hits, misses = retriever.stats()
    hit_rate = hits / (hits + misses)
    print(f"real measured Cache-Warmed RAG hit rate: {hits}/{hits + misses} = {hit_rate:.2%}")
    # This corpus/query mix's real ceiling is deterministic: 12 of 16
    # queries target the three warmed documents, so a healthy run lands at
    # 0.75. A tighter floor than a bare majority -- a review finding
    # confirmed empirically that >= 0.5 would still pass even with an
    # entire warmed document's matching silently broken (losing 4/16
    # queries drops the rate to exactly 0.5) -- 0.65 leaves real headroom
    # below the 0.75 ceiling while actually catching that regression.
    assert hit_rate >= 0.65

    # Step 4: real baseline-vs-combined latency comparison over the same
    # traffic (Combined Capability Validation, #10/#27's actual content).
    baseline_start = time.perf_counter()
    for query, _ in _QUERY_MIX:
        await baseline.execute(tenant_id, query, top_k=1)
    baseline_ms = (time.perf_counter() - baseline_start) * 1000

    combined_start = time.perf_counter()
    for query, _ in _QUERY_MIX:
        await retriever.execute(tenant_id, query, top_k=1)
    combined_ms = (time.perf_counter() - combined_start) * 1000

    print(
        f"real measured baseline (RAG-only) latency for {len(_QUERY_MIX)} queries: "
        f"{baseline_ms:.1f}ms; combined (Cache-Warmed RAG) latency: {combined_ms:.1f}ms"
    )
    # Honestly reported, not hidden: at this project's own small-corpus,
    # localhost-Qdrant scale, embedding cost dominates over Qdrant's own
    # round-trip cost, and CacheWarmedRetrieve pays one embedding call on
    # every query (to check the small warmed-set index) plus, on a miss, a
    # second embedding call inside the fallback retriever -- a real,
    # disclosed cost of keeping this batch strictly additive to RAG's
    # existing Retriever port rather than changing its signature to accept
    # a precomputed embedding. Whether combined ends up faster or slower
    # than baseline in this run depends on that real tradeoff, not on
    # anything this assertion should paper over -- the mechanism's actual
    # near-zero-TTFT claim is HFFrozenCache.prefill_latency_ms's own real
    # cold-vs-warm measurement, isolated from this harness's embedding-call
    # overhead (see test_hf_frozen_cache.py).


async def test_tiering_promotes_a_genuinely_hot_document_and_demotes_a_cooled_one(
    qdrant_url, embedding_model, distilgpt2_model, distilgpt2_tokenizer
):
    tenant_id = uuid.uuid4()
    vector_store = QdrantVectorStore(qdrant_url)
    await vector_store.ensure_collection()
    document_ids = await _seed_corpus(tenant_id, embedding_model, vector_store)
    content_by_document_id = {
        document_ids[key]: content for key, content in _DOCUMENTS.items()
    }
    hot_id = document_ids["return_policy"]
    cold_id = document_ids["investor_relations"]

    tracker = InMemoryAccessFrequencyTracker()
    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)
    policy = TieringPolicy(tracker, cache)

    # Window 1: return_policy gets real heavy traffic, investor_relations
    # gets none.
    for _ in range(10):
        tracker.record_access(tenant_id, hot_id, _NOW)

    promote_decision = policy.evaluate(
        tenant_id, hot_id, lambda doc_id: content_by_document_id[doc_id],
        promote_threshold=5, demote_threshold=2, window=_WINDOW, now=_NOW,
    )
    cold_decision = policy.evaluate(
        tenant_id, cold_id, lambda doc_id: content_by_document_id[doc_id],
        promote_threshold=5, demote_threshold=2, window=_WINDOW, now=_NOW,
    )
    print(
        f"real tiering decisions after heavy traffic: "
        f"hot={promote_decision}, cold={cold_decision}"
    )
    assert promote_decision == TierDecision.PROMOTED
    assert cache.contains(tenant_id, hot_id)
    assert cold_decision == TierDecision.UNCHANGED
    assert not cache.contains(tenant_id, cold_id)

    # Window 2: real time passes, hot_id's traffic dries up (no new
    # accesses inside the new window), so it demotes -- freeing whatever
    # the cache entry was occupying.
    later = _NOW + timedelta(hours=2)
    demote_decision = policy.evaluate(
        tenant_id, hot_id, lambda doc_id: content_by_document_id[doc_id],
        promote_threshold=5, demote_threshold=2, window=_WINDOW, now=later,
    )
    print(f"real tiering decision after traffic dried up: {demote_decision}")
    assert demote_decision == TierDecision.DEMOTED
    assert not cache.contains(tenant_id, hot_id)


async def test_sync_cycle_detects_and_evicts_a_real_content_change_with_real_latency(
    qdrant_url, embedding_model, distilgpt2_model, distilgpt2_tokenizer
):
    tenant_id = uuid.uuid4()
    vector_store = QdrantVectorStore(qdrant_url)
    await vector_store.ensure_collection()
    document_ids = await _seed_corpus(tenant_id, embedding_model, vector_store)
    authoritative_content = {
        document_ids[key]: content for key, content in _DOCUMENTS.items()
    }
    target_id = document_ids["return_policy"]

    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)
    cache.preload(tenant_id, target_id, authoritative_content[target_id])
    sync_cycle = SyncCycle(cache)

    # A real content change -- the same "$100 to $80" shape as OVERVIEW.md's
    # own worked example: the return window changes from thirty days to
    # forty-five. Reflected in both the test's own authoritative-content
    # lookup AND the real Qdrant index, since RAG's index is what actually
    # "wins" per the tiebreak rule.
    new_content = authoritative_content[target_id].replace("thirty days", "forty-five days")
    change_at = time.perf_counter()
    authoritative_content[target_id] = new_content
    new_chunk = Chunk(
        id=uuid.uuid4(), document_id=target_id, content=new_content,
        embedding=embedding_model.embed(new_content),
    )
    await vector_store.upsert(new_chunk, tenant_id)

    detected_at = None
    for _ in range(20):  # poll on a real short interval
        conflicts = sync_cycle.run(
            tenant_id, [target_id], lambda doc_id: authoritative_content[doc_id]
        )
        if conflicts:
            detected_at = time.perf_counter()
            break
        time.sleep(0.05)

    assert detected_at is not None, "SyncCycle never detected the real content change"
    latency_ms = (detected_at - change_at) * 1000
    print(
        f"real measured detection-to-eviction latency: {latency_ms:.1f}ms "
        f"(OVERVIEW.md's illustrative figure: 300000ms / 5 minutes -- not a "
        f"target this mechanism claims to hit, since this test drives "
        f"SyncCycle synchronously on a short poll rather than a real batch "
        f"schedule; the cadence is an operational choice for whoever "
        f"schedules SyncCycle.run in production, not something the "
        f"mechanism itself dictates)"
    )
    assert not cache.contains(tenant_id, target_id)
