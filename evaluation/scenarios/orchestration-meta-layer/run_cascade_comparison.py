"""Cascade measurements for the orchestration meta-layer, in three selectable parts:

- cascade: tier latency against Concept 5's 10/50/2000ms budgets, the
  router-off vs. router-on stale-text ablation, and dynamic vs. static budget
  slices across a sweep of context windows. No LLM, a few minutes.
- retrievers: the same tier latency with every question forced into a PARALLEL
  route, once per RAG composition behind RagTier (SearchDocuments, compression,
  bi-encoder reranking, HyDE, hybrid BM25 search, cross-encoder reranking, and
  CacheWarmedRetrieve), which shows whether a retriever's own CPU work
  starves the other tiers. No LLM: HyDE's passage comes from a stub.
- comparison: a self-versus-self RunComparison on qwen3.5, RAG-only
  AnswerQuestion vs. UnifiedAnswerQuestion routed by the MiniLM prototype
  classifier.
- oracle: the same comparison with the treatment routed by each question's
  labeled route in queries.yaml, which separates what the pipeline adds from
  the classifier's routing errors.

Starts its own Postgres and Qdrant containers, so it needs Docker, plus Ollama
for the two comparison parts; no compose stack. The four tier thresholds come
unchanged from evaluation/scenarios/orchestration_meta_layer_thresholds.py,
where they were measured on the integration corpus; this run tests whether
they carry over rather than re-tuning them. Not pytest-collected.

Usage (from the repository root):
    PYTHONPATH=. python evaluation/scenarios/orchestration-meta-layer/run_cascade_comparison.py
    PYTHONPATH=. python evaluation/scenarios/orchestration-meta-layer/run_cascade_comparison.py \
        --parts cascade
"""
import argparse
import asyncio
import dataclasses
import io
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ollama
import yaml
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession
from testcontainers.postgres import PostgresContainer
from testcontainers.qdrant import QdrantContainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from alembic import command
from evaluation.application.run_comparison import RunComparison
from evaluation.domain.cascade_metrics import (
    AllocatorTally,
    StaleAnswerTally,
    TierLatencySummary,
    summarize_tier_latency,
    tally_staleness,
)
from evaluation.domain.entities import Answer
from evaluation.infrastructure.cascade_report import (
    render_cascade_measurements,
    render_retriever_measurements,
)
from evaluation.infrastructure.markdown_report import render
from evaluation.infrastructure.ollama_judge import OllamaJudge
from evaluation.scenarios.loader import load_scenario
from evaluation.scenarios.orchestration_meta_layer_checks import SUCCESS_CHECKS
from evaluation.scenarios.orchestration_meta_layer_thresholds import (
    CAG_HIT,
    CAG_PARTIAL,
    MAG_HIT,
    MAG_PARTIAL,
)
from src.identity.infrastructure.db import get_engine, get_sessionmaker, set_tenant_context
from src.mag.domain.entities import SemanticMemory
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.orchestration.application.assemble_context import assemble_context
from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.application.cascade_tiers import CagTier, MagTier, RagTier
from src.orchestration.application.latency_cascade import LatencyCascade, TierTimeouts
from src.orchestration.application.unified_answer_question import UnifiedAnswerQuestion
from src.orchestration.domain.budget_allocator import allocate
from src.orchestration.domain.entities import (
    PARADIGM_ORDER,
    CascadeResult,
    Paradigm,
    RoutingDecision,
    RoutingMode,
    TierAttempt,
    TierRequest,
)
from src.orchestration.domain.paradigm_router import decide
from src.orchestration.domain.ports import QueryClassifier
from src.orchestration.infrastructure.hf_frozen_cache import HFFrozenCache
from src.orchestration.infrastructure.lexical_query_classifier import LexicalQueryClassifier
from src.orchestration.infrastructure.postgres_session_budget_recorder import (
    PostgresSessionBudgetRecorder,
)
from src.orchestration.infrastructure.prototype_query_classifier import (
    PrototypeQueryClassifier,
)
from src.orchestration.infrastructure.session_scoped_semantic_fact_search import (
    SessionScopedSemanticFactSearch,
)
from src.rag.application.answer_question import AnswerQuestion
from src.rag.application.search_documents import SearchDocuments
from src.rag.domain.entities import Chunk, Document
from src.rag.domain.ports import ChatModel, DocumentRepository, Retriever
from src.rag.infrastructure.bi_encoder_rerank_reranker import BiEncoderRerankReranker
from src.rag.infrastructure.bm25_keyword_search import BM25KeywordSearch
from src.rag.infrastructure.caching_embedding_model import CachingEmbeddingModel
from src.rag.infrastructure.compressing_retriever import CompressingRetriever
from src.rag.infrastructure.cross_encoder_reranker import CrossEncoderReranker
from src.rag.infrastructure.hybrid_search_documents import HybridSearchDocuments
from src.rag.infrastructure.hyde_retriever import HyDERetriever
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore
from src.rag.infrastructure.reranking_retriever import RerankingRetriever
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder

_SCENARIO_DIR = Path(__file__).parent
_CASCADE_REPORT = Path("evaluation/reports/orchestration-meta-layer-cascade.md")
_RETRIEVERS_REPORT = Path("evaluation/reports/orchestration-meta-layer-retrievers.md")
_COMPARISON_REPORT = Path("evaluation/reports/orchestration-meta-layer-comparison.md")
_ORACLE_REPORT = Path("evaluation/reports/orchestration-meta-layer-comparison-oracle.md")
_PARTS = ("cascade", "retrievers", "comparison", "oracle")
_MODEL_ID = "qwen3.5"
_APP_DB_PASSWORD = "evaluation-only-app-user-password"
# A syntactically valid Argon2id hash for seeded users; nobody logs in here.
_SEED_PASSWORD_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
_LATENCY_REPEATS = 5
_COMPARISON_REPEATS = 5
# 1,000 tokens leaves every slice roomier than this corpus needs; the smaller
# windows are where static slices start to bind and reallocation can matter.
_ALLOCATOR_WINDOWS = (1_000, 400, 250, 150)
_BUDGETS_MS = {Paradigm.CAG: 10.0, Paradigm.MAG: 50.0, Paradigm.RAG: 2000.0}
_GENEROUS = TierTimeouts(cag=5.0, mag=5.0, rag=10.0)

_SHIPPING = (
    "Standard shipping takes five to seven business days. Expedited shipping arrives "
    "within two business days."
)
_ACCOUNT = (
    "To set up an account, provide your email address and choose a password of at least "
    "twelve characters. A verification email is sent immediately after registration."
)
_WARRANTY = "Every product carries a two-year limited warranty covering manufacturing defects."
# (text Qdrant holds now, text frozen into CAG or None when the document is RAG-only)
_DOCUMENTS: list[tuple[str, str | None]] = [
    (
        # Never mentions the superseded thirty days, so an answer drawn from the
        # current document cannot name only the old value (see orchestration_env.py).
        "Our return policy allows customers to return unopened items within forty-five days "
        "of purchase for a full refund. The return window was extended to forty-five days today.",
        "Our return policy allows customers to return unopened items within thirty days of "
        "purchase for a full refund.",
    ),
    (_SHIPPING, _SHIPPING),
    (_ACCOUNT, _ACCOUNT),
    (_WARRANTY, _WARRANTY),
    (
        "As of this morning, the blue running shoes are out of stock in sizes 9 and 10, with "
        "a restock expected on Friday.",
        None,
    ),
    ("A flash sale today takes 20% off every backpack until midnight.", None),
]
_FACTS = [
    ("preferred_shipping_speed", "The user prefers expedited two-day shipping on every order."),
    ("shoe_size", "The user wears size 10 running shoes."),
    ("loyalty_tier", "The user is a Gold loyalty member."),
]
# Every pipeline piece here is deterministic, so the ablation uses varied
# phrasings rather than repeats of one phrasing.
_FRESHNESS_QUERIES = [
    "What changed in the return policy today?",
    "Has the return window been updated today?",
    "What is the latest return policy?",
    "Did the refund window change recently?",
    "What is the current number of days to return an item?",
    "Is there a new return policy as of today?",
]
_CAVEATS = (
    "CAVEAT 1: judge and generator are both qwen3.5 (self-grading risk, as in every "
    "earlier batch). CAVEAT 2: CAG is a CPU distilgpt2 proxy whose lookup, not generation "
    "speed, is what the treatment exercises. CAVEAT 3: question 1 is a static policy "
    "question, so Concept 1's rule sends it to CAG, which holds the superseded thirty-day "
    "policy in this corpus; when CAG answers it alone, the stale value is what the model "
    "sees. That is a Sync Mixer failure routing cannot fix, and it is reported, not "
    "relabeled."
)


class _OracleClassifier(QueryClassifier):
    """Scores each question 1.0 for its labeled paradigms and 0.0 for the rest,
    which decide() turns into exactly that route, in CASCADE mode."""

    def __init__(self, routes: dict[str, frozenset[Paradigm]]) -> None:
        self._routes = routes

    async def score(self, query: str, query_embedding: list[float]) -> dict[Paradigm, float]:
        expected = self._routes[query]
        return {paradigm: 1.0 if paradigm in expected else 0.0 for paradigm in PARADIGM_ORDER}


class _FreshPassageChatModel(ChatModel):
    """Stands in for HyDE's generator: returns at once, with a new passage every call.

    Generation is awaited network I/O that yields the event loop, so it isn't what
    this measurement is after. A passage that differs on every call means no cache can
    turn its embedding into a lookup, so every attempt pays that CPU cost.
    """

    def __init__(self) -> None:
        self._calls = 0

    async def complete(self, prompt: str) -> str:
        self._calls += 1
        question = prompt.rsplit("Question: ", 1)[-1]
        return f"A confident hypothetical answer to {question!r}, draft {self._calls}."

    async def generate(self, question: str, context: str) -> str:
        return await self.complete(question)

    async def stream(self, question: str, context: str) -> AsyncIterator[str]:
        # This harness measures classification cost, not generation streaming --
        # a single-chunk stand-in keeps it instantiable without adding a
        # streaming-specific measurement nothing here asks for.
        yield await self.generate(question, context)


class _InMemoryChunkRepository(DocumentRepository):
    """Holds the corpus's chunks for BM25KeywordSearch, which ranks whatever
    get_chunks_for_tenant returns. The runner seeds one tenant, so there is nothing
    to filter by."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    async def save_document(self, document: Document) -> None:
        raise NotImplementedError("the measurement only reads chunks")

    async def update_document_status(
        self, document_id: uuid.UUID, status: str, chunk_count: int
    ) -> None:
        raise NotImplementedError("the measurement only reads chunks")

    async def save_chunks(self, chunks: list[Chunk], tenant_id: uuid.UUID) -> None:
        raise NotImplementedError("the measurement only reads chunks")

    async def get_chunks_for_tenant(self, tenant_id: uuid.UUID) -> list[Chunk]:
        return self._chunks

    async def get_chunk_by_id(self, chunk_id: uuid.UUID) -> Chunk | None:
        return next((chunk for chunk in self._chunks if chunk.id == chunk_id), None)


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


async def _create_user_and_session(
    session: AsyncSession, tenant_id: uuid.UUID
) -> tuple[uuid.UUID, uuid.UUID]:
    await set_tenant_context(session, tenant_id)
    now = datetime.now(UTC)
    user_id, session_id = uuid.uuid4(), uuid.uuid4()
    await session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id, created_at, updated_at) "
            "VALUES (:id, :email, :hashed_password, :tenant_id, :created_at, :updated_at)"
        ),
        {
            "id": user_id,
            "email": f"{user_id}@example.com",
            "hashed_password": _SEED_PASSWORD_HASH,
            "tenant_id": tenant_id,
            "created_at": now,
            "updated_at": now,
        },
    )
    await session.execute(
        text(
            "INSERT INTO sessions (id, user_id, tenant_id, title) "
            "VALUES (:id, :user_id, :tenant_id, :title)"
        ),
        {"id": session_id, "user_id": user_id, "tenant_id": tenant_id, "title": "evaluation"},
    )
    return user_id, session_id


async def _measure(
    app_url: str,
    qdrant_url: str,
    parts: frozenset[str],
    retrievers_report: Path,
    retrievers_note: str,
) -> None:
    scenario = load_scenario(_SCENARIO_DIR)
    questions = [q.question for q in scenario.questions]
    checks = dict(zip(questions, SUCCESS_CHECKS, strict=True))
    raw_questions: list[dict[str, Any]] = yaml.safe_load(
        (_SCENARIO_DIR / "queries.yaml").read_text(encoding="utf-8")
    )["questions"]
    routes = {
        str(entry["question"]): frozenset(Paradigm(str(value)) for value in entry["route"])
        for entry in raw_questions
    }
    unlabeled = [question for question in questions if question not in routes]
    if unlabeled:
        # The oracle classifier would raise on these, and UnifiedAnswerQuestion would
        # quietly route them to every tier in parallel instead.
        raise SystemExit(f"queries.yaml has no route label for: {unlabeled}")
    # One shared caching embedder. UnifiedAnswerQuestion embeds each question and
    # SearchDocuments embeds the same text again inside the RAG tier; sharing the
    # cache makes that second embed a lookup instead of ~9ms of CPU on the event
    # loop, which starved the other tiers in PARALLEL routes before this was wired.
    raw_embedder = SentenceTransformersEmbedder()
    embedder = CachingEmbeddingModel(raw_embedder)
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    chat_model = OllamaChatModel(ollama.AsyncClient(), _MODEL_ID)
    engine = get_engine(app_url)
    sessionmaker = get_sessionmaker(engine)

    tenant_id = uuid.uuid4()
    vector_store = QdrantVectorStore(qdrant_url)
    await vector_store.ensure_collection()
    search = SearchDocuments(embedder, vector_store)
    cache = HFFrozenCache(tokenizer=tokenizer, model=model)
    warmed = CacheWarmedRetrieve(embedder, cache, search, similarity_threshold=CAG_HIT)
    chunks: list[Chunk] = []
    frozen_documents: list[tuple[uuid.UUID, str]] = []
    for current, frozen in _DOCUMENTS:
        document_id = uuid.uuid4()
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=document_id,
            content=current,
            embedding=embedder.embed(current),
        )
        await vector_store.upsert(chunk, tenant_id)
        chunks.append(chunk)
        if frozen is not None:
            cache.preload(tenant_id, document_id, frozen)
            warmed.note_warmed(tenant_id, document_id, frozen)
            frozen_documents.append((document_id, frozen))

    # Seeding is the only work done through a caller-held session. Every
    # tier and the budget recorder open their own, so no MAG timeout below
    # can leave a broken session for anything else to trip over.
    async with sessionmaker() as seed_session, seed_session.begin():
        user_id, session_id = await _create_user_and_session(seed_session, tenant_id)
        repository = PostgresSemanticMemoryRepository(seed_session)
        for key, value in _FACTS:
            fact = SemanticMemory(
                id=uuid.uuid4(),
                user_id=user_id,
                fact_key=key,
                fact_value=value,
                embedding=embedder.embed(value),
            )
            await repository.save(fact, tenant_id)

    def cascade(timeouts: TierTimeouts, rag: Retriever = search) -> LatencyCascade:
        return LatencyCascade(
            [
                CagTier(warmed, hit_threshold=CAG_HIT, partial_threshold=CAG_PARTIAL),
                MagTier(
                    SessionScopedSemanticFactSearch(sessionmaker),
                    hit_threshold=MAG_HIT,
                    partial_threshold=MAG_PARTIAL,
                ),
                RagTier(rag, top_k=3),
            ],
            timeouts,
        )

    def request(query: str, embedding: list[float]) -> TierRequest:
        return TierRequest(tenant_id, user_id, session_id, query, embedding)

    prototype = PrototypeQueryClassifier(embedder)
    lexical = LexicalQueryClassifier()
    generous = cascade(_GENEROUS)

    if "cascade" in parts:
        # Part 1: tier latency under Concept 5's real budgets.
        budgeted = cascade(TierTimeouts())
        await budgeted.run(request("warm up", embedder.embed("warm up")))  # untimed warm-up
        attempts: list[TierAttempt] = []
        embed_ms: list[float] = []
        for _ in range(_LATENCY_REPEATS):
            for question in questions:
                # Embedding cost is timed on the raw model: after the first repeat the
                # shared cache would make it look free, which a new question isn't.
                started = time.perf_counter()
                raw_embedder.embed(question)
                embed_ms.append(_ms(started))
                embedding = embedder.embed(question)
                decision = decide(await prototype.score(question, embedding))
                result = await budgeted.run(request(question, embedding), decision)
                attempts.extend(result.attempts)
        await budgeted.drain()
        embed_ms.sort()
        tiers = summarize_tier_latency(attempts, _BUDGETS_MS)

        # Part 2: which return-policy text reaches the model, router off vs. on.
        arms: dict[str, QueryClassifier | None] = {
            "router off (Concept 5 cascade as drawn)": None,
            "router on (lexical)": lexical,
            "router on (prototype, MiniLM)": prototype,
        }
        tallies: list[StaleAnswerTally] = []
        for arm, classifier in arms.items():
            contexts: list[str] = []
            for query in _FRESHNESS_QUERIES:
                embedding = embedder.embed(query)
                arm_decision: RoutingDecision | None = (
                    None
                    if classifier is None
                    else decide(await classifier.score(query, embedding))
                )
                result = await generous.run(request(query, embedding), arm_decision)
                contexts.append("\n".join(item.content for item in result.items))
            tallies.append(
                tally_staleness(arm, contexts, "within thirty days", "forty-five days")
            )

        # Part 3a: dynamic reallocation vs. static base slices, across windows.
        sweep_results: list[CascadeResult] = []
        for question in questions:
            embedding = embedder.embed(question)
            decision = decide(await prototype.score(question, embedding))
            sweep_results.append(await generous.run(request(question, embedding), decision))
        allocators: list[AllocatorTally] = []
        for window in _ALLOCATOR_WINDOWS:
            dynamic_dropped = static_dropped = dynamic_tokens = static_tokens = 0
            for result in sweep_results:
                dynamic = assemble_context(result.items, allocate(window, result.contributing))
                static = assemble_context(result.items, allocate(window, PARADIGM_ORDER))
                dynamic_dropped += sum(dynamic.dropped.values())
                static_dropped += sum(static.dropped.values())
                dynamic_tokens += sum(dynamic.tokens_used.values())
                static_tokens += sum(static.tokens_used.values())
            allocators.append(
                AllocatorTally(
                    window,
                    len(sweep_results),
                    dynamic_dropped,
                    static_dropped,
                    dynamic_tokens,
                    static_tokens,
                )
            )

        frozen_count = sum(1 for _, frozen in _DOCUMENTS if frozen is not None)
        cascade_notes = (
            f"Corpus: {len(_DOCUMENTS)} documents in Qdrant ({frozen_count} also frozen into a "
            f"distilgpt2 HFFrozenCache on CPU, the return policy in its superseded thirty-day "
            f"version), {len(_FACTS)} MAG semantic facts in Postgres. Thresholds carried over "
            f"unchanged from the integration corpus: CAG hit {CAG_HIT}, partial {CAG_PARTIAL}; "
            f"MAG hit {MAG_HIT}, partial {MAG_PARTIAL}. Tier latency: {len(questions)} "
            f"questions x {_LATENCY_REPEATS} repeats, prototype routing, default TierTimeouts, "
            f"after one untimed warm-up. Query embedding (MiniLM, CPU) is paid before any tier "
            f"runs: p50 {embed_ms[len(embed_ms) // 2]:.2f}ms, max {embed_ms[-1]:.2f}ms. The RAG "
            f"retriever shares a CachingEmbeddingModel with the pipeline, so its embedding of "
            f"the already-embedded question is a lookup rather than CPU work on the event "
            f"loop, which starved the other tiers in PARALLEL routes before it was wired. A "
            f"timed-out CAG match keeps its worker thread running to completion (threads can't "
            f"be cancelled), so a tier measured right after a CAG timeout can include that "
            f"overlap. Allocator sweep: the same {len(questions)} prototype-routed questions, "
            f"assembled at each window with dynamic reallocation and with static base slices."
        )
        _CASCADE_REPORT.write_text(
            render_cascade_measurements(tiers, tallies, allocators, cascade_notes),
            encoding="utf-8",
        )
        print(_CASCADE_REPORT.read_text(encoding="utf-8"))

    if "retrievers" in parts:
        # Every tier runs at once, so CPU work any RAG composition does on the event
        # loop shows up as CAG and MAG attempts overrunning their budgets. The
        # compositions' own extra embeds go through the uncached model, so every
        # repeat pays them; the question's embedding stays a shared-cache lookup.
        uncached_search = SearchDocuments(raw_embedder, vector_store)
        # Its own warmed set, so matching a query embeds it through the uncached model.
        uncached_warmed = CacheWarmedRetrieve(
            raw_embedder, cache, search, similarity_threshold=CAG_HIT
        )
        for document_id, frozen in frozen_documents:
            uncached_warmed.note_warmed(tenant_id, document_id, frozen)
        compositions: dict[str, Retriever] = {
            "SearchDocuments": search,
            "CompressingRetriever": CompressingRetriever(search, raw_embedder),
            "RerankingRetriever + BiEncoderRerankReranker": RerankingRetriever(
                search, BiEncoderRerankReranker(raw_embedder)
            ),
            "HyDERetriever": HyDERetriever(uncached_search, _FreshPassageChatModel()),
            "HybridSearchDocuments (SearchDocuments + BM25KeywordSearch)": (
                HybridSearchDocuments(search, BM25KeywordSearch(_InMemoryChunkRepository(chunks)))
            ),
            "RerankingRetriever + CrossEncoderReranker": RerankingRetriever(
                search, CrossEncoderReranker()
            ),
            "CacheWarmedRetrieve": uncached_warmed,
        }
        every_tier = RoutingDecision(
            frozenset(PARADIGM_ORDER),
            RoutingMode.PARALLEL,
            {paradigm: 1.0 for paradigm in PARADIGM_ORDER},
        )
        rows: list[tuple[str, list[TierLatencySummary]]] = []
        for name, composition in compositions.items():
            parallel = cascade(TierTimeouts(), composition)
            await parallel.run(request("warm up", embedder.embed("warm up")), every_tier)
            composition_attempts: list[TierAttempt] = []
            for _ in range(_LATENCY_REPEATS):
                for question in questions:
                    embedding = embedder.embed(question)
                    result = await parallel.run(request(question, embedding), every_tier)
                    composition_attempts.extend(result.attempts)
            await parallel.drain()
            rows.append((name, summarize_tier_latency(composition_attempts, _BUDGETS_MS)))
        retrievers_notes = (
            f"Every question is forced into a PARALLEL route across CAG, MAG, and RAG, once "
            f"per RAG composition behind RagTier: {len(questions)} questions x "
            f"{_LATENCY_REPEATS} repeats each, default TierTimeouts, after one untimed "
            f"warm-up, on the corpus and thresholds of orchestration-meta-layer-cascade.md. "
            f"The pipeline's shared CachingEmbeddingModel serves the question's embedding to "
            f"every tier and to SearchDocuments, so SearchDocuments alone embeds nothing new. "
            f"The other compositions embed through the uncached MiniLM model on every "
            f"attempt: CompressingRetriever its query and every candidate sentence, "
            f"BiEncoderRerankReranker its query and every candidate, HyDERetriever's search "
            f"its generated passage, and CacheWarmedRetrieve its query, against its own "
            f"warmed copy of the {len(frozen_documents)} frozen documents. BM25KeywordSearch, "
            f"inside HybridSearchDocuments, ranks an in-memory copy of the same "
            f"{len(chunks)} chunks, and CrossEncoderReranker runs the real ms-marco MiniLM "
            f"cross-encoder on CPU. HyDE's passage comes from a stub chat model that "
            f"returns a new passage at once: generation is awaited network I/O that yields "
            f"the loop, so the stub isolates the embedding. A timed-out CAG match keeps its "
            f"worker thread running to completion, so an attempt measured right after a CAG "
            f"timeout can include that overlap. {retrievers_note}"
        ).strip()
        retrievers_report.write_text(
            render_retriever_measurements(rows, retrievers_notes), encoding="utf-8"
        )
        print(retrievers_report.read_text(encoding="utf-8"))

    baseline_answerer = AnswerQuestion(search_documents=search, chat_model=chat_model, top_k=3)

    async def compare(
        classifier: QueryClassifier, report_path: Path, treatment_description: str
    ) -> None:
        unified = UnifiedAnswerQuestion(
            embedder,
            classifier,
            generous,
            chat_model,
            budget_recorder=PostgresSessionBudgetRecorder(sessionmaker),
        )
        routes_seen: dict[str, str] = {}

        async def baseline(question: str) -> Answer:
            result = await baseline_answerer.execute(tenant_id=tenant_id, question=question)
            return Answer(
                text=result.answer,
                input_tokens=chat_model.last_input_tokens,
                output_tokens=chat_model.last_output_tokens,
                context="\n\n".join(source.content for source in result.sources),
            )

        async def treatment(question: str) -> Answer:
            result = await unified.execute(tenant_id, user_id, session_id, question)
            if result.routing_fallback is not None:
                # A fallback runs every tier in parallel, which isn't the arm being measured.
                raise RuntimeError(
                    f"routing fell back ({result.routing_fallback}) on {question!r}"
                )
            route = sorted(p.value for p in result.decision.paradigms) if result.decision else []
            routes_seen[question] = (
                f"routed {route}, attempts "
                f"{[(a.paradigm.value, a.outcome.value) for a in result.attempts]}"
            )
            return Answer(
                text=result.answer,
                input_tokens=chat_model.last_input_tokens,
                output_tokens=chat_model.last_output_tokens,
                context="\n\n".join(source.content for source in result.sources),
            )

        comparison = await RunComparison(
            judge=OllamaJudge(client=ollama.AsyncClient(), model_id=_MODEL_ID),
            repeat_count=_COMPARISON_REPEATS,
        ).execute(
            scenario_name=scenario.name,
            model_config=f"{_MODEL_ID}, Ollama",
            success_criterion=(
                "each question's success_criterion in "
                "evaluation/scenarios/orchestration-meta-layer/queries.yaml, checked by "
                "evaluation/scenarios/orchestration_meta_layer_checks.py"
            ),
            rag=True,
            cag=True,
            mag=True,
            questions=questions,
            baseline=baseline,
            treatment=treatment,
            success_check=lambda question, answer: checks[question](answer.text),
            reference_contexts=[q.gold_passage for q in scenario.questions],
            notes=(
                f"Baseline = RAG-only AnswerQuestion (top_k=3). Treatment = "
                f"{treatment_description}, with the CAG/MAG/RAG cascade under generous "
                f"timeouts, the 128K budget allocator, and a real session budget record. "
                f"{_CAVEATS}"
            ),
        )
        # Let cancelled tiers finish cleaning up before the next arm or engine disposal.
        await generous.drain()
        comparison = dataclasses.replace(
            comparison,
            notes=comparison.notes
            + " ROUTES (last repeat): "
            + "; ".join(f"{q!r}: {r}" for q, r in routes_seen.items()),
        )
        report_path.write_text(render(comparison), encoding="utf-8")
        print(report_path.read_text(encoding="utf-8"))

    if "comparison" in parts:
        await compare(
            prototype,
            _COMPARISON_REPORT,
            "UnifiedAnswerQuestion routed by the MiniLM prototype classifier",
        )
    if "oracle" in parts:
        await compare(
            _OracleClassifier(routes),
            _ORACLE_REPORT,
            "UnifiedAnswerQuestion routed by each question's labeled route in queries.yaml "
            "(an oracle), which separates what the pipeline adds from classifier routing errors",
        )
    await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestration meta-layer cascade measurements.")
    parser.add_argument(
        "--parts",
        default=",".join(_PARTS),
        help="comma-separated subset of: " + ", ".join(_PARTS),
    )
    parser.add_argument(
        "--retrievers-report",
        type=Path,
        default=_RETRIEVERS_REPORT,
        help="where the retrievers part writes its report",
    )
    parser.add_argument(
        "--retrievers-note",
        default="",
        help="text appended to the retrievers report's notes, such as the commit measured",
    )
    args = parser.parse_args()
    parts = frozenset(part.strip() for part in args.parts.split(",") if part.strip())
    unknown = parts - set(_PARTS)
    if unknown:
        parser.error(f"unknown parts: {sorted(unknown)}")
    # The comparison report contains characters (Δ) the Windows console's cp1252
    # can't encode; printing it crashed a full run after both reports were
    # written, skipping engine.dispose().
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    # Migrations run before the event loop starts: alembic's env drives its own
    # async engine with asyncio.run, which cannot nest inside a running loop.
    with (
        PostgresContainer("pgvector/pgvector:pg16") as postgres,
        QdrantContainer("qdrant/qdrant:v1.16.2") as qdrant,
    ):
        # 127.0.0.1, not the container's "localhost": on this Windows host asyncpg
        # connections via "localhost" stall (see database_url in
        # tests/integration/conftest.py for the measurement).
        database_url = (
            make_url(postgres.get_connection_url())
            .set(drivername="postgresql+asyncpg", host="127.0.0.1")
            .render_as_string(hide_password=False)
        )
        os.environ["DATABASE_URL"] = database_url
        os.environ["APP_DB_PASSWORD"] = _APP_DB_PASSWORD
        command.upgrade(Config("alembic.ini"), "head")
        app_url = (
            make_url(database_url)
            .set(username="app_user", password=_APP_DB_PASSWORD)
            .render_as_string(hide_password=False)
        )
        asyncio.run(
            _measure(
                app_url,
                f"http://127.0.0.1:{qdrant.get_exposed_port(6333)}",
                parts,
                args.retrievers_report,
                args.retrievers_note,
            )
        )


if __name__ == "__main__":
    main()
