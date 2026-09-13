"""Cascade measurements for the orchestration meta-layer, in three parts:
tier latency against Concept 5's 10/50/2000ms budgets, the router-off vs.
router-on stale-text ablation, and a self-versus-self RunComparison
(RAG-only AnswerQuestion vs. UnifiedAnswerQuestion) on qwen3.5.

Starts its own Postgres and Qdrant containers, so it needs Docker and Ollama
but no compose stack. The four tier thresholds are imported unchanged from
tests/integration/orchestration_env.py, where they were measured on a
smaller corpus; this run tests whether they carry over rather than re-tuning
them. Not pytest-collected.

Usage (from the repository root):
    PYTHONPATH=. python evaluation/scenarios/orchestration-meta-layer/run_cascade_comparison.py
"""
import asyncio
import dataclasses
import os
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import ollama
from alembic.config import Config
from sqlalchemy.engine import make_url
from testcontainers.postgres import PostgresContainer
from testcontainers.qdrant import QdrantContainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from alembic import command
from evaluation.application.run_comparison import RunComparison
from evaluation.domain.cascade_metrics import (
    AllocatorTally,
    StaleAnswerTally,
    summarize_tier_latency,
    tally_staleness,
)
from evaluation.domain.entities import Answer
from evaluation.infrastructure.cascade_report import render_cascade_measurements
from evaluation.infrastructure.markdown_report import render
from evaluation.infrastructure.ollama_judge import OllamaJudge
from evaluation.scenarios.loader import load_scenario
from src.identity.infrastructure.db import get_engine, get_sessionmaker, set_tenant_context
from src.mag.application.queries.find_semantic_facts import FindSemanticFacts
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
    Paradigm,
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
from src.rag.application.answer_question import AnswerQuestion
from src.rag.application.search_documents import SearchDocuments
from src.rag.domain.entities import Chunk
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder
from tests.integration.orchestration_env import (
    CAG_HIT,
    CAG_PARTIAL,
    MAG_HIT,
    MAG_PARTIAL,
    create_user_and_session,
)

_SCENARIO_DIR = Path(__file__).parent
_CASCADE_REPORT = Path("evaluation/reports/orchestration-meta-layer-cascade.md")
_COMPARISON_REPORT = Path("evaluation/reports/orchestration-meta-layer-comparison.md")
_MODEL_ID = "qwen3.5"
_APP_DB_PASSWORD = "evaluation-only-app-user-password"
_LATENCY_REPEATS = 5
_COMPARISON_REPEATS = 5
_SMALL_WINDOW = 1_000
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
        "Our return policy allows customers to return unopened items within forty-five days "
        "of purchase for a full refund. The window changed from thirty days today.",
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
_TEN = re.compile(r"\b(10|ten)\b")
_CHECKS: list[Callable[[str], bool]] = [
    lambda t: "45" in t or "forty-five" in t,
    lambda t: "45" in t or "forty-five" in t,
    lambda t: ("five" in t and "seven" in t) or ("5" in t and "7" in t),
    lambda t: "verification" in t or "twelve" in t or "12" in t,
    lambda t: any(term in t for term in ("two-year", "two year", "2-year", "2 year")),
    lambda t: "out of stock" in t,
    lambda t: "20%" in t or "20 percent" in t or "twenty percent" in t,
    lambda t: "expedited" in t or "two-day" in t or "2-day" in t,
    lambda t: _TEN.search(t) is not None,
    lambda t: "out of stock" in t and _TEN.search(t) is not None,
]


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


async def _measure(app_url: str, qdrant_url: str) -> None:
    scenario = load_scenario(_SCENARIO_DIR)
    questions = [q.question for q in scenario.questions]
    checks = dict(zip(questions, _CHECKS, strict=True))
    embedder = SentenceTransformersEmbedder()
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    chat_model = OllamaChatModel(ollama.AsyncClient(), _MODEL_ID)
    engine = get_engine(app_url)
    sessionmaker = get_sessionmaker(engine)

    async with sessionmaker() as db_session:
        tenant_id = uuid.uuid4()
        user_id, session_id = await create_user_and_session(db_session, tenant_id)
        vector_store = QdrantVectorStore(qdrant_url)
        await vector_store.ensure_collection()
        search = SearchDocuments(embedder, vector_store)
        cache = HFFrozenCache(tokenizer=tokenizer, model=model)
        warmed = CacheWarmedRetrieve(embedder, cache, search, similarity_threshold=CAG_HIT)
        for current, frozen in _DOCUMENTS:
            document_id = uuid.uuid4()
            chunk = Chunk(
                id=uuid.uuid4(),
                document_id=document_id,
                content=current,
                embedding=embedder.embed(current),
            )
            await vector_store.upsert(chunk, tenant_id)
            if frozen is not None:
                cache.preload(tenant_id, document_id, frozen)
                warmed.note_warmed(tenant_id, document_id, frozen)
        await set_tenant_context(db_session, tenant_id)
        repository = PostgresSemanticMemoryRepository(db_session)
        for key, value in _FACTS:
            fact = SemanticMemory(
                id=uuid.uuid4(),
                user_id=user_id,
                fact_key=key,
                fact_value=value,
                embedding=embedder.embed(value),
            )
            await repository.save(fact, tenant_id)
        await db_session.commit()

        async def begin_turn() -> None:
            # invalidate(), not rollback(): a MAG query the cascade cancels
            # mid-flight leaves this session on a terminated connection that
            # rollback() and close() cannot recover (measured in
            # tests/integration/test_orchestration_meta_layer.py). Invalidating
            # before every turn keeps that recovery out of the timed path.
            await db_session.invalidate()
            await set_tenant_context(db_session, tenant_id)

        def cascade(timeouts: TierTimeouts) -> LatencyCascade:
            return LatencyCascade(
                [
                    CagTier(warmed, hit_threshold=CAG_HIT, partial_threshold=CAG_PARTIAL),
                    MagTier(
                        FindSemanticFacts(repository),
                        hit_threshold=MAG_HIT,
                        partial_threshold=MAG_PARTIAL,
                    ),
                    RagTier(search, top_k=3),
                ],
                timeouts,
            )

        def request(query: str, embedding: list[float]) -> TierRequest:
            return TierRequest(tenant_id, user_id, session_id, query, embedding)

        prototype = PrototypeQueryClassifier(embedder)
        lexical = LexicalQueryClassifier()

        # Part 1: tier latency under Concept 5's real budgets.
        budgeted = cascade(TierTimeouts())
        await begin_turn()
        await budgeted.run(request("warm up", embedder.embed("warm up")))  # untimed warm-up
        attempts: list[TierAttempt] = []
        embed_ms: list[float] = []
        for _ in range(_LATENCY_REPEATS):
            for question in questions:
                await begin_turn()
                started = time.perf_counter()
                embedding = embedder.embed(question)
                embed_ms.append(_ms(started))
                decision = decide(await prototype.score(question, embedding))
                result = await budgeted.run(request(question, embedding), decision)
                attempts.extend(result.attempts)
        embed_ms.sort()
        tiers = summarize_tier_latency(attempts, _BUDGETS_MS)

        # Part 2: which return-policy text reaches the model, router off vs. on.
        generous = cascade(_GENEROUS)
        arms: dict[str, QueryClassifier | None] = {
            "router off (Concept 5 cascade as drawn)": None,
            "router on (lexical)": lexical,
            "router on (prototype, MiniLM)": prototype,
        }
        tallies: list[StaleAnswerTally] = []
        for arm, classifier in arms.items():
            contexts: list[str] = []
            for query in _FRESHNESS_QUERIES:
                await begin_turn()
                embedding = embedder.embed(query)
                arm_decision = (
                    None
                    if classifier is None
                    else decide(await classifier.score(query, embedding))
                )
                result = await generous.run(request(query, embedding), arm_decision)
                contexts.append("\n".join(item.content for item in result.items))
            tallies.append(
                tally_staleness(arm, contexts, "within thirty days", "forty-five days")
            )

        # Part 3a: dynamic reallocation vs. static base slices in a small window.
        dynamic_dropped = static_dropped = dynamic_tokens = static_tokens = 0
        for question in questions:
            await begin_turn()
            embedding = embedder.embed(question)
            decision = decide(await prototype.score(question, embedding))
            result = await generous.run(request(question, embedding), decision)
            dynamic = assemble_context(
                result.items, allocate(_SMALL_WINDOW, result.contributing)
            )
            static = assemble_context(result.items, allocate(_SMALL_WINDOW, PARADIGM_ORDER))
            dynamic_dropped += sum(dynamic.dropped.values())
            static_dropped += sum(static.dropped.values())
            dynamic_tokens += sum(dynamic.tokens_used.values())
            static_tokens += sum(static.tokens_used.values())
        allocator = AllocatorTally(
            _SMALL_WINDOW,
            len(questions),
            dynamic_dropped,
            static_dropped,
            dynamic_tokens,
            static_tokens,
        )

        frozen_count = sum(1 for _, frozen in _DOCUMENTS if frozen is not None)
        cascade_notes = (
            f"Corpus: {len(_DOCUMENTS)} documents in Qdrant ({frozen_count} also frozen into a "
            f"distilgpt2 HFFrozenCache on CPU, the return policy in its superseded thirty-day "
            f"version), {len(_FACTS)} MAG semantic facts in Postgres. Thresholds carried over "
            f"unchanged from the integration corpus: CAG hit {CAG_HIT}, partial {CAG_PARTIAL}; "
            f"MAG hit {MAG_HIT}, partial {MAG_PARTIAL}. Tier latency: {len(questions)} questions "
            f"x {_LATENCY_REPEATS} repeats, prototype routing, default TierTimeouts, after one "
            f"untimed warm-up. Query embedding (MiniLM, CPU) is paid before any tier runs: p50 "
            f"{embed_ms[len(embed_ms) // 2]:.2f}ms, max {embed_ms[-1]:.2f}ms."
        )
        _CASCADE_REPORT.write_text(
            render_cascade_measurements(tiers, tallies, allocator, cascade_notes),
            encoding="utf-8",
        )
        print(_CASCADE_REPORT.read_text(encoding="utf-8"))

        # Part 3b: self-versus-self comparison with real generation and judging.
        unified = UnifiedAnswerQuestion(
            embedder,
            prototype,
            generous,
            chat_model,
            budget_recorder=PostgresSessionBudgetRecorder(sessionmaker),
        )
        baseline_answerer = AnswerQuestion(
            search_documents=search, chat_model=chat_model, top_k=3
        )
        routes: dict[str, str] = {}

        async def baseline(question: str) -> Answer:
            await begin_turn()
            result = await baseline_answerer.execute(tenant_id=tenant_id, question=question)
            return Answer(
                text=result.answer,
                input_tokens=chat_model.last_input_tokens,
                output_tokens=chat_model.last_output_tokens,
                context="\n\n".join(source.content for source in result.sources),
            )

        async def treatment(question: str) -> Answer:
            await begin_turn()
            result = await unified.execute(tenant_id, user_id, session_id, question)
            route = sorted(p.value for p in result.decision.paradigms) if result.decision else []
            routes[question] = (
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
            success_criterion="see evaluation/scenarios/orchestration-meta-layer/queries.yaml",
            rag=True,
            cag=True,
            mag=True,
            questions=questions,
            baseline=baseline,
            treatment=treatment,
            success_check=lambda question, answer: checks[question](answer.text.lower()),
            reference_contexts=[q.gold_passage for q in scenario.questions],
            notes=(
                "Baseline = RAG-only AnswerQuestion (top_k=3). Treatment = "
                "UnifiedAnswerQuestion: prototype routing, the CAG/MAG/RAG cascade with "
                "generous timeouts, the 128K budget allocator, and a real session budget "
                "record. CAVEAT 1: judge and generator are both qwen3.5 (self-grading risk, "
                "as in every earlier batch). CAVEAT 2: CAG is a CPU distilgpt2 proxy whose "
                "lookup, not generation speed, is what the treatment exercises. CAVEAT 3: "
                "question 1 asks for the return window without any freshness cue; a CAG hit "
                "on the superseded policy there is a Sync Mixer failure the router is not "
                "designed to catch, and is reported, not hidden."
            ),
        )
        comparison = dataclasses.replace(
            comparison,
            notes=comparison.notes
            + " ROUTES (last repeat): "
            + "; ".join(f"{q!r}: {r}" for q, r in routes.items()),
        )
        _COMPARISON_REPORT.write_text(render(comparison), encoding="utf-8")
        print(_COMPARISON_REPORT.read_text(encoding="utf-8"))
    await engine.dispose()


def main() -> None:
    # Migrations run before the event loop starts: alembic's env drives its own
    # async engine with asyncio.run, which cannot nest inside a running loop.
    with (
        PostgresContainer("pgvector/pgvector:pg16") as postgres,
        QdrantContainer("qdrant/qdrant:v1.16.2") as qdrant,
    ):
        database_url = postgres.get_connection_url().replace(
            "postgresql+psycopg2", "postgresql+asyncpg"
        )
        os.environ["DATABASE_URL"] = database_url
        os.environ["APP_DB_PASSWORD"] = _APP_DB_PASSWORD
        command.upgrade(Config("alembic.ini"), "head")
        app_url = (
            make_url(database_url)
            .set(username="app_user", password=_APP_DB_PASSWORD)
            .render_as_string(hide_password=False)
        )
        asyncio.run(_measure(app_url, f"http://127.0.0.1:{qdrant.get_exposed_port(6333)}"))


if __name__ == "__main__":
    main()
