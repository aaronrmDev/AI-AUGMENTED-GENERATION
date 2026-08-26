"""Manual comparison runner for the RAG Combinations batch (Batch E). Not
pytest-collected (no test_* function at module level) -- mirrors this
project's established pattern for scripts that make real model calls and
are run by hand.

Named run_combinations_comparison.py, not run_comparison.py: every
hyphenated scenario directory under evaluation/scenarios/ shares one flat
mypy module namespace, so any two scripts both named run_comparison.py
collide -- this has broken `mypy evaluation/` twice already in this
project's history.

Usage (PYTHONPATH=. is required):
    PYTHONPATH=. uv run python evaluation/scenarios/rag-combinations/\
run_combinations_comparison.py <combination>

<combination> is one of: multi-query-hyde, reranking-crag, production-grade,
fort-knox, speed-demon
"""
import asyncio
import dataclasses
import sys
import uuid
from pathlib import Path
from typing import Protocol

import ollama

from evaluation.application.run_comparison import RunComparison
from evaluation.domain.entities import Answer
from evaluation.infrastructure.markdown_report import render
from evaluation.infrastructure.ollama_judge import OllamaJudge
from evaluation.scenarios.loader import load_scenario
from src.rag.application.answer_question import AnswerQuestion
from src.rag.application.search_documents import SearchDocuments
from src.rag.application.self_rag_answer_question import SelfRAGAnswerQuestion
from src.rag.application.upload_document import UploadDocument
from src.rag.application.upload_document_with_parents import UploadDocumentWithParents
from src.rag.domain.entities import ChatAnswer, Chunk, Document, SearchResult
from src.rag.domain.ports import ChatModel, DocumentRepository, EmbeddingModel, Retriever
from src.rag.infrastructure.bm25_keyword_search import BM25KeywordSearch
from src.rag.infrastructure.compressing_retriever import CompressingRetriever
from src.rag.infrastructure.corrective_retriever import CorrectiveRetriever
from src.rag.infrastructure.cross_encoder_reranker import CrossEncoderReranker
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.hybrid_search_documents import HybridSearchDocuments
from src.rag.infrastructure.hyde_retriever import HyDERetriever
from src.rag.infrastructure.local_file_storage import LocalFileStorage
from src.rag.infrastructure.multi_query_retriever import MultiQueryRetriever
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel
from src.rag.infrastructure.parent_document_chunker import ParentDocumentChunker
from src.rag.infrastructure.parent_document_retriever import ParentDocumentRetriever
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore
from src.rag.infrastructure.reranking_retriever import RerankingRetriever
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.rag.infrastructure.text_extractor import TextExtractor

_SCENARIO_DIR = Path(__file__).parent
_MODEL_CONFIG = "qwen3.5, Ollama"
_COMBINATIONS = (
    "multi-query-hyde", "reranking-crag", "production-grade", "fort-knox", "speed-demon",
)


class _Answerer(Protocol):
    async def execute(self, tenant_id: uuid.UUID, question: str) -> ChatAnswer: ...


class _CorrectionObservingRetriever(Retriever):
    # Same non-invasive observation technique the rag-crag batch established:
    # counts how many times the wrapped inner retriever is called per
    # CorrectiveRetriever.execute() invocation -- 1 means no correction
    # fired, 2 means it did. Reused here rather than modifying
    # CorrectiveRetriever itself just to expose more.
    def __init__(self, inner: Retriever) -> None:
        self._inner = inner
        self.calls_this_question = 0

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        self.calls_this_question += 1
        return await self._inner.execute(tenant_id=tenant_id, query=query, top_k=top_k)

    def take_call_count(self) -> int:
        count = self.calls_this_question
        # A bypassed observer (e.g. a future refactor that stops wiring it
        # as CorrectiveRetriever's inner) would silently report count=0,
        # which reads identically to "correction never fired" -- this
        # batch's final review had to independently reproduce a live call to
        # confirm the observer was really being invoked (count=1, not the 0
        # a bypass would also produce) before trusting the 0%-correction
        # reports. Asserting >=1 here makes a bypass fail loudly instead.
        assert count >= 1, (
            "observer was never invoked -- it is not correctly wired as "
            "CorrectiveRetriever's inner retriever"
        )
        self.calls_this_question = 0
        return count


class _InMemoryDocumentRepository(DocumentRepository):
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    async def save_document(self, document: Document) -> None:
        pass

    async def update_document_status(
        self, document_id: uuid.UUID, status: str, chunk_count: int
    ) -> None:
        pass

    async def save_chunks(self, chunks: list[Chunk], tenant_id: uuid.UUID) -> None:
        self._chunks.extend(chunks)

    async def get_chunks_for_tenant(self, tenant_id: uuid.UUID) -> list[Chunk]:
        return self._chunks

    async def get_chunk_by_id(self, chunk_id: uuid.UUID) -> Chunk | None:
        return next((c for c in self._chunks if c.id == chunk_id), None)


def _build_retriever(
    combination: str,
    vector_retriever: Retriever,
    document_repository: DocumentRepository,
    chat_model: ChatModel,
    embedder: EmbeddingModel,
) -> tuple[Retriever, _CorrectionObservingRetriever | None]:
    # Composition orders per docs/superpowers/specs/2026-08-26-rag-combinations-design.md.
    # Returns (retriever, correction_observer) -- the observer is non-None
    # only for combinations that wrap a CorrectiveRetriever, so treatment()
    # can honestly report how often correction actually fires (the exact
    # measurement the rag-crag batch's final review found missing).
    #
    # CrossEncoderReranker/BM25KeywordSearch/HybridSearchDocuments are built
    # lazily, per branch below (not unconditionally at the top): the
    # cross-encoder loads a real HuggingFace model, and multi-query-hyde
    # uses none of these three -- this batch's final review caught the
    # earlier unconditional construction wasting a model load on every
    # multi-query-hyde run.
    if combination == "multi-query-hyde":
        # Multi-Query + HyDE only -- no Hybrid Search in this pairing's own
        # name, so the base stays plain vector search.
        retriever: Retriever = MultiQueryRetriever(
            inner=HyDERetriever(inner=vector_retriever, chat_model=chat_model),
            chat_model=chat_model,
        )
        return retriever, None

    reranker = CrossEncoderReranker()
    keyword_retriever = BM25KeywordSearch(document_repository=document_repository)
    hybrid: Retriever = HybridSearchDocuments(
        vector_retriever=vector_retriever, keyword_retriever=keyword_retriever
    )

    if combination == "reranking-crag":
        # Reranking + CRAG only -- no Hybrid Search in this pairing's own
        # name either, so the base stays plain vector search. CRAG wraps
        # Reranking so a correction's re-search is reranked again too
        # (RAG.md: "reranking runs again on the new results").
        observed = _CorrectionObservingRetriever(
            inner=RerankingRetriever(inner=vector_retriever, reranker=reranker)
        )
        return CorrectiveRetriever(inner=observed, chat_model=chat_model), observed

    if combination == "production-grade":
        observed = _CorrectionObservingRetriever(
            inner=RerankingRetriever(inner=hybrid, reranker=reranker)
        )
        corrected = CorrectiveRetriever(inner=observed, chat_model=chat_model)
        return CompressingRetriever(inner=corrected, embedding_model=embedder), observed

    if combination == "fort-knox":
        enriched: Retriever = MultiQueryRetriever(
            inner=HyDERetriever(inner=hybrid, chat_model=chat_model), chat_model=chat_model
        )
        observed = _CorrectionObservingRetriever(
            inner=RerankingRetriever(inner=enriched, reranker=reranker)
        )
        corrected = CorrectiveRetriever(inner=observed, chat_model=chat_model)
        return (
            ParentDocumentRetriever(inner=corrected, document_repository=document_repository),
            observed,
        )

    raise ValueError(f"_build_retriever does not handle {combination!r}")


def _make_answerer(
    combination: str,
    vector_retriever: Retriever,
    document_repository: DocumentRepository,
    chat_model: ChatModel,
    embedder: EmbeddingModel,
    top_k: int,
) -> tuple[_Answerer, _CorrectionObservingRetriever | None]:
    if combination == "speed-demon":
        # Self-RAG isn't a Retriever (its NO-gate branch skips retrieval
        # entirely, which the Retriever contract can't express -- same
        # reasoning as the rag-query-enhancement batch's design), so it
        # wraps the whole Hybrid Search -> Reranking -> Compression chain
        # as its search_documents dependency instead of composing through
        # _build_retriever.
        reranker = CrossEncoderReranker()
        keyword_retriever = BM25KeywordSearch(document_repository=document_repository)
        hybrid: Retriever = HybridSearchDocuments(
            vector_retriever=vector_retriever, keyword_retriever=keyword_retriever
        )
        search_documents = CompressingRetriever(
            inner=RerankingRetriever(inner=hybrid, reranker=reranker), embedding_model=embedder
        )
        return (
            SelfRAGAnswerQuestion(
                search_documents=search_documents, chat_model=chat_model, top_k=top_k
            ),
            None,
        )

    retriever, observer = _build_retriever(
        combination, vector_retriever, document_repository, chat_model, embedder
    )
    return AnswerQuestion(search_documents=retriever, chat_model=chat_model, top_k=top_k), observer


async def _run(combination: str) -> None:
    if combination not in _COMBINATIONS:
        raise ValueError(
            f"unknown combination {combination!r}; choose one of {list(_COMBINATIONS)}"
        )

    scenario = load_scenario(_SCENARIO_DIR)
    embedder = SentenceTransformersEmbedder()
    vector_store = QdrantVectorStore("http://localhost:6333")
    await vector_store.ensure_collection()

    ollama_client = ollama.AsyncClient(host="http://localhost:11434")
    chat_model = OllamaChatModel(client=ollama_client, model_id="qwen3.5")

    tenant_id = uuid.uuid4()
    document_repository = _InMemoryDocumentRepository()
    corpus_bytes = (_SCENARIO_DIR / "corpus" / "rag.md").read_bytes()

    if combination == "fort-knox":
        # Only Fort Knox needs parent-linked chunks (ParentDocumentRetriever
        # requires them) -- every other combination in this batch uploads
        # plain, unparented chunks.
        upload_with_parents = UploadDocumentWithParents(
            document_repository=document_repository,
            embedding_model=embedder,
            vector_store=vector_store,
            parent_document_chunker=ParentDocumentChunker(),
            extractor=TextExtractor(),
            file_storage=LocalFileStorage(),
        )
        document = await upload_with_parents.execute(
            tenant_id=tenant_id, filename="rag.txt", content=corpus_bytes
        )
    else:
        upload = UploadDocument(
            document_repository=document_repository,
            embedding_model=embedder,
            vector_store=vector_store,
            chunker=FixedSizeChunker(),
            extractor=TextExtractor(),
            file_storage=LocalFileStorage(),
        )
        document = await upload.execute(
            tenant_id=tenant_id, filename="rag.txt", content=corpus_bytes
        )
    print(f"Uploaded corpus under combination {combination!r}: {document.chunk_count} chunks")

    vector_retriever = SearchDocuments(embedding_model=embedder, vector_store=vector_store)
    answerer, observer = _make_answerer(
        combination, vector_retriever, document_repository, chat_model, embedder, top_k=5
    )

    async def baseline(question: str) -> Answer:
        text = await chat_model.generate(question=question, context="")
        return Answer(text=text, input_tokens=0, output_tokens=0, context="")

    # Populated only inside treatment() below, only for combinations that
    # wrap a CorrectiveRetriever (observer is not None) -- closed over here
    # rather than built into the notes= kwarg directly, since that argument
    # is evaluated BEFORE execute() calls treatment(). Same ordering-bug
    # precaution as every prior batch's dataclasses.replace()-after-execute()
    # pattern.
    correction_fired_log: list[bool] = []

    async def treatment(question: str) -> Answer:
        result = await answerer.execute(tenant_id=tenant_id, question=question)
        if observer is not None:
            fired = observer.take_call_count() > 1
            correction_fired_log.append(fired)
            print(f"[{combination} correction] {question!r} -> {'FIRED' if fired else 'not fired'}")
        context = "\n\n".join(source.content for source in result.sources)
        return Answer(text=result.answer, input_tokens=0, output_tokens=0, context=context)

    def success_check(question: str, answer: Answer) -> bool:
        # Each check is a direct encoding of the success_criterion written
        # for that question in queries.yaml, derived from the same reading
        # of the live docs/architecture/RAG.md.
        text = answer.text.lower()
        checks = {
            scenario.questions[0].question: lambda: "double-layer" in text and "rerank" in text,
            scenario.questions[1].question: lambda: (
                "maximum recall" in text and "vague" in text
            ),
            scenario.questions[2].question: lambda: (
                "hybrid" in text and "rerank" in text and "crag" in text
                and "compression" in text and "gold standard" in text
            ),
            scenario.questions[3].question: lambda: (
                "hybrid" in text and "multi-query" in text and "hyde" in text
                and "rerank" in text and "crag" in text and "parent" in text
                and "unbeatable" in text
            ),
            scenario.questions[4].question: lambda: (
                "self-rag" in text and "hybrid" in text and "rerank" in text
                and "compression" in text and "fort knox" in text
            ),
            scenario.questions[5].question: lambda: "parent" in text and "small chunk" in text,
            scenario.questions[6].question: lambda: "zero" in text and "conflict" in text,
        }
        return checks.get(question, lambda: False)()

    judge = OllamaJudge(client=ollama_client, model_id="qwen3.5")
    use_case = RunComparison(judge=judge, repeat_count=3)

    correction_caveat = (
        " CAVEAT 4: this combination wraps CorrectiveRetriever -- correction "
        "firing rate is instrumented and reported below, not assumed. "
        "IMPORTANT SCOPE LIMIT (added after this batch's final review): the "
        "instrument measures ONLY whether CorrectiveRetriever's re-search "
        "fired (i.e. whether zero of top_k passed relevance review) -- it "
        "does NOT measure the relevance filter's rejection rate. A 0% "
        "correction rate is fully compatible with the filter discarding "
        "most candidates on every call, as long as at least one survives; "
        "spot-checks during review found CRAG discarding roughly 3 of every "
        "5 reranked results on individual live calls while still reporting "
        "0% correction. A 0% rate here also isn't directly comparable to "
        "the standalone rag-crag batch's measured 14% firing rate: that "
        "batch used a different question set (including 2 deliberately "
        "vague, non-technical diagnostic questions written specifically to "
        "stress the correction path) and wrapped CorrectiveRetriever around "
        "a plain vector search with no reranker inside it, whereas every "
        "combination here already reranks before CRAG ever sees the "
        "results, structurally lowering how often nothing passes."
        if observer is not None
        else ""
    )
    fort_knox_caveat = (
        " CAVEAT 5 (FORT KNOX ONLY): BM25 keyword search reads every saved "
        "chunk, including the 7 parent chunks UploadDocumentWithParents "
        "also saves (48 total chunks in the repository; document.chunk_count "
        "below reports only the 41 children UploadDocumentWithParents "
        "returns, not both tiers). A parent chunk surfaced by the keyword "
        "arm is silently skipped by ParentDocumentRetriever (its parent_id "
        "is None) rather than expanded or erroring. MEASURED IMPACT (added "
        "after this batch's final review measured it directly, rather than "
        "assuming it was bounded): parents are BM25's natural favorites, "
        "not a rare edge case -- at roughly 969 tokens each against "
        "children's roughly 182, they accumulate far more of BM25Plus's "
        "per-matched-term score. Measured on this batch's real corpus and "
        "question set: parents occupied 6-7 of the 7 available parent "
        "slots in BM25's own top-20 on every one of the 7 questions (47 of "
        "49 possible appearances), with a parent's best BM25 rank landing "
        "at position 2 on all 7 questions. After Hybrid Search's RRF fusion "
        "and Reranking's cross-encoder pass, parents still made up roughly "
        "37% of results reaching ParentDocumentRetriever in a live sample -- "
        "over a third of what CRAG and Reranking worked to surface got "
        "silently dropped at the final stage. This can degenerate a "
        "treatment sample into an empty context (if too few of the "
        "survivors are children), which AnswerQuestion then answers with no "
        "retrieved material at all -- indistinguishable from a baseline "
        "call for that one sample. Not engineered around at this stage "
        "(would require production changes to BM25KeywordSearch or "
        "ParentDocumentRetriever, out of scope for this composition-only "
        "batch); disclosed here with its real measured size instead."
        if combination == "fort-knox"
        else ""
    )
    result = await use_case.execute(
        scenario_name=f"{scenario.name}-{combination}",
        model_config=_MODEL_CONFIG,
        success_criterion="see evaluation/scenarios/rag-combinations/queries.yaml",
        rag=True, cag=False, mag=False,
        notes=(
            f"combination={combination}, corpus=docs/architecture/RAG.md, "
            f"{document.chunk_count} chunks. CAVEAT 1: qualitative judge is "
            f"Ollama/qwen3.5, not Claude (no API credit balance at run "
            f"time) -- self-grading-bias risk, same as every prior batch in "
            f"this project. CAVEAT 2: baseline calls generate() with empty "
            f"context deliberately (strict context-only methodology, same "
            f"as every prior batch) -- expected to refuse or answer from "
            f"general training regardless of what a bare model might know. "
            f"Baseline's task-success figure can differ slightly run to run "
            f"for the identical call: task_success_rate is computed from "
            f"only the LAST of repeat_count=3 samples per question, not an "
            f"average, so a small amount of run-to-run variance on "
            f"borderline questions is expected and not itself evidence of a "
            f"methodology difference between combinations' reports. CAVEAT "
            f"3: the qualitative judge scores both arms against the "
            f"TREATMENT's own retrieved context (evaluation/application/"
            f"run_comparison.py, pre-existing, tracked as #148) -- baseline "
            f"can be penalized on 'groundedness' for citing facts sourced "
            f"from context it genuinely retrieved but that differs from "
            f"treatment's; no quality conclusion should be drawn from the "
            f"qualitative table alone until #148 is fixed."
            f"{correction_caveat}{fort_knox_caveat}"
        ),
        questions=[q.question for q in scenario.questions],
        baseline=baseline,
        treatment=treatment,
        success_check=success_check,
    )

    if observer is not None:
        fired_count = sum(correction_fired_log)
        total = len(correction_fired_log)
        rate = 0 if total == 0 else round(100 * fired_count / total)
        correction_note = (
            f" CORRECTION FIRING RATE: across this run's {total} treatment "
            f"calls (repeat_count=3 x 7 questions), correction fired "
            f"{fired_count} times ({rate}%). Real, per-question decisions "
            f"were printed live as '[{combination} correction] <question> "
            f"-> FIRED/not fired'; read that output directly rather than "
            f"treating this aggregate as the full record."
        )
        result = dataclasses.replace(result, notes=result.notes + correction_note)

    report_path = Path(f"evaluation/reports/rag-combinations-{combination}.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render(result), encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    asyncio.run(_run(sys.argv[1]))
