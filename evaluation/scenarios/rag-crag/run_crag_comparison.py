"""Manual comparison runner for the RAG CRAG batch. Not pytest-collected (no
test_* function at module level) -- mirrors this project's established
pattern (test_rag_smoke.py, and every prior batch's own scenario script) for
scripts that make real model calls and are run by hand.

Named run_crag_comparison.py, not run_comparison.py: every hyphenated
scenario directory under evaluation/scenarios/ shares one flat mypy module
namespace, so any two scripts both named run_comparison.py collide -- this
has broken `mypy evaluation/` twice already in this project's history
(rag-hybrid-reranking vs. rag-chunking-strategies, both originally
run_comparison.py).

Usage (PYTHONPATH=. is required -- these imports resolve relative to the
repo root, which "uv run python <path>" does not add to sys.path on its
own):
    PYTHONPATH=. uv run python evaluation/scenarios/rag-crag/run_crag_comparison.py

Unlike Batch C's three-strategy script, CRAG has only one meaningful
treatment arm (a plain vector search wrapped in CorrectiveRetriever), so
this script takes no CLI argument.
"""
import asyncio
import dataclasses
import uuid
from pathlib import Path

import ollama

from evaluation.application.run_comparison import RunComparison
from evaluation.domain.entities import Answer
from evaluation.infrastructure.markdown_report import render
from evaluation.infrastructure.ollama_judge import OllamaJudge
from evaluation.scenarios.loader import load_scenario
from src.rag.application.answer_question import AnswerQuestion
from src.rag.application.search_documents import SearchDocuments
from src.rag.application.upload_document import UploadDocument
from src.rag.domain.entities import Chunk, Document, SearchResult
from src.rag.domain.ports import DocumentRepository, Retriever
from src.rag.infrastructure.corrective_retriever import CorrectiveRetriever
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.local_file_storage import LocalFileStorage
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.rag.infrastructure.text_extractor import TextExtractor

_SCENARIO_DIR = Path(__file__).parent
_MODEL_CONFIG = "qwen3.5, Ollama"


class _CorrectionObservingRetriever(Retriever):
    # Counts how many times the wrapped inner retriever is called per
    # CorrectiveRetriever.execute() invocation -- 1 means the relevance
    # filter passed something (no correction), 2 means correction fired
    # (the initial evaluation pass, then the corrected re-search). This is
    # the non-invasive, non-production observation technique this project
    # established in the rag-query-enhancement batch's HyDE query log:
    # wrap the inner retriever, don't modify CorrectiveRetriever itself just
    # to expose more. Added after this batch's final review found the
    # original report never measured how often correction actually fires --
    # a real gap for a technique whose entire behavior hinges on that rate.
    def __init__(self, inner: Retriever) -> None:
        self._inner = inner
        self.calls_this_question = 0

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        self.calls_this_question += 1
        return await self._inner.execute(tenant_id=tenant_id, query=query, top_k=top_k)

    def take_call_count(self) -> int:
        count = self.calls_this_question
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


async def _run() -> None:
    scenario = load_scenario(_SCENARIO_DIR)
    embedder = SentenceTransformersEmbedder()
    vector_store = QdrantVectorStore("http://localhost:6333")
    await vector_store.ensure_collection()
    chunker = FixedSizeChunker()

    ollama_client = ollama.AsyncClient(host="http://localhost:11434")
    chat_model = OllamaChatModel(client=ollama_client, model_id="qwen3.5")

    tenant_id = uuid.uuid4()
    document_repository = _InMemoryDocumentRepository()
    upload = UploadDocument(
        document_repository=document_repository,
        embedding_model=embedder,
        vector_store=vector_store,
        chunker=chunker,
        extractor=TextExtractor(),
        file_storage=LocalFileStorage(),
    )
    corpus_bytes = (_SCENARIO_DIR / "corpus" / "rag.md").read_bytes()
    # filename="rag.txt", not the on-disk "rag.md" name: same verified-safe
    # substitution every prior batch's script uses -- TextExtractor only
    # branches on ".txt"/".pdf" and both hit the identical utf-8 decode path
    # for this plain-text corpus.
    document = await upload.execute(tenant_id=tenant_id, filename="rag.txt", content=corpus_bytes)
    print(f"Uploaded corpus: {document.chunk_count} chunks")

    vector_retriever = SearchDocuments(embedding_model=embedder, vector_store=vector_store)
    observed_inner = _CorrectionObservingRetriever(inner=vector_retriever)
    crag_retriever = CorrectiveRetriever(inner=observed_inner, chat_model=chat_model)

    plain_answerer = AnswerQuestion(
        search_documents=vector_retriever, chat_model=chat_model, top_k=5
    )
    crag_answerer = AnswerQuestion(
        search_documents=crag_retriever, chat_model=chat_model, top_k=5
    )

    async def baseline(question: str) -> Answer:
        result = await plain_answerer.execute(tenant_id=tenant_id, question=question)
        context = "\n\n".join(source.content for source in result.sources)
        return Answer(text=result.answer, input_tokens=0, output_tokens=0, context=context)

    # Populated only inside treatment() below -- closed over here rather than
    # built into the notes= keyword argument passed to RunComparison.execute()
    # directly, since that argument is evaluated BEFORE execute() ever calls
    # treatment(). Same ordering-bug precaution as every prior batch's
    # dataclasses.replace()-after-execute() pattern.
    correction_fired_log: list[bool] = []

    async def treatment(question: str) -> Answer:
        result = await crag_answerer.execute(tenant_id=tenant_id, question=question)
        call_count = observed_inner.take_call_count()
        correction_fired = call_count > 1
        correction_fired_log.append(correction_fired)
        print(f"[crag correction] {question!r} -> {'FIRED' if correction_fired else 'not fired'}")
        context = "\n\n".join(source.content for source in result.sources)
        return Answer(text=result.answer, input_tokens=0, output_tokens=0, context=context)

    def success_check(question: str, answer: Answer) -> bool:
        # Each check is a direct encoding of the success_criterion written
        # for that question in queries.yaml, derived from the same reading
        # of the live docs/architecture/RAG.md. Q6/Q7 are deliberately loose
        # (see queries.yaml) -- exploratory/diagnostic questions, not
        # sharp keyword checks, since they're testing whether CRAG's
        # relevance filter/correction shows any value at all, not confirming
        # it does.
        text = answer.text.lower()
        checks = {
            scenario.questions[0].question: lambda: (
                "10-20%" in text or "10–20%" in text or "10-20" in text
            ),
            scenario.questions[1].question: lambda: (
                "cross-encoder" in text
                and ("15-25%" in text or "15–25%" in text or "15-25" in text)
            ),
            scenario.questions[2].question: lambda: (
                "75%" in text and "50%" in text
            ),
            scenario.questions[3].question: lambda: (
                ("40-60%" in text or "40–60%" in text or "40-60" in text)
                and "hallucination" in text
            ),
            scenario.questions[4].question: lambda: (
                "hybrid" in text
                and "rerank" in text
                and "crag" in text
                and "compression" in text
                and "gold standard" in text
            ),
            scenario.questions[5].question: lambda: any(
                term in text
                for term in ("filter", "remove", "irrelevant", "noise", "valid", "relevan")
            ),
            scenario.questions[6].question: lambda: any(
                term in text
                for term in (
                    "re-search", "research", "retry", "again",
                    "alternative", "refine", "re-quer",
                )
            ),
        }
        return checks.get(question, lambda: False)()

    judge = OllamaJudge(client=ollama_client, model_id="qwen3.5")
    use_case = RunComparison(judge=judge, repeat_count=3)
    result = await use_case.execute(
        scenario_name=scenario.name,
        model_config=_MODEL_CONFIG,
        success_criterion="see evaluation/scenarios/rag-crag/queries.yaml",
        rag=True, cag=False, mag=False,
        notes=(
            f"corpus=docs/architecture/RAG.md, {document.chunk_count} chunks. "
            f"Baseline = plain vector search (SearchDocuments), Treatment = "
            f"CorrectiveRetriever-wrapped search (relevance-filters retrieved "
            f"results, single-shot corrected re-search when nothing passes). "
            f"CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no "
            f"API credit balance at run time) -- self-grading-bias risk, same "
            f"as every prior batch in this project. CAVEAT 2: this treatment "
            f"issues one extra complete() call per retrieved result (relevance "
            f"evaluation, run concurrently via asyncio.gather) plus, on a "
            f"correction, one more for the refined query -- expect the same "
            f"class of latency overhead every prior batch's extra-LLM-call "
            f"techniques showed. CAVEAT 3: this corpus is a single 14-chunk "
            f"document, which makes it hard to manufacture genuinely "
            f"irrelevant top-k results for a well-targeted query -- questions "
            f"6-7 are a deliberate stress test of CRAG's value on vague, "
            f"non-technical phrasing, but a null result there (no measurable "
            f"difference from baseline) is a legitimate, disclosed finding "
            f"about this corpus's small size, not evidence CorrectiveRetriever "
            f"itself is broken. CAVEAT 4 (JUDGE HARNESS, PROJECT-WIDE, NOT "
            f"SPECIFIC TO THIS BATCH): this batch's final review found "
            f"evaluation/application/run_comparison.py's qualitative judge "
            f"call used to score BOTH the baseline and treatment answers "
            f"against the TREATMENT's retrieved context only, structurally "
            f"penalizing baseline for citing facts sourced from context it "
            f"genuinely retrieved but that differed from treatment's -- "
            f"filed as #148 and confirmed against this exact batch's baseline "
            f"claims (verbatim-true against RAG.md, flagged unverifiable "
            f"anyway). #148 is now fixed: the judge scores each arm against "
            f"its own retrieved context (context_a/context_b), so this "
            f"report's groundedness/unverifiable-claims columns reflect what "
            f"each arm actually retrieved, not a shared, treatment-only view."
        ),
        questions=[q.question for q in scenario.questions],
        baseline=baseline,
        treatment=treatment,
        success_check=success_check,
    )

    yes_count = sum(correction_fired_log)
    total = len(correction_fired_log)
    correction_note = (
        f" CORRECTION FIRING RATE: across this run's {total} treatment calls "
        f"(repeat_count=3 x 7 questions), correction fired {yes_count} times "
        f"({0 if total == 0 else round(100 * yes_count / total)}%). Real, "
        f"honest, per-question decisions were printed live as '[crag "
        f"correction] <question> -> FIRED/not fired'; read that output "
        f"directly rather than treating this aggregate as the full record."
    )
    result = dataclasses.replace(result, notes=result.notes + correction_note)

    report_path = Path("evaluation/reports/rag-crag-crag.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render(result), encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    asyncio.run(_run())
