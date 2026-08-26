"""Manual comparison runner for the RAG Chunking Strategies batch. Not
pytest-collected (no test_* function at module level) -- mirrors this
project's existing test_rag_smoke.py / test_docker_compose_smoke.py pattern
for scripts that make real model calls and are run by hand.

Usage:
    uv run python evaluation/scenarios/rag-chunking-strategies/run_chunking_comparison.py <strategy>

<strategy> is one of: fixed-size, sentence-based, semantic, sliding-window, structure-aware
"""
import asyncio
import sys
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
from src.rag.domain.entities import Chunk, Document
from src.rag.domain.ports import Chunker, DocumentRepository
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.local_file_storage import LocalFileStorage
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore
from src.rag.infrastructure.semantic_chunker import SemanticChunker
from src.rag.infrastructure.sentence_based_chunker import SentenceBasedChunker
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.rag.infrastructure.sliding_window_chunker import SlidingWindowChunker
from src.rag.infrastructure.structure_aware_chunker import StructureAwareChunker
from src.rag.infrastructure.text_extractor import TextExtractor

_SCENARIO_DIR = Path(__file__).parent
_MODEL_CONFIG = "qwen3.5, Ollama"


def _make_chunker(name: str, embedder: SentenceTransformersEmbedder) -> Chunker:
    chunkers: dict[str, Chunker] = {
        "fixed-size": FixedSizeChunker(),
        "sentence-based": SentenceBasedChunker(),
        "semantic": SemanticChunker(embedder),
        "sliding-window": SlidingWindowChunker(),
        "structure-aware": StructureAwareChunker(),
    }
    if name not in chunkers:
        raise ValueError(f"unknown strategy {name!r}; choose one of {sorted(chunkers)}")
    return chunkers[name]


async def _run(strategy: str) -> None:
    scenario = load_scenario(_SCENARIO_DIR)
    embedder = SentenceTransformersEmbedder()
    vector_store = QdrantVectorStore("http://localhost:6333")
    await vector_store.ensure_collection()
    chunker = _make_chunker(strategy, embedder)

    ollama_client = ollama.AsyncClient(host="http://localhost:11434")
    chat_model = OllamaChatModel(client=ollama_client, model_id="qwen3.5")

    tenant_id = uuid.uuid4()
    upload = UploadDocument(
        document_repository=_NullDocumentRepository(),
        embedding_model=embedder,
        vector_store=vector_store,
        chunker=chunker,
        extractor=TextExtractor(),
        file_storage=LocalFileStorage(),
    )
    corpus_bytes = (_SCENARIO_DIR / "corpus" / "rag.md").read_bytes()
    # filename="rag.txt" here, not the on-disk "rag.md" name: verified against
    # the real TextExtractor (src/rag/infrastructure/text_extractor.py) that it
    # only branches on ".txt" and ".pdf" and raises UnsupportedFileType for
    # anything else, including ".md". Uploading under a ".txt" name hits the
    # exact same decode path (utf-8, errors="replace") a ".md" extension would
    # need, since the corpus is plain text either way -- this is the smallest
    # change that makes the upload succeed without touching TextExtractor
    # itself, which is shared production code out of this task's scope.
    document = await upload.execute(
        tenant_id=tenant_id, filename="rag.txt", content=corpus_bytes
    )
    print(f"Uploaded corpus under strategy {strategy!r}: {document.chunk_count} chunks")

    search = SearchDocuments(embedding_model=embedder, vector_store=vector_store)
    answer_question = AnswerQuestion(search_documents=search, chat_model=chat_model, top_k=5)

    async def baseline(question: str) -> Answer:
        text = await chat_model.generate(question=question, context="")
        return Answer(text=text, input_tokens=0, output_tokens=0, context="")

    async def treatment(question: str) -> Answer:
        result = await answer_question.execute(tenant_id=tenant_id, question=question)
        # Reconstructs exactly what AnswerQuestion.execute() itself joined
        # internally to build the chat model's context -- it isn't returned
        # on ChatAnswer, so this is the only way to get it back out for the
        # judge without changing AnswerQuestion's return contract.
        retrieved_context = "\n\n".join(source.content for source in result.sources)
        return Answer(
            text=result.answer, input_tokens=0, output_tokens=0, context=retrieved_context
        )

    def success_check(question: str, answer: Answer) -> bool:
        # Each check is a direct encoding of the success_criterion written for
        # that question in queries.yaml -- both were derived from the same
        # reading of the real, current docs/architecture/RAG.md. Matching is
        # case-insensitive substring matching, tolerant of minor wording
        # variance the model might introduce (e.g. "15-20%" vs "15–20%"),
        # but still anchored to the specific figures/phrases RAG.md states.
        checks = {
            # Q1: Fixed Size's suggested starting point -- RAG.md: "Fixed Size
            # at 512 tokens with 10% overlap is the source's own suggested
            # starting point."
            scenario.questions[0].question: lambda a: (
                "512" in a.text
                and ("10%" in a.text or "10 %" in a.text or "10-percent" in a.text.lower())
            ),
            # Q2: Semantic Chunking's requirement and complexity -- RAG.md:
            # "at the cost of needing an embedding model to do the grouping,
            # which makes it the most complex of the six to implement."
            scenario.questions[1].question: lambda a: (
                "embedding model" in a.text.lower() and "complex" in a.text.lower()
            ),
            # Q3: The Production Grade archetype's four techniques and the
            # "gold standard" phrase -- RAG.md: "Hybrid Search casts the wide
            # net, Reranking narrows it to true relevance, CRAG validates the
            # result and triggers correction when needed, and Context
            # Compression optimizes what's left before it reaches the LLM,
            # which the source calls 'the gold standard for production RAG
            # systems'."
            scenario.questions[2].question: lambda a: (
                "hybrid search" in a.text.lower()
                and "reranking" in a.text.lower()
                and "crag" in a.text.lower()
                and "context compression" in a.text.lower()
                and "gold standard" in a.text.lower()
            ),
            # Q4: Parent Document Retrieval's expected impact figure --
            # RAG.md: "Its expected impact in isolation is a +15-20% gain in
            # completeness."
            scenario.questions[3].question: lambda a: (
                ("15-20%" in a.text or "15–20%" in a.text or "15-20" in a.text)
                and "completeness" in a.text.lower()
            ),
            # Q5: Sliding Window's tradeoff -- RAG.md: "gives good coverage
            # and keeps the narrative flow intact across chunk boundaries,
            # but multiplies how much has to be stored since the windows
            # overlap."
            scenario.questions[4].question: lambda a: (
                "stor" in a.text.lower()
                and (
                    "narrative" in a.text.lower()
                    or "continuity" in a.text.lower()
                    or "flow" in a.text.lower()
                )
            ),
        }
        return checks.get(question, lambda a: False)(answer)

    # Deviation from the harness design (ClaudeJudge): the Anthropic account had
    # no credit balance to run a live Claude-judged comparison (confirmed via a
    # direct API call, not assumed). Ruled with the user: use qwen3.5 itself as
    # the qualitative judge instead, at zero cost. This is a real methodology
    # weakening, not a like-for-like swap -- the same model family that
    # generated the treatment answer is also scoring it, a self-grading-bias
    # risk a stronger, independent judge model doesn't carry. Flagged in every
    # report's notes field for exactly this reason; re-running with ClaudeJudge
    # once credits exist is the correct follow-up.
    judge = OllamaJudge(client=ollama_client, model_id="qwen3.5")
    use_case = RunComparison(judge=judge, repeat_count=3)
    result = await use_case.execute(
        scenario_name=scenario.name,
        model_config=_MODEL_CONFIG,
        success_criterion="see evaluation/scenarios/rag-chunking-strategies/queries.yaml",
        rag=True, cag=False, mag=False,
        notes=(
            f"strategy={strategy}, corpus=docs/architecture/RAG.md, "
            f"{document.chunk_count} chunks. CAVEAT: qualitative judge is "
            f"Ollama/qwen3.5, not Claude (no API credit balance at run time) -- "
            f"same model family judging its own treatment output, a real "
            f"self-grading-bias risk; re-run with ClaudeJudge once credits "
            f"exist before treating these judge scores as final."
        ),
        questions=[q.question for q in scenario.questions],
        baseline=baseline,
        treatment=treatment,
        success_check=success_check,
    )

    report_path = Path(f"evaluation/reports/rag-chunking-{strategy}.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render(result), encoding="utf-8")
    print(f"Report written to {report_path}")


class _NullDocumentRepository(DocumentRepository):
    async def save_document(self, document: Document) -> None:
        pass

    async def update_document_status(
        self, document_id: uuid.UUID, status: str, chunk_count: int
    ) -> None:
        pass

    async def save_chunks(self, chunks: list[Chunk], tenant_id: uuid.UUID) -> None:
        pass

    async def get_chunks_for_tenant(self, tenant_id: uuid.UUID) -> list[Chunk]:
        # DocumentRepository gained this abstract method after this script
        # was first written (rag-hybrid-reranking batch, BM25's data
        # source) -- this scenario never uses it (no BM25/Hybrid strategy
        # here), so an empty list keeps this class instantiable without
        # changing anything this scenario's own chunkers actually measure.
        return []

    async def get_chunk_by_id(self, chunk_id: uuid.UUID) -> Chunk | None:
        # DocumentRepository gained this abstract method too (rag-parent-
        # doc-compression batch, ParentDocumentRetriever's parent lookup) --
        # this scenario never uses it (no parent-document strategy here),
        # so None keeps this class instantiable without changing anything
        # this scenario's own chunkers actually measure.
        return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    asyncio.run(_run(sys.argv[1]))
