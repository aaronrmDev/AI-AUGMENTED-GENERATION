"""Manual comparison runner for the RAG Parent Document Chunking + Retrieval
+ Context Compression batch. Not pytest-collected (no test_* function at
module level) -- mirrors this project's existing test_rag_smoke.py /
test_docker_compose_smoke.py pattern for scripts that make real model calls
and are run by hand.

Named run_parent_doc_compression_comparison.py, not run_comparison.py:
the rag-hybrid-reranking batch (already merged to develop) broke
`mypy evaluation/` because two scenario scripts both resolved to the bare
module name run_comparison -- neither hyphenated scenario directory
(evaluation/scenarios/rag-*) can be a Python package, so mypy sees both
files as the same top-level module. Every scenario's entrypoint needs a
name unique across the whole evaluation/scenarios/ tree.

Usage (PYTHONPATH=. is required -- these imports resolve relative to the
repo root, which "uv run python <path>" does not add to sys.path on its
own):
    PYTHONPATH=. uv run python evaluation/scenarios/rag-parent-doc-compression/\
run_parent_doc_compression_comparison.py <strategy>

<strategy> is one of: parent-document, context-compression,
parent-document-compression
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
from src.rag.application.upload_document_with_parents import UploadDocumentWithParents
from src.rag.domain.entities import Chunk, Document
from src.rag.domain.ports import DocumentRepository, EmbeddingModel, Retriever
from src.rag.infrastructure.compressing_retriever import CompressingRetriever
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.local_file_storage import LocalFileStorage
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel
from src.rag.infrastructure.parent_document_chunker import ParentDocumentChunker
from src.rag.infrastructure.parent_document_retriever import ParentDocumentRetriever
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.rag.infrastructure.text_extractor import TextExtractor

_SCENARIO_DIR = Path(__file__).parent
_MODEL_CONFIG = "qwen3.5, Ollama"

_STRATEGIES = ("parent-document", "context-compression", "parent-document-compression")
# Only these two strategies need the two-tier parent/child index built by
# UploadDocumentWithParents/ParentDocumentChunker. context-compression alone
# compresses whatever a plain retriever returns and has no use for a parent
# tier -- it uploads with the plain UploadDocument/FixedSizeChunker pair
# instead (see the upload branch in _run below).
_PARENT_DOCUMENT_STRATEGIES = ("parent-document", "parent-document-compression")


def _make_retriever(
    name: str,
    vector_retriever: Retriever,
    document_repository: DocumentRepository,
    embedder: EmbeddingModel,
) -> Retriever:
    # DISCLOSURE, proactive: unlike RerankingRetriever/HybridSearchDocuments
    # (the rag-hybrid-reranking batch), which both widen a candidate pool
    # with candidate_k before narrowing it back down, neither
    # ParentDocumentRetriever nor CompressingRetriever does that -- both
    # request exactly top_k from their inner retriever (see
    # src/rag/infrastructure/parent_document_retriever.py and
    # src/rag/infrastructure/compressing_retriever.py). There is no
    # widen-then-narrow step here, so the earlier batch's "candidate_k
    # exceeds corpus size" measurement caveat does NOT apply to this batch --
    # every strategy below always retrieves exactly top_k candidates from the
    # vector store, regardless of how small the corpus is.
    if name == "parent-document":
        return ParentDocumentRetriever(
            inner=vector_retriever, document_repository=document_repository
        )
    if name == "context-compression":
        return CompressingRetriever(inner=vector_retriever, embedding_model=embedder)
    if name == "parent-document-compression":
        parent = ParentDocumentRetriever(
            inner=vector_retriever, document_repository=document_repository
        )
        return CompressingRetriever(inner=parent, embedding_model=embedder)
    raise ValueError(f"unknown strategy {name!r}; choose one of {list(_STRATEGIES)}")


async def _run(strategy: str) -> None:
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose one of {list(_STRATEGIES)}")

    scenario = load_scenario(_SCENARIO_DIR)
    embedder = SentenceTransformersEmbedder()
    vector_store = QdrantVectorStore("http://localhost:6333")
    await vector_store.ensure_collection()

    ollama_client = ollama.AsyncClient(host="http://localhost:11434")
    chat_model = OllamaChatModel(client=ollama_client, model_id="qwen3.5")

    tenant_id = uuid.uuid4()
    # Single _InMemoryDocumentRepository instance, reused for both the upload
    # path and ParentDocumentRetriever's get_chunk_by_id lookups (parent and
    # parent-document-compression strategies). Same rationale as the
    # rag-hybrid-reranking script this is patterned on: this script's single
    # process, single tenant_id, one-shot run has no need for real
    # persistence across processes, and a real PostgresDocumentRepository
    # here would only re-exercise infrastructure already covered by
    # tests/integration/test_postgres_document_repository.py.
    document_repository = _InMemoryDocumentRepository()
    corpus_bytes = (_SCENARIO_DIR / "corpus" / "rag.md").read_bytes()

    # Upload path branches by strategy -- a simple if is enough for 2
    # branches, not a shared abstraction. filename="rag.txt" here, not the
    # on-disk "rag.md" name: verified against the real TextExtractor
    # (src/rag/infrastructure/text_extractor.py) that it only branches on
    # ".txt" and ".pdf" and raises UnsupportedFileType for anything else,
    # including ".md". Uploading under a ".txt" name hits the exact same
    # decode path (utf-8, errors="replace") a ".md" extension would need,
    # since the corpus is plain text either way -- this is the smallest
    # change that makes the upload succeed without touching TextExtractor
    # itself, which is shared production code out of this task's scope.
    if strategy in _PARENT_DOCUMENT_STRATEGIES:
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
    print(f"Uploaded corpus under strategy {strategy!r}: {document.chunk_count} chunks")

    vector_retriever = SearchDocuments(embedding_model=embedder, vector_store=vector_store)
    retriever = _make_retriever(strategy, vector_retriever, document_repository, embedder)
    answer_question = AnswerQuestion(search_documents=retriever, chat_model=chat_model, top_k=5)

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
        # variance the model might introduce (e.g. "15-20%" vs "15-20%"),
        # but still anchored to the specific figures/phrases RAG.md states.
        checks = {
            # Q1: Parent Document Retrieval's mechanic and its own expected
            # impact figure -- RAG.md: "The mechanic is to search against
            # small chunks for precision, then map each top result back to
            # its parent document, page, or section before sending anything
            # to the LLM, so retrieval stays precise while the model still
            # receives complete context." ... "Its expected impact in
            # isolation is a +15-20% gain in completeness."
            scenario.questions[0].question: lambda a: (
                ("15-20%" in a.text or "15–20%" in a.text or "15-20" in a.text)
                and "completeness" in a.text.lower()
            ),
            # Q2: Context Compression's two distinct payoff numbers, which
            # RAG.md is explicit shouldn't be conflated -- the -75% worked
            # example (40 chunks down to 8) versus the more conservative
            # general expected-impact figure (-50% tokens, +10% focus).
            # Requiring both "75" and "50" plus "focus" anchors the check to
            # an answer that states both numbers rather than only one --
            # "focus" only shows up alongside the general -50%/+10% figure,
            # not the worked example, so its presence is evidence the answer
            # kept the two numbers distinct rather than conflating them.
            scenario.questions[1].question: lambda a: (
                "75" in a.text and "50" in a.text and "focus" in a.text.lower()
            ),
            # Q3: The "Parent Document + Context Compression" combination's
            # named quality -- RAG.md: "**Parent Document + Context
            # Compression** -- described as 'the perfect balance' for
            # complete yet compact context."
            scenario.questions[2].question: lambda a: ("perfect balance" in a.text.lower()),
            # Q4: The intermediate roadmap phase's payoff from adding Parent
            # Document Retrieval and Context Compression together -- RAG.md:
            # "The source describes the payoff as context quality improving
            # 'dramatically,' with answers becoming more complete and more
            # focused at the same time."
            scenario.questions[3].question: lambda a: (
                "dramatically" in a.text.lower()
                and "complete" in a.text.lower()
                and "focused" in a.text.lower()
            ),
            # Q5: Parent Document Retrieval's recommended use cases --
            # RAG.md: "The source recommends this specifically for very
            # small chunks (100-300 tokens), documents that lean heavily on
            # cross-references, and questions whose answers genuinely need
            # full surrounding context."
            scenario.questions[4].question: lambda a: (
                ("100-300" in a.text or "100–300" in a.text)
                and "cross-reference" in a.text.lower()
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
        success_criterion="see evaluation/scenarios/rag-parent-doc-compression/queries.yaml",
        rag=True, cag=False, mag=False,
        notes=(
            f"strategy={strategy}, corpus=docs/architecture/RAG.md, "
            f"{document.chunk_count} chunks. CAVEAT 1: qualitative judge is "
            f"Ollama/qwen3.5, not Claude (no API credit balance at run time) -- "
            f"same model family judging its own treatment output, a real "
            f"self-grading-bias risk; re-run with ClaudeJudge once credits "
            f"exist before treating these judge scores as final. "
            f"CAVEAT 2 does NOT apply to this batch: unlike "
            f"rag-hybrid-reranking's candidate_k-exceeds-corpus-size caveat, "
            f"ParentDocumentRetriever and CompressingRetriever both request "
            f"exactly top_k from their inner retriever (no widen-then-narrow "
            f"candidate pool), so this run has no equivalent measurement gap."
        ),
        questions=[q.question for q in scenario.questions],
        baseline=baseline,
        treatment=treatment,
        success_check=success_check,
    )

    report_path = Path(f"evaluation/reports/rag-parent-doc-compression-{strategy}.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render(result), encoding="utf-8")
    print(f"Report written to {report_path}")


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
        # No tenant filtering: this script only ever uses one tenant_id per
        # process, so every stored chunk already belongs to the one tenant
        # that could ask for them.
        return self._chunks

    async def get_chunk_by_id(self, chunk_id: uuid.UUID) -> Chunk | None:
        # Simple loop over the same list get_chunks_for_tenant already
        # returns -- ParentDocumentRetriever calls this once per matched
        # child result (to find its parent_id) and once more per distinct
        # parent (to fetch that parent's content), so it's on the hot path
        # for the parent-document and parent-document-compression
        # strategies. This process's chunk count (a few hundred at most for
        # this corpus) makes an O(n) scan fine for a one-shot manual script;
        # a dict keyed by id would only matter at a scale this harness
        # doesn't operate at.
        return next((c for c in self._chunks if c.id == chunk_id), None)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    asyncio.run(_run(sys.argv[1]))
