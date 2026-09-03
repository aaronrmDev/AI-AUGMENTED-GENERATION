"""Manual comparison runner for the RAG Hybrid Search + Reranking batch. Not
pytest-collected (no test_* function at module level) -- mirrors this
project's existing test_rag_smoke.py / test_docker_compose_smoke.py pattern
for scripts that make real model calls and are run by hand.

Usage:
    uv run python evaluation/scenarios/rag-hybrid-reranking/\
run_hybrid_reranking_comparison.py <strategy>

<strategy> is one of: hybrid, rerank-cross-encoder, rerank-bi-encoder,
rerank-llm, hybrid-rerank-cross-encoder
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
from src.rag.domain.ports import ChatModel, DocumentRepository, EmbeddingModel, Retriever
from src.rag.infrastructure.bi_encoder_rerank_reranker import BiEncoderRerankReranker
from src.rag.infrastructure.bm25_keyword_search import BM25KeywordSearch
from src.rag.infrastructure.cross_encoder_reranker import CrossEncoderReranker
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.hybrid_search_documents import HybridSearchDocuments
from src.rag.infrastructure.llm_reranker import LLMReranker
from src.rag.infrastructure.local_file_storage import LocalFileStorage
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore
from src.rag.infrastructure.reranking_retriever import RerankingRetriever
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.rag.infrastructure.text_extractor import TextExtractor

_SCENARIO_DIR = Path(__file__).parent
_MODEL_CONFIG = "qwen3.5, Ollama"


def _make_retriever(
    name: str,
    vector_retriever: Retriever,
    document_repository: DocumentRepository,
    embedder: EmbeddingModel,
    chat_model: ChatModel,
) -> Retriever:
    if name == "hybrid":
        return HybridSearchDocuments(
            vector_retriever=vector_retriever,
            keyword_retriever=BM25KeywordSearch(document_repository=document_repository),
        )
    if name == "rerank-cross-encoder":
        return RerankingRetriever(inner=vector_retriever, reranker=CrossEncoderReranker())
    if name == "rerank-bi-encoder":
        return RerankingRetriever(
            inner=vector_retriever, reranker=BiEncoderRerankReranker(embedding_model=embedder)
        )
    if name == "rerank-llm":
        return RerankingRetriever(
            inner=vector_retriever, reranker=LLMReranker(chat_model=chat_model)
        )
    if name == "hybrid-rerank-cross-encoder":
        hybrid = HybridSearchDocuments(
            vector_retriever=vector_retriever,
            keyword_retriever=BM25KeywordSearch(document_repository=document_repository),
        )
        return RerankingRetriever(inner=hybrid, reranker=CrossEncoderReranker())
    raise ValueError(
        f"unknown strategy {name!r}; choose one of "
        f"['hybrid', 'rerank-cross-encoder', 'rerank-bi-encoder', 'rerank-llm', "
        f"'hybrid-rerank-cross-encoder']"
    )


async def _run(strategy: str) -> None:
    scenario = load_scenario(_SCENARIO_DIR)
    embedder = SentenceTransformersEmbedder()
    vector_store = QdrantVectorStore("http://localhost:6333")
    await vector_store.ensure_collection()
    # Chunker stays fixed across every strategy in this batch -- chunking
    # itself already got its own comparison batch (rag-chunking-strategies);
    # what varies here is retrieval/reranking, not chunk boundaries.
    chunker = FixedSizeChunker()

    ollama_client = ollama.AsyncClient(host="http://localhost:11434")
    chat_model = OllamaChatModel(client=ollama_client, model_id="qwen3.5")

    tenant_id = uuid.uuid4()
    # Single _InMemoryDocumentRepository instance, reused for both the upload
    # path and the BM25/Hybrid strategies' document_repository param. Unlike
    # the chunking-strategies scenario's _NullDocumentRepository (which this
    # script started from), save_chunks here actually keeps what it's given
    # -- BM25KeywordSearch's whole job is reading real chunk content back via
    # get_chunks_for_tenant, and a null store would silently zero out the
    # keyword-search arm of every 'hybrid' strategy. A real
    # PostgresDocumentRepository-backed session was considered and rejected
    # for this harness: this script's single process, single tenant_id, one-
    # shot run has no need for real persistence across processes, and adding
    # Postgres session plumbing here would only be exercising infrastructure
    # already covered by tests/integration/test_postgres_document_repository.py,
    # not anything this measurement needs.
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

    vector_retriever = SearchDocuments(embedding_model=embedder, vector_store=vector_store)
    retriever = _make_retriever(
        strategy, vector_retriever, document_repository, embedder, chat_model
    )
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
        # variance the model might introduce (e.g. "15-25%" vs "15–25%"),
        # but still anchored to the specific figures/phrases RAG.md states.
        checks = {
            # Q1: How Hybrid Search merges its two result sets -- RAG.md:
            # "Hybrid Search runs both in parallel -- vector search returns
            # its top results, BM25 returns its top results -- and merges
            # the two result sets, typically via Reciprocal Rank Fusion
            # (RRF)".
            scenario.questions[0].question: lambda a: (
                "reciprocal rank fusion" in a.text.lower() or "rrf" in a.text.lower()
            ),
            # Q2: The three reranker types' accuracy/speed/cost tradeoff --
            # RAG.md's Concept 1 table: Cross-Encoder=High/Slower/Medium,
            # Bi-Encoder + Rerank Model=Balanced/Balanced/Balanced,
            # LLM-based Reranker=Highest/Slowest/Costliest.
            scenario.questions[1].question: lambda a: (
                "cross-encoder" in a.text.lower()
                and "bi-encoder" in a.text.lower()
                and ("llm-based" in a.text.lower() or "llm based" in a.text.lower())
            ),
            # Q3: Reranking's expected impact figure in isolation -- RAG.md:
            # "Reranking's own expected impact, in isolation, is a +15-25%
            # gain in answer accuracy".
            scenario.questions[2].question: lambda a: (
                ("15-25%" in a.text or "15–25%" in a.text or "15-25" in a.text)
                and "accuracy" in a.text.lower()
            ),
            # Q4: The Hybrid Search + Reranking combination's own named
            # quality -- RAG.md: "Hybrid Search + Reranking -- the source
            # calls this 'the production standard.'"
            scenario.questions[3].question: lambda a: (
                "production standard" in a.text.lower()
            ),
            # Q5: Hybrid Search's own expected impact figure in isolation --
            # RAG.md: "Its expected impact in isolation is a +20-30% recall
            # gain."
            # M6 fix: RAG.md states an identical "+20-30% recall gain" figure
            # for HyDE too (a different technique, same corpus) -- checking
            # only "20-30" and "recall" would also score a HyDE-flavored
            # answer as a success. Requiring "hybrid" too anchors the check
            # to the technique this question actually asks about.
            scenario.questions[4].question: lambda a: (
                ("20-30%" in a.text or "20–30%" in a.text or "20-30" in a.text)
                and "recall" in a.text.lower()
                and "hybrid" in a.text.lower()
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
    # candidate_k (20, set on every Retriever/RerankingRetriever/
    # HybridSearchDocuments instance above) exceeds this corpus's own chunk
    # count (14 with FixedSizeChunker on docs/architecture/RAG.md) -- so
    # every retrieval arm in every strategy here always returns the whole
    # corpus, and "widen the pool, then narrow" never actually narrows
    # anything upstream of reranking/merging. Disclosed rather than
    # discovered later: this measures reordering/reranking over the full
    # corpus, not the recall-union or candidate-filtering properties Hybrid
    # Search and Reranking are meant to demonstrate at real corpus scale.
    candidate_pool_note = (
        "CAVEAT 2: candidate_k=20 exceeds this corpus's 14 chunks, so every "
        "retrieval arm always returns the full corpus in this run -- Hybrid "
        "Search's recall-union property and Reranking's candidate-narrowing "
        "were not exercised at the scale they're meant for; this measures "
        "reordering over the whole corpus, not filtering a wider pool."
    )
    llm_reranker_note = (
        " CAVEAT 3: this strategy's latency is inflated by qwen3.5's "
        "reasoning mode -- each scoring call generates ~1.4-2.8K hidden "
        "thinking tokens that OllamaChatModel discards before returning "
        "just the score. The technique-cost ranking (LLM reranking costliest "
        "of the three) is real; the specific magnitude is not a clean "
        "property of LLM reranking as a technique, it's this model's "
        "reasoning overhead paid 14 times."
        if strategy == "rerank-llm"
        else ""
    )
    result = await use_case.execute(
        scenario_name=scenario.name,
        model_config=_MODEL_CONFIG,
        success_criterion="see evaluation/scenarios/rag-hybrid-reranking/queries.yaml",
        rag=True, cag=False, mag=False,
        notes=(
            f"strategy={strategy}, corpus=docs/architecture/RAG.md, "
            f"{document.chunk_count} chunks. CAVEAT 1: qualitative judge is "
            f"Ollama/qwen3.5, not Claude (no API credit balance at run time) -- "
            f"same model family judging its own treatment output, a real "
            f"self-grading-bias risk; re-run with ClaudeJudge once credits "
            f"exist before treating these judge scores as final. CAVEAT 1b: "
            f"baseline has no retrieved context of its own (empty string), so "
            f"per the #148 fix (each arm judged against its own context) "
            f"baseline's groundedness is judged against '(none provided)' -- "
            f"treat any baseline-vs-treatment groundedness gap as reflecting "
            f"that structural difference in what each arm had to work with, "
            f"not a measured claim about factual accuracy. "
            f"{candidate_pool_note}{llm_reranker_note}"
        ),
        questions=[q.question for q in scenario.questions],
        baseline=baseline,
        treatment=treatment,
        success_check=success_check,
    )

    report_path = Path(f"evaluation/reports/rag-hybrid-reranking-{strategy}.md")
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
        # DocumentRepository gained this abstract method after this script
        # was first written (rag-parent-doc-compression batch,
        # ParentDocumentRetriever's parent lookup) -- this scenario never
        # uses it (no parent-document strategy here), so a lookup that can
        # only ever return None keeps this class instantiable without
        # changing anything this scenario's own strategies actually measure.
        return next((c for c in self._chunks if c.id == chunk_id), None)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    asyncio.run(_run(sys.argv[1]))
