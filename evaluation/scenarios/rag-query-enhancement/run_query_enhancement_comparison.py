"""Manual comparison runner for the RAG Multi-Query + HyDE + Self-RAG batch.
Not pytest-collected (no test_* function at module level) -- mirrors this
project's existing test_rag_smoke.py / test_docker_compose_smoke.py pattern
for scripts that make real model calls and are run by hand.

Named run_query_enhancement_comparison.py, not run_comparison.py: the
rag-hybrid-reranking and rag-parent-doc-compression batches (both already
merged to develop) each broke `mypy evaluation/` because a scenario script
collided with another scenario script's bare module name -- neither
hyphenated scenario directory (evaluation/scenarios/rag-*) can be a Python
package, so mypy sees any two files both named run_comparison.py as the same
top-level module. Every scenario's entrypoint needs a name unique across the
whole evaluation/scenarios/ tree.

Usage (PYTHONPATH=. is required -- these imports resolve relative to the
repo root, which "uv run python <path>" does not add to sys.path on its
own):
    PYTHONPATH=. uv run python evaluation/scenarios/rag-query-enhancement/\
run_query_enhancement_comparison.py <strategy>

<strategy> is one of: multi-query, hyde, self-rag
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
from src.rag.domain.entities import ChatAnswer, Chunk, Document
from src.rag.domain.ports import ChatModel, DocumentRepository, Retriever
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.hyde_retriever import HyDERetriever
from src.rag.infrastructure.local_file_storage import LocalFileStorage
from src.rag.infrastructure.multi_query_retriever import MultiQueryRetriever
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.rag.infrastructure.text_extractor import TextExtractor

_SCENARIO_DIR = Path(__file__).parent
_MODEL_CONFIG = "qwen3.5, Ollama"

_STRATEGIES = ("multi-query", "hyde", "self-rag")


class _Answerer(Protocol):
    # AnswerQuestion and SelfRAGAnswerQuestion share this exact call shape
    # (see src/rag/application/answer_question.py and
    # src/rag/application/self_rag_answer_question.py) but have no common
    # base class -- SelfRAGAnswerQuestion is a sibling to AnswerQuestion, not
    # a Retriever decorator, because "skip retrieval entirely" isn't
    # expressible through the Retriever port. A structural Protocol lets
    # treatment() below call whichever one is active through one typed name
    # without inventing a shared ABC neither production class actually needs.
    async def execute(self, tenant_id: uuid.UUID, question: str) -> ChatAnswer: ...


def _make_answerer(
    name: str, vector_retriever: Retriever, chat_model: ChatModel, top_k: int
) -> _Answerer:
    # multi-query and hyde: build a Retriever decorator (MultiQueryRetriever /
    # HyDERetriever), wired into a plain AnswerQuestion exactly like every
    # other batch's strategies.
    if name == "multi-query":
        retriever: Retriever = MultiQueryRetriever(inner=vector_retriever, chat_model=chat_model)
        return AnswerQuestion(search_documents=retriever, chat_model=chat_model, top_k=top_k)
    if name == "hyde":
        retriever = HyDERetriever(inner=vector_retriever, chat_model=chat_model)
        return AnswerQuestion(search_documents=retriever, chat_model=chat_model, top_k=top_k)
    # self-rag: not a Retriever at all -- construct SelfRAGAnswerQuestion
    # directly over the plain vector SearchDocuments, skipping AnswerQuestion
    # entirely, since SelfRAGAnswerQuestion already owns the "search, then
    # generate" flow (and, on a NO gate, skips the search step altogether).
    if name == "self-rag":
        return SelfRAGAnswerQuestion(
            search_documents=vector_retriever, chat_model=chat_model, top_k=top_k
        )
    raise ValueError(f"unknown strategy {name!r}; choose one of {list(_STRATEGIES)}")


async def _run(strategy: str) -> None:
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose one of {list(_STRATEGIES)}")

    scenario = load_scenario(_SCENARIO_DIR)
    embedder = SentenceTransformersEmbedder()
    vector_store = QdrantVectorStore("http://localhost:6333")
    await vector_store.ensure_collection()
    # Chunker stays fixed and plain across every strategy in this batch --
    # none of Multi-Query, HyDE, or Self-RAG touch chunking or need the
    # two-tier parent index (unlike the rag-parent-doc-compression batch);
    # all three operate purely on what gets searched for / whether to search
    # at all, upstream or in place of a plain vector retrieval call.
    chunker = FixedSizeChunker()

    ollama_client = ollama.AsyncClient(host="http://localhost:11434")
    chat_model = OllamaChatModel(client=ollama_client, model_id="qwen3.5")

    tenant_id = uuid.uuid4()
    # Single _InMemoryDocumentRepository instance, used only to satisfy
    # UploadDocument's required document_repository param. Unlike the
    # rag-hybrid-reranking (BM25) or rag-parent-doc-compression (parent
    # lookups) batches, none of this batch's three strategies read chunks
    # back out of the document repository -- MultiQueryRetriever and
    # HyDERetriever only wrap a Retriever + ChatModel, and
    # SelfRAGAnswerQuestion only wraps a Retriever + ChatModel too. A real
    # PostgresDocumentRepository-backed session was considered and rejected
    # for the same reason as the prior two batches: this script's single
    # process, single tenant_id, one-shot run has no need for real
    # persistence across processes, and that infrastructure is already
    # covered by tests/integration/test_postgres_document_repository.py.
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
    answerer = _make_answerer(strategy, vector_retriever, chat_model, top_k=5)

    async def baseline(question: str) -> Answer:
        text = await chat_model.generate(question=question, context="")
        return Answer(text=text, input_tokens=0, output_tokens=0, context="")

    # Populated only inside treatment() below, for the self-rag strategy --
    # closed over here rather than built into the notes= keyword argument
    # passed to RunComparison.execute() directly. That argument is evaluated
    # BEFORE execute() ever calls treatment(), so anything computed only
    # inside treatment() would be empty/stale if read at that point -- a real
    # ordering bug the rag-parent-doc-compression batch's controller had to
    # catch and fix in exactly this scenario. Read only AFTER execute()
    # returns, via dataclasses.replace() on the frozen ComparisonResult
    # (same pattern that batch's script uses for context_token_counts).
    self_rag_gate_log: list[str] = []

    async def treatment(question: str) -> Answer:
        result = await answerer.execute(tenant_id=tenant_id, question=question)
        # Reconstructs exactly what AnswerQuestion/SelfRAGAnswerQuestion
        # themselves joined internally to build the chat model's context --
        # it isn't returned on ChatAnswer, so this is the only way to get it
        # back out for the judge without changing either use case's return
        # contract.
        retrieved_context = "\n\n".join(source.content for source in result.sources)
        if strategy == "self-rag":
            # SelfRAGAnswerQuestion doesn't return its gate decision on
            # ChatAnswer -- the minimal, non-invasive way to observe it from
            # outside (without modifying SelfRAGAnswerQuestion itself just to
            # expose more) is to re-derive it from what the gate's own
            # behavior implies: an empty sources list means the gate said NO
            # and skipped retrieval entirely; a non-empty one means it said
            # YES (see src/rag/application/self_rag_answer_question.py --
            # the NO branch returns ChatAnswer(answer=..., sources=[])).
            gate_decision = "YES/retrieved" if len(result.sources) > 0 else "NO/skipped"
            print(f"[self-rag gate] {question!r} -> {gate_decision}")
            self_rag_gate_log.append(gate_decision)
        return Answer(
            text=result.answer, input_tokens=0, output_tokens=0, context=retrieved_context
        )

    def success_check(question: str, answer: Answer) -> bool:
        # Each check is a direct encoding of the success_criterion written for
        # that question in queries.yaml -- both were derived from the same
        # reading of the real, current docs/architecture/RAG.md (the 5
        # grounded questions), or need no document lookup at all (the final 2
        # general-knowledge questions). Matching is case-insensitive
        # substring matching, tolerant of minor wording variance the model
        # might introduce, but still anchored to the specific figures/
        # phrases/answers each question is checking for.
        checks = {
            # Q1: Multi-Query Retrieval's own distinguishing mechanic vs.
            # Query Expansion, and its own recall gain figure -- RAG.md:
            # "Multi-Query Retrieval instead has an LLM generate several
            # genuinely different framings of the underlying information
            # need -- not just reworded, but viewed from different angles --
            # and searches each one independently before merging and
            # reranking the combined results." ... "its recall gain
            # (+25-35%, per the source's Category 1 figures) sits noticeably
            # above Query Expansion's more modest 'good.'"
            scenario.questions[0].question: lambda a: (
                ("25-35%" in a.text or "25–35%" in a.text or "25-35" in a.text)
                and ("differen" in a.text.lower() or "diverse" in a.text.lower()
                     or "angle" in a.text.lower())
            ),
            # Q2: HyDE's own mechanic (what actually gets embedded) and its
            # own expected impact figure -- RAG.md: "HyDE has the LLM
            # generate a hypothetical answer to the question first, and it is
            # *that generated answer's embedding* -- not the original
            # question's embedding -- that gets used to search the vector
            # database" ... "Its expected impact is a +20-30% recall gain
            # specifically for vague queries."
            scenario.questions[1].question: lambda a: (
                ("20-30%" in a.text or "20–30%" in a.text or "20-30" in a.text)
                and "hypothetical" in a.text.lower()
            ),
            # Q3: Self-RAG's own mechanic (the YES/NO gate) and its own
            # expected impact figure -- RAG.md: "**Self-RAG** is the gate:
            # instead of retrieving for every query unconditionally, the LLM
            # first checks its own knowledge and decides YES or NO." ...
            # "the source's expected impact is a 30-50% reduction in
            # retrieval costs, purely from not paying for lookups the model
            # didn't need."
            scenario.questions[2].question: lambda a: (
                ("30-50%" in a.text or "30–50%" in a.text or "30-50" in a.text)
                and "gate" in a.text.lower()
            ),
            # Q4: The Multi-Query + HyDE pairing's own named scoring, among
            # RAG.md's "Top 5 Most Impactful Pairs" -- RAG.md: "**Multi-Query
            # + HyDE** -- scored for 'maximum recall for vague queries.'"
            scenario.questions[3].question: lambda a: (
                "maximum recall" in a.text.lower() and "vague" in a.text.lower()
            ),
            # Q5: The "Fort Knox" pattern's six chained concepts and what
            # RAG.md says it buys -- RAG.md: "its 'Fort Knox' pattern chains
            # six concepts together -- Hybrid Search, Multi-Query, HyDE,
            # Reranking, CRAG, and Parent Document -- for what it describes
            # as 'unbeatable accuracy' at the cost of high latency and high
            # cost."
            scenario.questions[4].question: lambda a: ("unbeatable accuracy" in a.text.lower()),
            # Q6: general knowledge, no RAG.md lookup needed -- exists to give
            # the self-rag strategy's gate a real chance to say NO.
            scenario.questions[5].question: lambda a: ("42" in a.text),
            # Q7: general knowledge, no RAG.md lookup needed -- exists to give
            # the self-rag strategy's gate a real chance to say NO.
            scenario.questions[6].question: lambda a: (
                "application programming interface" in a.text.lower()
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

    # DISCLOSURE, proactive (matching the rag-hybrid-reranking and
    # rag-parent-doc-compression batches' pattern of flagging a real cost
    # before a review has to point it out): multi-query and hyde both make
    # ONE EXTRA LLM call per query -- generating query variants, or a
    # hypothetical answer -- before retrieval even runs, on top of the one
    # generate() call every strategy already pays for the final answer. The
    # rag-hybrid-reranking batch's rerank-llm finding already showed that an
    # extra reasoning-model call in the loop can add substantial latency
    # (observed ~1.4-2.8K hidden thinking tokens per call on qwen3.5's
    # reasoning mode); expect the same class of effect here, proportional to
    # each strategy's one extra call. self-rag pays a smaller but real
    # version of the same cost: its gate check is its own extra generate()
    # call on every single question, win or lose -- even a NO decision that
    # skips retrieval still paid for that one gate call first.
    extra_llm_call_note = (
        " CAVEAT 2: this strategy issues one extra LLM call per query before "
        "retrieval even runs (query-variant generation for multi-query, "
        "hypothetical-answer generation for hyde), on top of the one "
        "generate() call every strategy already pays for the final answer -- "
        "the rag-hybrid-reranking batch's rerank-llm finding already showed "
        "an extra reasoning-model call in the loop can add substantial "
        "latency; expect the same class of effect here."
        if strategy in ("multi-query", "hyde")
        else (
            " CAVEAT 2: self-rag's gate check is itself an extra generate() "
            "call on every single question, independent of what the gate "
            "decides -- even a NO decision that skips retrieval entirely "
            "still paid for that one gate call first, so self-rag is not "
            "free relative to a plain AnswerQuestion baseline on questions "
            "where it ends up retrieving anyway; its savings are specifically "
            "in the retrieval + context-assembly + larger-context generation "
            "work it skips on a NO, not in avoiding LLM calls altogether."
            if strategy == "self-rag"
            else ""
        )
    )
    result = await use_case.execute(
        scenario_name=scenario.name,
        model_config=_MODEL_CONFIG,
        success_criterion="see evaluation/scenarios/rag-query-enhancement/queries.yaml",
        rag=True, cag=False, mag=False,
        notes=(
            f"strategy={strategy}, corpus=docs/architecture/RAG.md, "
            f"{document.chunk_count} chunks. CAVEAT 1: qualitative judge is "
            f"Ollama/qwen3.5, not Claude (no API credit balance at run time) -- "
            f"same model family judging its own treatment output, a real "
            f"self-grading-bias risk; re-run with ClaudeJudge once credits "
            f"exist before treating these judge scores as final."
            f"{extra_llm_call_note}"
        ),
        questions=[q.question for q in scenario.questions],
        baseline=baseline,
        treatment=treatment,
        success_check=success_check,
    )

    if strategy == "self-rag":
        # self_rag_gate_log is only fully populated once every treatment()
        # call has actually run -- which happens inside execute() above, not
        # before it (see the comment on self_rag_gate_log's declaration).
        # Appended via dataclasses.replace on the frozen result, same pattern
        # the rag-parent-doc-compression batch's script uses for
        # context_token_counts.
        yes_count = sum(1 for g in self_rag_gate_log if g.startswith("YES"))
        no_count = len(self_rag_gate_log) - yes_count
        gate_note = (
            f" SELF-RAG GATE LOG: across this run's {len(self_rag_gate_log)} "
            f"treatment calls (repeat_count=3 x 7 questions), the gate said "
            f"YES/retrieved {yes_count} times and NO/skipped {no_count} "
            f"times. This is only the aggregate count -- the real, honest, "
            f"question-by-question gate decisions were printed to stdout as "
            f"'[self-rag gate] <question> -> <decision>' while this run "
            f"executed; read that output directly rather than treating this "
            f"aggregate as the full record."
        )
        result = dataclasses.replace(result, notes=result.notes + gate_note)

    report_path = Path(f"evaluation/reports/rag-query-enhancement-{strategy}.md")
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
        # None of this batch's three strategies call this (no parent-document
        # lookups here) -- implemented anyway to keep this class instantiable
        # against the real DocumentRepository ABC, same as the
        # rag-hybrid-reranking batch's equivalent class.
        return next((c for c in self._chunks if c.id == chunk_id), None)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    asyncio.run(_run(sys.argv[1]))
