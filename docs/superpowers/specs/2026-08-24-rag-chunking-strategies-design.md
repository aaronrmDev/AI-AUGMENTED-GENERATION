# RAG Chunking Strategies (Batch 1) — Design Spec

## Why this exists

`docs/architecture/RAG.md` documents six chunking strategies (issue #48's six sub-tasks: #77 Fixed Size, #82 Sentence-Based, #84 Semantic, #85 Parent Document, #87 Sliding Window, #88 Structure-Aware). Fixed Size (#77) already exists as `FixedSizeChunker`, built during the RAG Pipeline sub-project, but has never been measured against RAG.md's own claimed effect ("+10-20% gain in relevance, isolation figure") — every one of these six issues' Definition of Done requires a real measured result, not just an implementation, and the Evaluation Harness now exists to produce one.

## Scope

**In this batch:** implement Sentence-Based (#82), Semantic (#84), Sliding Window (#87), and Structure-Aware (#88); measure all five strategies that exist after this batch (the four new ones plus the already-built Fixed Size) against a shared real corpus using the Evaluation Harness; post each result to its GitHub issue.

**Deferred, not in this batch: Parent Document Chunking (#85).** Unlike the other five, it doesn't produce a flat list of same-purpose chunks — RAG.md describes it as keeping retrieval units small while storing a pointer to a larger parent section, which means two chunk tiers (searchable children, larger stored-but-not-necessarily-searched parents) and populating the `Chunk.parent_id` column that has existed unused in the schema since RAG Pipeline. It also has a structural dependency this batch doesn't resolve: RAG.md's own framing is that Parent Document Chunking's *effect* shows up at retrieval time, when a matched child chunk gets mapped back to its parent before reaching the LLM (Parent Document *Retrieval*, issues #56/#105 — Post-Retrieval stage, not yet built). Measuring #85 in isolation, without its retrieval-side companion, would show no meaningful quality delta — not because the technique doesn't work, but because nothing downstream reads `parent_id` yet. Building it now would mean either a fake/misleading measured result or documenting "no effect, needs #56/#105" as this batch's own finding for a technique this batch didn't fully build. Cleaner to defer #85 as its own later piece of work, sized and measured together with #56/#105 once both exist.

## A new `Chunker` port

Five chunker implementations are about to exist and all need to be swappable behind the Evaluation Harness's baseline-vs-treatment comparison — this is exactly the moment the RAG Pipeline final review flagged as when a `Chunker` port should stop being deferred ("the application layer imports concrete infrastructure... worth recording so the next paradigm module doesn't inherit the ambiguity"). Add to `src/rag/domain/ports.py`:

```python
class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]: ...
```

This is `FixedSizeChunker.chunk`'s existing exact signature — making it implement the port costs nothing beyond adding `(Chunker)` to its class declaration. `UploadDocument.__init__`'s `chunker: FixedSizeChunker` parameter type narrows to `chunker: Chunker`, and nothing else about `UploadDocument` changes, since every strategy in this batch (as scoped above) returns a flat `list[str]`.

## The four new chunkers

All four live in `src/rag/infrastructure/`, implement `Chunker`, and use `tiktoken`'s `cl100k_base` encoding for token counting — the same approximate-but-consistent convention `FixedSizeChunker` already established.

**`SentenceBasedChunker`** (#82) — splits text into sentences via a regex on sentence-ending punctuation (`.`, `!`, `?`) followed by whitespace and a capital letter or end-of-string, then greedily accumulates consecutive sentences into a chunk until adding the next sentence would exceed `chunk_size_tokens` (default 512), at which point it starts a new chunk. A single sentence longer than `chunk_size_tokens` on its own becomes its own chunk rather than being force-split (splitting mid-sentence is exactly what this strategy exists to avoid). No token-level overlap between chunks — RAG.md's own tradeoff description for this strategy doesn't mention overlap, only boundary preservation.

**`SemanticChunker`** (#84) — splits into sentences with the same regex as `SentenceBasedChunker` (reused, not reimplemented), embeds each sentence with the already-built `SentenceTransformersEmbedder`, computes cosine similarity between every pair of *consecutive* sentence embeddings, and finds breakpoints where that similarity drops more than one standard deviation below the document's own mean consecutive-similarity (adaptive to each document's own similarity distribution, the same intent the original 25th-percentile design had — but a pure bottom-quartile rule turned out to have no floor: the final whole-branch review proved it force-splits ANY document, including a maximally cohesive one where every similarity is identical, because a quantile always has a bottom 25% even when nothing in it is actually low; on the real measured corpus the quantile rule contributed exactly its arithmetic floor of splits and zero content-dependent signal. Gating on distance from the document's own mean fixes this: with zero variance the threshold equals the mean, and a value can never be strictly less than a constant it's tied with, so a maximally cohesive document now correctly produces zero semantic breakpoints). Sentences between two breakpoints become one chunk. This is the strategy RAG.md itself describes as needing an embedding model to do the grouping — a community comment on issue #84 suggested a lighter lexical-cohesion alternative requiring no embedding model, which is a legitimate implementation choice for a different technique, but this batch builds what RAG.md actually documents as Semantic Chunking, reusing infrastructure this project already has.

**`SlidingWindowChunker`** (#87) — mechanically the same algorithm `FixedSizeChunker` already implements (a fixed-size window walked across the token stream with overlap), configured with a much higher default overlap ratio (0.5 instead of 0.1). RAG.md's own description of the difference between these two strategies is a difference of degree and intent, not a different algorithm: Fixed Size uses "some overlap" as a boundary-softening measure, Sliding Window's overlap is the point ("gives good coverage and keeps narrative flow intact... multiplies how much has to be stored since the windows overlap"). Implemented as a thin subclass:

```python
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker


class SlidingWindowChunker(FixedSizeChunker):
    def __init__(self, chunk_size_tokens: int = 512, overlap_ratio: float = 0.5) -> None:
        super().__init__(chunk_size_tokens, overlap_ratio)
```

Documented plainly as what it is — not a separately-invented algorithm dressed up as one, since pretending otherwise would misrepresent what's actually being measured against RAG.md's claim.

**`StructureAwareChunker`** (#88) — splits primarily on Markdown heading boundaries (`#` through `######`), keeping each heading's content together as one candidate chunk; a section that exceeds `chunk_size_tokens` gets further split using `SentenceBasedChunker`'s own sentence-accumulation logic (reused via composition) so a long section still doesn't cut mid-sentence. Fenced code blocks (triple-backtick delimited) are treated as atomic — never split internally, and if a code block plus its surrounding paragraph would together exceed the size limit, the code block is still kept whole and the limit is allowed to be exceeded for that one chunk (a broken code block is a worse failure than a slightly oversized chunk). Documents with no heading structure at all fall back to `SentenceBasedChunker`'s behavior entirely — Structure-Aware only works as well as the parser recognizing structure, and a structure-less document has none to recognize, matching RAG.md's own stated caveat for this strategy.

## Scenario and measurement

A new scenario directory `evaluation/scenarios/rag-chunking-strategies/` — `corpus/` holds a copy of `docs/architecture/RAG.md` itself (real, already-existing, structured Markdown: headings, prose of varying sentence length, topic shifts between sections, no fabricated fixture content needed), and `queries.yaml` holds 5 questions whose answers are genuinely present in that document, each with a stated success criterion (a specific fact or figure the answer must contain — checkable by substring match, matching the harness's existing `success_check: Callable[[str, Answer], bool]` shape).

For each of the five chunking strategies (Fixed Size included, since it's never been measured despite already being built), a small comparison script under `evaluation/scenarios/rag-chunking-strategies/run_<strategy-slug>.py` wires `RunComparison`: baseline = `ChatModel.generate(question, context="")` (no retrieval at all — RAG completely off, matching `quantitative-template.md`'s literal baseline definition), treatment = `AnswerQuestion` with `UploadDocument` using that one chunker uploading the corpus first. `rag=True, cag=False, mag=False` for every run in this batch (chunking is exclusively a RAG-stage concern). Each script's `render()` output goes to `evaluation/reports/rag-chunking-<strategy-slug>.md` (committed) and its `render_github_comment()` output gets posted by hand to that strategy's GitHub issue once the run is judged trustworthy, per the harness's own established human-in-the-loop posting step.

Chat model for these runs: Ollama/Qwen3.5, per the project's own sanctioned use of Ollama for development-time proofs (`docs/architecture/OVERVIEW.md`'s "When you might deviate from this stack" section) — these are the first real exercises of both the Evaluation Harness and the `CHAT_PROVIDER=ollama` switch built earlier, at zero API cost.

## Testing strategy

Each of the four new chunkers gets unit tests mirroring `FixedSizeChunker`'s own test shape: a short-text case (single chunk), a long-text case (multiple chunks, boundary behavior specific to that strategy — sentence boundaries never split, semantic breakpoints land where similarity actually drops, sliding window chunks overlap more than fixed-size's default, structure-aware keeps a heading's content together and never splits a fenced code block), and an empty-text case. `SemanticChunker`'s tests use a real `SentenceTransformersEmbedder` (integration-tier, matching how `test_sentence_transformers_embedder.py` already handles the real-model-load cost) rather than a fake, since the whole point of this strategy is genuine semantic grouping — a fake embedder would prove nothing about whether the breakpoint heuristic actually works. `SlidingWindowChunker` gets a light test suite of its own (constructor defaults, that it's genuinely a `FixedSizeChunker` subclass) rather than re-testing inherited chunking logic already covered by `FixedSizeChunker`'s own tests.

## Non-goals

No Parent Document Chunking (#85) — deferred, reasoning above. No changes to `UploadDocument`'s chunk-representation shape (stays `list[str]`) — that only needs to change for #85, which is out of scope here. No vLLM-on-ROCm wiring — Ollama is the chat model for every comparison in this batch, consistent with its sanctioned development-time role.
