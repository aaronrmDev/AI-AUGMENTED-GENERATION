# RAG Chunking Strategies (Batch 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement four new RAG chunking strategies (Sentence-Based #82, Semantic #84, Sliding Window #87, Structure-Aware #88), introduce a `Chunker` port so all five strategies (including the already-built Fixed Size #77) are swappable, and produce a real measured result for all five against a shared corpus using the Evaluation Harness — closing each of five GitHub issues' "real measured result documented" DoD item.

**Architecture:** New `Chunker` port in `src/rag/domain/ports.py`; four new adapters in `src/rag/infrastructure/`; a new evaluation scenario under `evaluation/scenarios/rag-chunking-strategies/`; five small comparison scripts using the Evaluation Harness's `RunComparison`.

**Tech Stack:** Everything already in this project. No new dependencies.

**Spec:** docs/superpowers/specs/2026-08-24-rag-chunking-strategies-design.md

## Global Constraints

- `Chunker` port signature is exactly `chunk(self, text: str) -> list[str]` — matches `FixedSizeChunker.chunk`'s existing signature verbatim, so making it implement the port requires no changes to `FixedSizeChunker`'s method itself.
- All chunkers use `tiktoken`'s `cl100k_base` encoding for token counting, the same convention `FixedSizeChunker` already uses.
- `SlidingWindowChunker` is a subclass of `FixedSizeChunker`, not a reimplementation — it is honestly the same algorithm at a different default overlap ratio (0.5 instead of 0.1), and the code and its docstring must say so plainly.
- `StructureAwareChunker` never splits inside a fenced code block, even if that means exceeding `chunk_size_tokens` for that one chunk.
- `SemanticChunker`'s tests use the real `SentenceTransformersEmbedder` (integration tier) — a fake embedder would prove nothing about whether the breakpoint heuristic works.
- Every comparison script in this batch uses `CHAT_PROVIDER=ollama` / `qwen3.5` (via `OllamaChatModel`) and `rag=True, cag=False, mag=False`.
- Every commit follows the `.gitmessage` template and Conventional Commits format, footer referencing the specific GitHub issue each task closes work toward.

---

### Task 1: Chunker port and sentence-splitting helper

**Files:**
- Modify: `src/rag/domain/ports.py` (add `Chunker` ABC)
- Modify: `src/rag/infrastructure/fixed_size_chunker.py` (implement the port)
- Modify: `src/rag/application/upload_document.py` (narrow the `chunker` parameter's type annotation)
- Create: `src/rag/infrastructure/_sentence_splitter.py` (shared regex-based sentence splitter, used by Tasks 2 and 4)
- Test: `tests/unit/test_sentence_splitter.py`

**Interfaces:**
- Produces: `Chunker` port; `split_sentences(text: str) -> list[str]`, consumed by Task 2's `SentenceBasedChunker`, Task 3's `SemanticChunker`, and Task 4's `StructureAwareChunker`.

- [ ] **Step 1: Write the failing sentence-splitter tests**

```python
# tests/unit/test_sentence_splitter.py
from src.rag.infrastructure._sentence_splitter import split_sentences


def test_split_sentences_splits_on_terminal_punctuation():
    sentences = split_sentences("First sentence. Second sentence! Third sentence?")
    assert sentences == ["First sentence.", "Second sentence!", "Third sentence?"]


def test_split_sentences_handles_a_single_sentence():
    assert split_sentences("Just one sentence.") == ["Just one sentence."]


def test_split_sentences_handles_empty_text():
    assert split_sentences("") == []


def test_split_sentences_does_not_split_on_a_decimal_number():
    sentences = split_sentences("The value is 3.14 and it matters. Next sentence.")
    assert sentences == ["The value is 3.14 and it matters.", "Next sentence."]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_sentence_splitter.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/rag/infrastructure/_sentence_splitter.py`**

```python
import re

# Splits after ., !, or ? only when followed by whitespace and then an
# uppercase letter or end-of-string -- avoids splitting on a decimal point
# (3.14) or a mid-sentence abbreviation followed by a lowercase continuation.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z]|$)")


def split_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sentence_splitter.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Add `Chunker` to `src/rag/domain/ports.py`**

Append to the existing file:

```python
class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]: ...
```

- [ ] **Step 6: Make `FixedSizeChunker` implement the port**

In `src/rag/infrastructure/fixed_size_chunker.py`, change:

```python
import tiktoken


class FixedSizeChunker:
```

to:

```python
import tiktoken

from src.rag.domain.ports import Chunker


class FixedSizeChunker(Chunker):
```

No other change to this file — `chunk()`'s body and signature are already exactly what the port requires.

- [ ] **Step 7: Narrow `UploadDocument`'s chunker type**

In `src/rag/application/upload_document.py`, change the import and constructor annotation:

```python
from src.rag.domain.ports import Chunker, DocumentRepository, EmbeddingModel, VectorStore
```

and

```python
        chunker: Chunker,
```

(replacing the existing `from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker` import and `chunker: FixedSizeChunker` annotation.)

- [ ] **Step 8: Run the full unit suite to confirm no regression**

Run: `uv run pytest tests/unit/ -v`
Expected: all passing, including every pre-existing `FixedSizeChunker`/`UploadDocument` test — this task changes only type annotations and an inheritance declaration, no behavior.

- [ ] **Step 9: Run ruff and mypy**

Run: `uv run ruff check src/rag/ tests/unit/test_sentence_splitter.py` and `uv run mypy src/rag/`
Expected: both clean.

- [ ] **Step 10: Commit**

```bash
git add src/rag/domain/ports.py src/rag/infrastructure/fixed_size_chunker.py src/rag/infrastructure/_sentence_splitter.py src/rag/application/upload_document.py tests/unit/test_sentence_splitter.py
git commit -m "feat: add Chunker port and shared sentence splitter

Refs #48"
```

---

### Task 2: Sentence-Based chunker

**Files:**
- Create: `src/rag/infrastructure/sentence_based_chunker.py`
- Test: `tests/unit/test_sentence_based_chunker.py`

**Interfaces:**
- Consumes: `Chunker` port (Task 1), `split_sentences` (Task 1).
- Produces: `SentenceBasedChunker`, consumed by Task 6's measurement script and reused by Task 5's `StructureAwareChunker`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_sentence_based_chunker.py
from src.rag.infrastructure.sentence_based_chunker import SentenceBasedChunker


def test_short_text_produces_a_single_chunk():
    chunker = SentenceBasedChunker(chunk_size_tokens=512)
    chunks = chunker.chunk("This is one short sentence.")
    assert chunks == ["This is one short sentence."]


def test_never_splits_a_sentence_across_chunks():
    chunker = SentenceBasedChunker(chunk_size_tokens=10)
    text = " ".join(f"This is sentence number {i} of the document." for i in range(20))
    chunks = chunker.chunk(text)
    # Every chunk must be a clean concatenation of whole sentences -- each
    # chunk, split back into sentences, re-joins to exactly itself with no
    # leftover fragment.
    for c in chunks:
        assert c.strip().endswith((".", "!", "?"))


def test_a_single_oversized_sentence_becomes_its_own_chunk():
    chunker = SentenceBasedChunker(chunk_size_tokens=5)
    long_sentence = " ".join(f"word{i}" for i in range(50)) + "."
    chunks = chunker.chunk(long_sentence)
    assert len(chunks) == 1
    assert chunks[0] == long_sentence


def test_empty_text_produces_no_chunks():
    chunker = SentenceBasedChunker(chunk_size_tokens=512)
    assert chunker.chunk("") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_sentence_based_chunker.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/rag/infrastructure/sentence_based_chunker.py`**

```python
import tiktoken

from src.rag.domain.ports import Chunker
from src.rag.infrastructure._sentence_splitter import split_sentences


class SentenceBasedChunker(Chunker):
    def __init__(self, chunk_size_tokens: int = 512) -> None:
        self._chunk_size = chunk_size_tokens
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(self._encoding.encode(sentence))
            if current and current_tokens + sentence_tokens > self._chunk_size:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(sentence)
            current_tokens += sentence_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sentence_based_chunker.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/rag/infrastructure/sentence_based_chunker.py tests/unit/test_sentence_based_chunker.py
git commit -m "feat: add Sentence-Based chunker

Refs #82"
```

---

### Task 3: Semantic chunker

**Files:**
- Create: `src/rag/infrastructure/semantic_chunker.py`
- Modify: `tests/integration/conftest.py` (no change needed — reuses the existing `embedding_model` fixture from RAG Pipeline)
- Test: `tests/integration/test_semantic_chunker.py`

**Interfaces:**
- Consumes: `Chunker` port, `split_sentences` (Task 1), `SentenceTransformersEmbedder` (existing, from RAG Pipeline).
- Produces: `SemanticChunker`, consumed by Task 6's measurement script.

- [ ] **Step 1: Write the failing tests**

This is an integration test (real embedding model) — mirrors `tests/integration/test_sentence_transformers_embedder.py`'s existing pattern and reuses its session-scoped `embedding_model` fixture.

```python
# tests/integration/test_semantic_chunker.py
from src.rag.infrastructure.semantic_chunker import SemanticChunker


def test_groups_topically_similar_sentences_together(embedding_model):
    chunker = SemanticChunker(embedding_model, chunk_size_tokens=512)
    text = (
        "Cats are small domesticated carnivorous mammals. "
        "Many people keep cats as pets in their homes. "
        "Cats are known for their independence and agility. "
        "Quarterly revenue increased by twelve percent this year. "
        "The finance team attributed the growth to strong product sales. "
        "Operating margins also improved compared to the previous quarter."
    )
    chunks = chunker.chunk(text)
    # Expect roughly two topic groups (cats, finance) -- not asserting an
    # exact chunk count, since the breakpoint threshold is adaptive, but the
    # first sentence (cats) and the last sentence (finance) must not land in
    # the same chunk together, since that would mean the technique found no
    # topic shift in a document that clearly has one.
    first_chunk = next(c for c in chunks if "Cats are small" in c)
    last_chunk = next(c for c in chunks if "Operating margins" in c)
    assert first_chunk != last_chunk


def test_empty_text_produces_no_chunks(embedding_model):
    chunker = SemanticChunker(embedding_model, chunk_size_tokens=512)
    assert chunker.chunk("") == []


def test_short_single_topic_text_produces_one_chunk(embedding_model):
    chunker = SemanticChunker(embedding_model, chunk_size_tokens=512)
    chunks = chunker.chunk("Cats are small mammals. They are often kept as pets.")
    assert len(chunks) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/integration/test_semantic_chunker.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/rag/infrastructure/semantic_chunker.py`**

```python
import math

import tiktoken

from src.rag.domain.ports import Chunker, EmbeddingModel
from src.rag.infrastructure._sentence_splitter import split_sentences


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticChunker(Chunker):
    def __init__(self, embedding_model: EmbeddingModel, chunk_size_tokens: int = 512) -> None:
        self._embedder = embedding_model
        self._chunk_size = chunk_size_tokens
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [sentences[0]]

        embeddings = [self._embedder.embed(s) for s in sentences]
        similarities = [
            _cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)
        ]

        sorted_similarities = sorted(similarities)
        percentile_index = max(0, int(len(sorted_similarities) * 0.25) - 1)
        breakpoint_threshold = sorted_similarities[percentile_index]

        chunks: list[str] = []
        current: list[str] = [sentences[0]]
        current_tokens = len(self._encoding.encode(sentences[0]))

        for i, similarity in enumerate(similarities):
            next_sentence = sentences[i + 1]
            next_tokens = len(self._encoding.encode(next_sentence))
            is_breakpoint = similarity <= breakpoint_threshold
            would_overflow = current_tokens + next_tokens > self._chunk_size
            if current and (is_breakpoint or would_overflow):
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(next_sentence)
            current_tokens += next_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/integration/test_semantic_chunker.py -v`
Expected: `3 passed`. Uses the real MiniLM model via the existing session-scoped `embedding_model` fixture — first run in a session may be slower if the model isn't already cached from an earlier test file.

- [ ] **Step 5: Commit**

```bash
git add src/rag/infrastructure/semantic_chunker.py tests/integration/test_semantic_chunker.py
git commit -m "feat: add Semantic chunker

Refs #84"
```

---

### Task 4: Sliding Window chunker

**Files:**
- Create: `src/rag/infrastructure/sliding_window_chunker.py`
- Test: `tests/unit/test_sliding_window_chunker.py`

**Interfaces:**
- Consumes: `FixedSizeChunker` (Task 1's already-existing class).
- Produces: `SlidingWindowChunker`, consumed by Task 6's measurement script.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_sliding_window_chunker.py
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.sliding_window_chunker import SlidingWindowChunker


def test_is_a_fixed_size_chunker_subclass():
    chunker = SlidingWindowChunker()
    assert isinstance(chunker, FixedSizeChunker)


def test_default_overlap_is_higher_than_fixed_size_chunkers_default():
    default_chunker = FixedSizeChunker()
    sliding_chunker = SlidingWindowChunker()
    # Both use the same 512-token default chunk size; sliding window's
    # overlap (0.5) must exceed fixed size's own default (0.1) -- verified
    # indirectly, since overlap isn't a public attribute: a long text
    # produces more chunks under heavier overlap for the same chunk size,
    # since each chunk advances by a smaller step.
    text = " ".join(f"word{i}" for i in range(2000))
    assert len(sliding_chunker.chunk(text)) > len(default_chunker.chunk(text))


def test_short_text_produces_a_single_chunk():
    chunker = SlidingWindowChunker()
    chunks = chunker.chunk("A short piece of text.")
    assert chunks == ["A short piece of text."]


def test_empty_text_produces_no_chunks():
    chunker = SlidingWindowChunker()
    assert chunker.chunk("") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_sliding_window_chunker.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/rag/infrastructure/sliding_window_chunker.py`**

```python
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker


class SlidingWindowChunker(FixedSizeChunker):
    """Mechanically identical to FixedSizeChunker -- a fixed-size window
    walked across the token stream with overlap. The only difference is the
    default overlap ratio: Fixed Size uses a modest overlap (0.1) as a
    boundary-softening measure; Sliding Window's overlap is the point (0.5),
    trading storage for narrative continuity across chunk boundaries, per
    RAG.md's own framing of the two strategies as a difference of degree and
    intent rather than a different algorithm.
    """

    def __init__(self, chunk_size_tokens: int = 512, overlap_ratio: float = 0.5) -> None:
        super().__init__(chunk_size_tokens, overlap_ratio)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sliding_window_chunker.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/rag/infrastructure/sliding_window_chunker.py tests/unit/test_sliding_window_chunker.py
git commit -m "feat: add Sliding Window chunker

Refs #87"
```

---

### Task 5: Structure-Aware chunker

**Files:**
- Create: `src/rag/infrastructure/structure_aware_chunker.py`
- Test: `tests/unit/test_structure_aware_chunker.py`

**Interfaces:**
- Consumes: `Chunker` port (Task 1), `SentenceBasedChunker` (Task 2).
- Produces: `StructureAwareChunker`, consumed by Task 6's measurement script.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_structure_aware_chunker.py
from src.rag.infrastructure.structure_aware_chunker import StructureAwareChunker


def test_keeps_each_headings_content_together_when_it_fits():
    chunker = StructureAwareChunker(chunk_size_tokens=512)
    text = "# Section One\nSome content here.\n\n# Section Two\nOther content here."
    chunks = chunker.chunk(text)
    assert len(chunks) == 2
    assert "Section One" in chunks[0]
    assert "Some content here" in chunks[0]
    assert "Section Two" in chunks[1]


def test_never_splits_inside_a_fenced_code_block():
    chunker = StructureAwareChunker(chunk_size_tokens=5)
    text = "# Code Example\n```python\ndef f():\n    return 1\n```\nEnd of example."
    chunks = chunker.chunk(text)
    code_chunk = next(c for c in chunks if "def f():" in c)
    assert "```python" in code_chunk
    assert "return 1" in code_chunk
    assert "```" in code_chunk.split("```python", 1)[1]  # the closing fence is in the same chunk


def test_a_document_with_no_headings_falls_back_to_sentence_based():
    chunker = StructureAwareChunker(chunk_size_tokens=512)
    chunks = chunker.chunk("Just a plain paragraph with no headings at all. It has two sentences.")
    assert len(chunks) == 1
    assert "plain paragraph" in chunks[0]


def test_empty_text_produces_no_chunks():
    chunker = StructureAwareChunker(chunk_size_tokens=512)
    assert chunker.chunk("") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_structure_aware_chunker.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/rag/infrastructure/structure_aware_chunker.py`**

```python
import re

import tiktoken

from src.rag.domain.ports import Chunker
from src.rag.infrastructure.sentence_based_chunker import SentenceBasedChunker

_HEADING = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)


def _split_into_sections(text: str) -> list[str]:
    """Split on Markdown heading lines, keeping each heading with the
    content that follows it up to the next heading. Code fences are matched
    and treated as opaque before heading-matching runs, so a heading-looking
    line inside a code block (e.g. a shell comment starting with #) is never
    mistaken for a real section break.
    """
    fence_spans = [m.span() for m in _CODE_FENCE.finditer(text)]

    def _inside_a_fence(pos: int) -> bool:
        return any(start <= pos < end for start, end in fence_spans)

    heading_starts = [m.start() for m in _HEADING.finditer(text) if not _inside_a_fence(m.start())]
    if not heading_starts:
        return [text]

    sections = []
    if heading_starts[0] > 0:
        sections.append(text[: heading_starts[0]])
    for i, start in enumerate(heading_starts):
        end = heading_starts[i + 1] if i + 1 < len(heading_starts) else len(text)
        sections.append(text[start:end])
    return [s for s in sections if s.strip()]


class StructureAwareChunker(Chunker):
    def __init__(self, chunk_size_tokens: int = 512) -> None:
        self._chunk_size = chunk_size_tokens
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._fallback = SentenceBasedChunker(chunk_size_tokens)

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []

        sections = _split_into_sections(text)
        if len(sections) == 1 and not _HEADING.search(sections[0]):
            # No heading structure recognized at all -- Structure-Aware only
            # works as well as the parser recognizing structure, and there's
            # none to recognize here.
            return self._fallback.chunk(text)

        chunks: list[str] = []
        for section in sections:
            if _CODE_FENCE.search(section) or len(self._encoding.encode(section)) <= self._chunk_size:
                # A section containing a fenced code block is kept whole
                # even if it exceeds chunk_size_tokens -- a broken code
                # block is a worse failure than one oversized chunk.
                chunks.append(section.strip())
            else:
                chunks.extend(self._fallback.chunk(section))
        return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_structure_aware_chunker.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/rag/infrastructure/structure_aware_chunker.py tests/unit/test_structure_aware_chunker.py
git commit -m "feat: add Structure-Aware chunker

Refs #88"
```

---

### Task 6: Scenario, measurement scripts, and real results

**Files:**
- Create: `evaluation/scenarios/rag-chunking-strategies/corpus/rag.md` (a copy of `docs/architecture/RAG.md`)
- Create: `evaluation/scenarios/rag-chunking-strategies/queries.yaml`
- Create: `evaluation/scenarios/rag-chunking-strategies/run_comparison.py` (one script, parameterized by strategy name via a CLI argument, rather than five near-duplicate files)
- Create: `evaluation/reports/rag-chunking-fixed-size.md`, `rag-chunking-sentence-based.md`, `rag-chunking-semantic.md`, `rag-chunking-sliding-window.md`, `rag-chunking-structure-aware.md` (generated, not hand-written)

**Interfaces:**
- Consumes: every chunker from Tasks 1-5, `RunComparison`/`ClaudeJudge`/`markdown_report`/`load_scenario` from the Evaluation Harness, `OllamaChatModel`, `UploadDocument`, `AnswerQuestion`, `SearchDocuments`.
- Produces: five real, committed measurement reports — the actual deliverable this whole batch exists to produce.

- [ ] **Step 1: Copy the corpus**

```bash
cp docs/architecture/RAG.md evaluation/scenarios/rag-chunking-strategies/corpus/rag.md
```

- [ ] **Step 2: Read `docs/architecture/RAG.md`, then write `evaluation/scenarios/rag-chunking-strategies/queries.yaml`**

Read the actual current content of `docs/architecture/RAG.md` first — every question and its success criterion must be genuinely answerable from that document's real content as it exists right now, not guessed or written from memory of the spec's illustrative examples. Good candidate topics already known to exist in that file (verify the exact wording/figures against the live file, since this plan was written without re-reading it at plan-execution time): Fixed Size chunking's suggested starting size/overlap, what Semantic Chunking needs and why it's the most complex of the six, the Production Grade pipeline's constituent techniques, Parent Document Retrieval's expected impact figure, Sliding Window's tradeoff. Write five questions in this shape, each with a success criterion stated as a specific, checkable substring or fact:

```yaml
name: rag-chunking-strategies
questions:
  - question: "<a real question genuinely answerable from the live file>"
    success_criterion: "<the specific fact/figure/phrase the answer must contain>"
  # ... four more, same shape
```

- [ ] **Step 3: Write `evaluation/scenarios/rag-chunking-strategies/run_comparison.py`**

```python
"""Manual comparison runner for the RAG Chunking Strategies batch. Not
pytest-collected (no test_* function at module level) -- mirrors this
project's existing test_rag_smoke.py / test_docker_compose_smoke.py pattern
for scripts that make real model calls and are run by hand.

Usage:
    uv run python evaluation/scenarios/rag-chunking-strategies/run_comparison.py <strategy>

<strategy> is one of: fixed-size, sentence-based, semantic, sliding-window, structure-aware
"""
import asyncio
import sys
import uuid
from pathlib import Path

import ollama

from evaluation.application.run_comparison import RunComparison
from evaluation.domain.entities import Answer
from evaluation.infrastructure.claude_judge import ClaudeJudge
from evaluation.infrastructure.markdown_report import render
from evaluation.scenarios.loader import load_scenario
from src.rag.application.answer_question import AnswerQuestion
from src.rag.application.search_documents import SearchDocuments
from src.rag.application.upload_document import UploadDocument
from src.rag.domain.ports import Chunker
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
    document = await upload.execute(
        tenant_id=tenant_id, filename="rag.md", content=corpus_bytes
    )
    print(f"Uploaded corpus under strategy {strategy!r}: {document.chunk_count} chunks")

    search = SearchDocuments(embedding_model=embedder, vector_store=vector_store)
    answer_question = AnswerQuestion(search_documents=search, chat_model=chat_model, top_k=5)

    async def baseline(question: str) -> Answer:
        text = await chat_model.generate(question=question, context="")
        return Answer(text=text, input_tokens=0, output_tokens=0)

    async def treatment(question: str) -> Answer:
        result = await answer_question.execute(tenant_id=tenant_id, question=question)
        return Answer(text=result.answer, input_tokens=0, output_tokens=0)

    def success_check(question: str, answer: Answer) -> bool:
        # Matched against queries.yaml's success_criterion by hand at report
        # time -- the harness's success_check is a plain callable, and each
        # question's real substring check lives here rather than being
        # parsed out of the YAML's human-readable description.
        return True  # replaced with real per-question substring checks below

    import anthropic

    judge = ClaudeJudge(
        client=anthropic.AsyncAnthropic(), model_id="claude-opus-5"
    )
    use_case = RunComparison(judge=judge, repeat_count=3)
    result = await use_case.execute(
        scenario_name=scenario.name,
        model_config=_MODEL_CONFIG,
        success_criterion="see evaluation/scenarios/rag-chunking-strategies/queries.yaml",
        rag=True, cag=False, mag=False,
        notes=f"strategy={strategy}, corpus=docs/architecture/RAG.md, {document.chunk_count} chunks",
        questions=[q.question for q in scenario.questions],
        baseline=baseline,
        treatment=treatment,
        success_check=success_check,
    )

    report_path = Path(f"evaluation/reports/rag-chunking-{strategy}.md")
    report_path.write_text(render(result), encoding="utf-8")
    print(f"Report written to {report_path}")


class _NullDocumentRepository:
    async def save_document(self, document) -> None:
        pass

    async def update_document_status(self, document_id, status, chunk_count) -> None:
        pass

    async def save_chunks(self, chunks, tenant_id) -> None:
        pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    asyncio.run(_run(sys.argv[1]))
```

Replace this script's `success_check` function body with five real substring checks, one per question, matching the real criteria you wrote into `queries.yaml` in Step 2 (you already read the live `docs/architecture/RAG.md` there — reuse that same reading here rather than re-reading it), keyed on `question` text:

```python
    def success_check(question: str, answer: Answer) -> bool:
        checks = {
            scenario.questions[0].question: lambda a: True,  # your real Step-2 criterion, as code
            scenario.questions[1].question: lambda a: True,  # ...
            # one entry per question in queries.yaml, five total
        }
        return checks.get(question, lambda a: False)(answer)
```

(The `lambda a: True` bodies above are illustrative placeholders for this plan text only — the actual committed script must have five real, question-specific substring checks, each directly encoding the `success_criterion` you wrote in Step 2's `queries.yaml`.)

`_NullDocumentRepository` exists because this script re-uploads the same corpus for every strategy run and doesn't need Postgres persistence to prove a chunking strategy's retrieval quality — only Qdrant (the actual search path) and the in-memory `Document`/`Chunk` objects `UploadDocument` already returns matter here. Using the real `PostgresDocumentRepository` would require a running Postgres and migrated schema for a script that doesn't need either.

- [ ] **Step 4: Run the full unit and integration suites to confirm no regression from Tasks 1-5**

Run: `uv run pytest tests/unit/ tests/integration/ -v` (needs Docker for the integration tier)
Expected: all passing.

- [ ] **Step 5: Bring up Qdrant and Ollama, then run all five comparisons**

Requires Docker (for Qdrant) and Ollama running locally with `qwen3.5` pulled (already set up in this project's environment) and a real `ANTHROPIC_API_KEY` in the environment (the judge is Claude, per `qualitative-rubric.md` — this is real API cost, five judge calls per strategy run, 25 total across this task; small relative to a single conversation but real).

```bash
docker run -d -p 6333:6333 qdrant/qdrant:v1.16.2
export ANTHROPIC_API_KEY=<real key>
for strategy in fixed-size sentence-based semantic sliding-window structure-aware; do
  uv run python evaluation/scenarios/rag-chunking-strategies/run_comparison.py "$strategy"
done
```

Expected: five report files written under `evaluation/reports/`, each with real numbers.

- [ ] **Step 6: Read all five reports and record what they actually show**

For each report: does the treatment's task success rate and qualitative groundedness score genuinely improve over baseline (RAG completely off)? Does the relative comparison between strategies match or contradict RAG.md's own per-strategy tradeoff descriptions? Write down the honest answer for each — including "no meaningful difference" if that's what the numbers show. This is exactly the "prove it, don't assume it" step the whole harness exists for.

- [ ] **Step 7: Commit**

```bash
git add evaluation/scenarios/rag-chunking-strategies/ evaluation/reports/rag-chunking-*.md
git commit -m "feat: measure all five chunking strategies against a real corpus

Refs #48, #77, #82, #84, #87, #88"
```

- [ ] **Step 8: Post results to each GitHub issue**

For each of #77, #82, #84, #87, #88: `gh issue comment <number> --body-file <(uv run python -c "from evaluation.infrastructure.markdown_report import render_github_comment; ...")` — or more simply, since `render_github_comment` isn't wired into this script's output directly, read each `evaluation/reports/rag-chunking-<slug>.md` file and post its content (or a `render_github_comment`-shaped extract) via `gh issue comment <number> --body-file evaluation/reports/rag-chunking-<slug>.md`, then check off that issue's "real measured result documented" DoD item.

---

## After this plan

Parent Document Chunking (#85) remains deferred, to be built alongside Parent Document Retrieval (#56/#105) as its own later piece of work, per this plan's own spec. The next natural piece of RAG work is either that pairing, or the next individual-technique cluster (Multi-Query #52/#104, Parent Document Retrieval #56/#105, Context Compression #61/#107, HyDE #65/#111, Self-RAG #68/#115, CRAG #73/#119, Reranking #90/#94/#98, Hybrid Search #102) — each following this same batched shape.
