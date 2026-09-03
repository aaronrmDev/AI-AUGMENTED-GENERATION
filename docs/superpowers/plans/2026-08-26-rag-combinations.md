# RAG Combinations (Batch E) Implementation Plan

**Goal:** Build and live-measure the 5 remaining RAG combinations by nesting existing,
already-tested `Retriever` decorators — zero new production classes.

**Architecture:** One scenario script, `run_combinations_comparison.py <combination>`,
with `<combination>` one of `multi-query-hyde`, `speed-demon`, `reranking-crag`,
`production-grade`, `fort-knox`. Composition orders are locked in the design spec.

**Tech Stack:** Same as every prior batch — Python, Ollama (qwen3.5), Qdrant, pytest.

**Spec:** `docs/superpowers/specs/2026-08-26-rag-combinations-design.md`

## Global Constraints

- Script name unique across `evaluation/scenarios/`: `run_combinations_comparison.py`.
- Every non-QA `ChatModel` call (query rewriting, relevance judgment, gate checks) uses
  `complete()`, never `generate()` — the bug Batch C's review found and fixed.
- `dataclasses.replace(result, notes=...)` for anything computed only inside
  `treatment()`, applied after `execute()` returns, never built into the `notes=` kwarg
  directly.
- `_InMemoryDocumentRepository` implements all 5 `DocumentRepository` abstract methods.
- No new production code needed — if a combination seems to need one, that's a signal
  the composition order needs rethinking, not a signal to add a class.

## Task 1: Scenario, live measurement (all 5 combinations), GitHub reporting

**Files:**
- Create: `evaluation/scenarios/rag-combinations/corpus/rag.md`
- Create: `evaluation/scenarios/rag-combinations/queries.yaml` (7 questions, shared
  across all 5 arms)
- Create: `evaluation/scenarios/rag-combinations/run_combinations_comparison.py`

- [ ] Write `corpus/rag.md` (copy of `docs/architecture/RAG.md`, verified byte-identical).
- [ ] Write `queries.yaml`: 7 questions whose success criteria are verified against the
      live `docs/architecture/RAG.md`, spanning material touched by all 5 combinations'
      contributing techniques (chunking/hybrid search, reranking, HyDE/Multi-Query,
      CRAG, parent document, context compression, and at least one of the "Top 5 Most
      Impactful Pairs" / archetype names themselves).
- [ ] Write `run_combinations_comparison.py` per the design spec's five composition
      orders. Fort Knox uses `UploadDocumentWithParents`; the other four use plain
      `UploadDocument`. Each combination's `treatment()` wraps whichever composed
      retriever/use-case is active; `baseline()` is the same plain, no-retrieval
      `generate()` call every prior batch's scenario uses (same CAVEAT 4-style
      methodology disclosure Batch D established).
- [ ] Run `uv run ruff check src/rag/ evaluation/` and `uv run mypy src/rag/
      evaluation/` — both clean.
- [ ] Run `uv run pytest tests/unit/ tests/integration/ -v` — all passing, no
      regressions (no new test files expected, since no new production classes).
- [ ] Commit.
- [ ] Bring up / confirm Qdrant + Ollama reachable.
- [ ] Run all 5 arms: `PYTHONPATH=. uv run python evaluation/scenarios/rag-combinations/
      run_combinations_comparison.py <combination>` for each of the 5 names above.
- [ ] Read each report honestly — including any combination where the composed result
      is worse than a simpler ancestor combination already measured in Batches A/B,
      which is a legitimate, disclosable finding (RAG.md's own compatibility matrix
      promises "zero conflicts," not "monotonically better with more techniques
      stacked").
- [ ] Commit reports.
- [ ] Post results to the 10 GitHub issues (#124/#138, #136/#143, #125/#139, #132/#141,
      #135/#142) and close each pair once genuinely complete.

## Task 2: Final review, fix wave if needed, merge

Same sequence as every prior batch: final whole-branch review (opus, adversarial,
independent reproduction) via the `subagent-driven-development` skill's
`review-package` script; fix wave if the review finds anything (expected, given every
prior batch's review has found real issues); scoped re-review verifying the fix if
needed; merge to `develop` (locally, consistent with this session's established pattern
under the standing "continue the backlog" directive); clean up the worktree.
