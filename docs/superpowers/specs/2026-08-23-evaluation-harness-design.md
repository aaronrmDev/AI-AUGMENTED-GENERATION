# Evaluation Harness — Design Spec

## Why this exists

`docs/evaluation/COMPARISON_METHODOLOGY.md`, `quantitative-template.md`, and `qualitative-rubric.md` already fully specify how this project proves a RAG/CAG/MAG technique earns its complexity: a self-versus-self ablation (same model, technique on vs. off), a quantitative table (tokens, latency, task success), and a qualitative judge pass (Claude scoring both outputs on four dimensions), pasted together into the technique's GitHub issue. What doesn't exist yet is anything that *runs* that methodology — today it's a human manually timing calls, counting tokens, and prompting a judge by hand. Every one of the ~100 issues across the RAG/MAG/CAG backlog ends its Definition of Done with "real measured result documented," and this harness is what makes that a command instead of an afternoon of manual bookkeeping, built once so every later technique reuses it instead of re-inventing measurement.

## Scope

In scope: a reusable comparison runner that takes a baseline strategy and a treatment strategy for the same scenario, runs each enough times to produce real latency percentiles, captures token usage, checks task success against a caller-supplied criterion, invokes a judge model with the qualitative rubric's exact prompt structure, and renders both results into one report matching the two existing templates' formats exactly — plus a way to get that report's summary into the actual GitHub issue thread, since that's where `COMPARISON_METHODOLOGY.md` says the DoD-closing record lives. Out of scope: the techniques themselves (this harness measures them, doesn't implement HyDE or Multi-Query or anything else), any new judge-scoring rubric (the four dimensions and 1-5 scale are already fixed by `qualitative-rubric.md` — this harness executes that rubric, doesn't redesign it), and vLLM-on-ROCm wiring (the harness's `ChatModel`/judge ports don't care what's behind them; standing up the actual self-hosted ablation subjects for official numbers is separate, later infrastructure work).

## Module layout

A new top-level `evaluation/` package, sibling to `src/` rather than under it — this is a development-time research tool, not part of the shipped API, the same relationship `tests/` already has to `src/`. Hexagonal shape, matching every paradigm module already in this codebase:

```text
evaluation/
├── domain/
│   ├── entities.py      # Answer, RunResult, JudgeScores, ComparisonResult
│   └── ports.py         # Judge (ABC) — ChatModel is reused as-is from src/rag/domain/ports.py,
│                         # not redefined, since the subject-under-test IS a ChatModel
├── application/
│   └── run_comparison.py   # RunComparison use case
├── infrastructure/
│   ├── claude_judge.py       # Judge, using qualitative-rubric.md's exact prompt structure
│   └── markdown_report.py    # renders a ComparisonResult into quantitative-template.md's
│                              # table + qualitative-rubric.md's score block, verbatim format
├── scenarios/
│   └── <scenario-name>/
│       ├── corpus/            # documents a scenario's baseline/treatment runs ingest
│       └── queries.yaml       # the fixed input set: question + its stated success criterion
└── reports/
    └── <story-slug>.md        # generated output — committed, since this project's own "0
                                 # speculation" framing treats a measured result as evidence,
                                 # not scratch output
```

`RunComparison` takes two already-configured callables — `baseline: Callable[[str], Awaitable[Answer]]` and `treatment: Callable[[str], Awaitable[Answer]]` — rather than knowing anything about RAG/CAG/MAG itself. Producing "baseline" (technique off) vs. "treatment" (technique on) output is the calling code's job: a chunking-strategy comparison wires `baseline` to a direct `ChatModel.generate()` call with no retrieval, and `treatment` to the full `AnswerQuestion` use case with that chunking strategy's `UploadDocument` behind it. The harness's job starts once both callables exist: run each `repeat_count` times (default 5, per `quantitative-template.md`) across every question in the scenario's fixed input set, measure wall-clock latency per call, read `input_tokens`/`output_tokens` off each `Answer`, check the caller-supplied success predicate per question, then call the judge once per question with the baseline and treatment answers as Response A / Response B.

## Data model

```python
@dataclass(frozen=True)
class Answer:
    text: str
    input_tokens: int
    output_tokens: int

@dataclass(frozen=True)
class RunResult:
    label: str                    # "Baseline" | "Treatment"
    latency_p50_ms: float
    latency_p95_ms: float
    total_input_tokens: int
    total_output_tokens: int
    task_success_rate: float      # 0.0-1.0 across the scenario's questions
    answers: list[Answer]         # one per question, from the last of the repeat_count runs
                                    # (the one whose text actually gets judged)

@dataclass(frozen=True)
class JudgeScores:
    coherence: int                # 1-5, qualitative-rubric.md's scale
    relevance: int
    completeness: int
    groundedness: int
    unverifiable_claims: list[str]

@dataclass(frozen=True)
class ComparisonResult:
    scenario_name: str
    model_config: str             # e.g. "qwen3.5, Ollama, Q4_K_M" — free text, caller states it
    success_criterion: str        # what "task success" means for this Story, stated up front
    baseline: RunResult
    treatment: RunResult
    baseline_judge: JudgeScores
    treatment_judge: JudgeScores
```

`Judge` (`evaluation/domain/ports.py`) is a one-method port: `score(query: str, response_a: str, response_b: str) -> tuple[JudgeScores, JudgeScores]`, implemented by `ClaudeJudge` using the Anthropic adapter pattern already established (`ClaudeChatModel` is the direct precedent — same client construction, same "scan for the first text block" response handling). The judge prompt is built exactly per `qualitative-rubric.md`'s five-part structure: query, Response A (baseline, unlabeled as such), Response B (treatment), the four dimensions with their 1-5 definitions, and the hallucination-flag instruction — asking the judge to return structured JSON so the harness can parse scores reliably rather than free-text-parsing a judge's prose.

## Report generation

`markdown_report.render(result: ComparisonResult) -> str` produces one file matching both templates' exact structure: `quantitative-template.md`'s table (Run/RAG/CAG/MAG/Model/tokens/latency/task success/Δ columns, Baseline row then Treatment row, the delta computed by the renderer not hand-typed) followed by `qualitative-rubric.md`'s scoring block (both responses' four-dimension scores, flagged unverifiable claims). Written to `evaluation/reports/<story-slug>.md`. A second, shorter render (`markdown_report.render_github_comment(result)`) produces just the same content trimmed to what's meant to be pasted into the issue thread per `COMPARISON_METHODOLOGY.md`'s closing section — posted via `gh issue comment <number> --body-file <path>`, run by hand rather than automatically, since deciding *when* a result is ready to close a Story's DoD is a human call, not something this harness should do unattended.

## Testing strategy

Unit tests for `RunResult`'s latency/success-rate computation and `ClaudeJudge`'s prompt construction (against a fake Anthropic client, mirroring `test_claude_chat_model.py`) — no real model calls needed to prove the harness's own logic is correct. One integration-style test runs `RunComparison` end-to-end against two trivial fake callables (a "baseline" and "treatment" that return fixed `Answer`s) and a fake `Judge`, proving the whole pipeline wires together, still without spending real tokens. The harness's very first real exercise — proving `markdown_report`'s output actually matches the templates' format, and that a real Claude judge call parses correctly — happens as part of the first technique comparison that uses it (a chunking-strategy Story), not as a standalone smoke test with nothing real to compare.

## Non-goals

No new scoring dimensions beyond the four `qualitative-rubric.md` already defines. No automated GitHub-comment posting (a human decides when a result is ready). No vLLM-on-ROCm wiring — the harness's ports don't know or care what's behind `ChatModel`/`Judge`; Ollama today, vLLM later, same interface either way.
