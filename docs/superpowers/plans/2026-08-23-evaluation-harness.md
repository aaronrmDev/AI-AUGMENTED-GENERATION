# Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable comparison runner that automates the already-fully-specified manual methodology in `docs/evaluation/COMPARISON_METHODOLOGY.md`, `quantitative-template.md`, and `qualitative-rubric.md`, so every RAG/MAG/CAG technique Story can close its "real measured result documented" DoD item by running a comparison instead of hand-filling tables.

**Architecture:** A new top-level `evaluation/` package (sibling to `src/`, the same relationship `tests/` already has), hexagonal like every paradigm module already in this codebase — `domain/` (entities, the `Judge` port; the subject-under-test reuses `src/rag/domain/ports.py`'s existing `ChatModel`, not a new port), `application/` (the `RunComparison` use case), `infrastructure/` (a Claude-backed `Judge` adapter, a Markdown report renderer, a scenario YAML loader).

**Tech Stack:** Everything already in this project (Python 3.11+, pytest, uv, mypy, ruff), plus `pyyaml` for scenario files.

**Spec:** docs/superpowers/specs/2026-08-23-evaluation-harness-design.md

## Global Constraints

- `evaluation/domain/` has zero framework imports, matching `src/rag/domain/`'s own rule.
- `RunComparison` (application layer) depends only on the `Judge` port and plain callables passed in by the caller — it never imports a concrete technique, a concrete `ChatModel` adapter, or anything RAG/CAG/MAG-specific. This is what keeps it reusable across all three paradigms.
- Latency is measured across `repeat_count` (default 5) repeated calls per question, per `quantitative-template.md`'s "a single-run latency number is noise" rule — p50/p95 computed from ALL repeat runs across ALL questions in the scenario, pooled into one sorted list.
- Only the LAST of the `repeat_count` runs' answer (text + tokens) is kept as "the" answer for a question — the one used for token totals, task-success checking, and judging. This is a deliberate, documented simplification: judging every repeat run would multiply judge-model cost for no benefit the four-dimension rubric asks for.
- The judge is called once per question (baseline answer vs. treatment answer for that question), producing one `(JudgeScores, JudgeScores)` pair per question — never one aggregate score across a whole scenario, since `qualitative-rubric.md`'s rubric is defined per query/response pair.
- `ClaudeJudge` must scan `response.content` for the first `type == "text"` block rather than indexing `[0]` — the same defensive pattern `ClaudeChatModel` already uses, and for the same reason: `claude-opus-5` (a valid judge model choice per `COMPARISON_METHODOLOGY.md`'s roster) runs adaptive thinking by default.
- Every commit follows the `.gitmessage` template and Conventional Commits format, footer `Refs Epic #144`.

---

### Task 1: Scaffold and domain layer

**Files:**
- Create: `evaluation/__init__.py`, `evaluation/domain/__init__.py`, `evaluation/application/__init__.py`, `evaluation/infrastructure/__init__.py`, `evaluation/scenarios/__init__.py`
- Create: `evaluation/domain/entities.py`
- Create: `evaluation/domain/ports.py`
- Modify: `pyproject.toml` (add `pyyaml`)
- Test: `tests/unit/test_evaluation_domain.py`

**Interfaces:**
- Produces: `Answer`, `RunResult`, `JudgeScores`, `ComparisonResult` entities; `Judge` port ABC. Every later task in this plan imports from here.

- [ ] **Step 1: Add `pyyaml` to `pyproject.toml`**

Add to `dependencies`: `"pyyaml>=6.0",`

- [ ] **Step 2: Create the package skeleton**

Empty `__init__.py` files at each path listed above.

- [ ] **Step 3: Write the failing tests**

```python
# tests/unit/test_evaluation_domain.py
from evaluation.domain.entities import Answer, ComparisonResult, JudgeScores, RunResult


def test_answer_holds_text_and_token_counts():
    answer = Answer(text="42", input_tokens=10, output_tokens=2)
    assert answer.text == "42"
    assert answer.input_tokens == 10
    assert answer.output_tokens == 2


def test_judge_scores_defaults_to_no_unverifiable_claims():
    scores = JudgeScores(coherence=5, relevance=5, completeness=5, groundedness=5)
    assert scores.unverifiable_claims == []


def test_run_result_holds_aggregate_stats():
    result = RunResult(
        label="Baseline", latency_p50_ms=100.0, latency_p95_ms=150.0,
        total_input_tokens=50, total_output_tokens=20, task_success_rate=0.8,
        answers=[Answer(text="a", input_tokens=25, output_tokens=10)],
    )
    assert result.label == "Baseline"
    assert result.task_success_rate == 0.8


def test_comparison_result_holds_both_runs_and_judge_scores_per_question():
    baseline = RunResult(
        label="Baseline", latency_p50_ms=100.0, latency_p95_ms=150.0,
        total_input_tokens=50, total_output_tokens=20, task_success_rate=1.0, answers=[],
    )
    treatment = RunResult(
        label="Treatment", latency_p50_ms=80.0, latency_p95_ms=120.0,
        total_input_tokens=50, total_output_tokens=15, task_success_rate=1.0, answers=[],
    )
    scores = JudgeScores(coherence=4, relevance=4, completeness=4, groundedness=4)
    result = ComparisonResult(
        scenario_name="test-scenario", model_config="qwen3.5", success_criterion="exact match",
        baseline=baseline, treatment=treatment, judge_scores=[(scores, scores)],
    )
    assert result.scenario_name == "test-scenario"
    assert len(result.judge_scores) == 1
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_evaluation_domain.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 5: Write `evaluation/domain/entities.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Answer:
    text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class RunResult:
    label: str
    latency_p50_ms: float
    latency_p95_ms: float
    total_input_tokens: int
    total_output_tokens: int
    task_success_rate: float
    answers: list[Answer]


@dataclass(frozen=True)
class JudgeScores:
    coherence: int
    relevance: int
    completeness: int
    groundedness: int
    unverifiable_claims: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ComparisonResult:
    scenario_name: str
    model_config: str
    success_criterion: str
    baseline: RunResult
    treatment: RunResult
    judge_scores: list[tuple[JudgeScores, JudgeScores]]
```

- [ ] **Step 6: Write `evaluation/domain/ports.py`**

```python
from __future__ import annotations

from abc import ABC, abstractmethod

from evaluation.domain.entities import JudgeScores


class Judge(ABC):
    @abstractmethod
    async def score(
        self, query: str, response_a: str, response_b: str
    ) -> tuple[JudgeScores, JudgeScores]: ...
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_evaluation_domain.py -v`
Expected: `4 passed`.

- [ ] **Step 8: Confirm zero framework imports**

Run: `grep -rE "^(import|from) (fastapi|sqlalchemy|anthropic|ollama)" evaluation/domain/`
Expected: no output.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock evaluation/ tests/unit/test_evaluation_domain.py
git commit -m "feat: scaffold evaluation harness domain layer

Refs Epic #144"
```

---

### Task 2: RunComparison use case

**Files:**
- Create: `evaluation/application/run_comparison.py`
- Create: `tests/unit/evaluation_fakes.py`
- Test: `tests/unit/test_run_comparison.py`

**Interfaces:**
- Consumes: `Answer`, `ComparisonResult`, `RunResult`, `JudgeScores`, `Judge` (Task 1).
- Produces: `RunComparison`, consumed by every later technique comparison script (outside this plan's scope) and by Task 5's own report-generation test.

- [ ] **Step 1: Write `tests/unit/evaluation_fakes.py`**

```python
from evaluation.domain.entities import JudgeScores
from evaluation.domain.ports import Judge


class FakeJudge(Judge):
    def __init__(self, scores: JudgeScores | None = None) -> None:
        self._scores = scores or JudgeScores(coherence=4, relevance=4, completeness=4, groundedness=4)
        self.calls: list[tuple[str, str, str]] = []

    async def score(self, query: str, response_a: str, response_b: str) -> tuple[JudgeScores, JudgeScores]:
        self.calls.append((query, response_a, response_b))
        return self._scores, self._scores
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/test_run_comparison.py
from evaluation.application.run_comparison import RunComparison
from evaluation.domain.entities import Answer
from tests.unit.evaluation_fakes import FakeJudge


async def test_run_comparison_produces_a_result_for_baseline_and_treatment():
    call_count = {"baseline": 0, "treatment": 0}

    async def baseline(question: str) -> Answer:
        call_count["baseline"] += 1
        return Answer(text=f"baseline answer to {question}", input_tokens=10, output_tokens=5)

    async def treatment(question: str) -> Answer:
        call_count["treatment"] += 1
        return Answer(text=f"treatment answer to {question}", input_tokens=10, output_tokens=3)

    def success_check(question: str, answer: Answer) -> bool:
        return question in answer.text

    use_case = RunComparison(judge=FakeJudge(), repeat_count=3)
    result = await use_case.execute(
        scenario_name="test-scenario",
        model_config="qwen3.5",
        success_criterion="answer mentions the question",
        questions=["q1", "q2"],
        baseline=baseline,
        treatment=treatment,
        success_check=success_check,
    )

    assert result.scenario_name == "test-scenario"
    assert call_count["baseline"] == 6  # 2 questions x 3 repeats
    assert call_count["treatment"] == 6
    assert result.baseline.task_success_rate == 1.0
    assert result.treatment.task_success_rate == 1.0
    assert result.baseline.total_output_tokens == 10  # 5 x 2 questions, last-run answers only
    assert result.treatment.total_output_tokens == 6  # 3 x 2 questions
    assert len(result.judge_scores) == 2  # one pair per question


async def test_run_comparison_computes_latency_percentiles():
    async def instant(question: str) -> Answer:
        return Answer(text="ok", input_tokens=1, output_tokens=1)

    use_case = RunComparison(judge=FakeJudge(), repeat_count=5)
    result = await use_case.execute(
        scenario_name="latency-test", model_config="qwen3.5", success_criterion="n/a",
        questions=["q1"], baseline=instant, treatment=instant,
        success_check=lambda q, a: True,
    )

    assert result.baseline.latency_p50_ms >= 0.0
    assert result.baseline.latency_p95_ms >= result.baseline.latency_p50_ms


async def test_run_comparison_passes_the_right_answers_to_the_judge():
    async def baseline(question: str) -> Answer:
        return Answer(text="base", input_tokens=1, output_tokens=1)

    async def treatment(question: str) -> Answer:
        return Answer(text="treat", input_tokens=1, output_tokens=1)

    judge = FakeJudge()
    use_case = RunComparison(judge=judge, repeat_count=1)
    await use_case.execute(
        scenario_name="s", model_config="m", success_criterion="c",
        questions=["only question"], baseline=baseline, treatment=treatment,
        success_check=lambda q, a: True,
    )

    assert judge.calls == [("only question", "base", "treat")]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_run_comparison.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Write `evaluation/application/run_comparison.py`**

```python
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from evaluation.domain.entities import Answer, ComparisonResult, RunResult
from evaluation.domain.ports import Judge

Strategy = Callable[[str], Awaitable[Answer]]
SuccessCheck = Callable[[str, Answer], bool]


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, round(pct * (len(sorted_values) - 1)))
    return sorted_values[index]


class RunComparison:
    def __init__(self, judge: Judge, repeat_count: int = 5) -> None:
        self._judge = judge
        self._repeat_count = repeat_count

    async def _run_strategy(
        self, label: str, strategy: Strategy, questions: list[str], success_check: SuccessCheck
    ) -> RunResult:
        latencies_ms: list[float] = []
        last_answers: list[Answer] = []
        successes = 0

        for question in questions:
            answer: Answer | None = None
            for _ in range(self._repeat_count):
                start = time.perf_counter()
                answer = await strategy(question)
                latencies_ms.append((time.perf_counter() - start) * 1000)
            assert answer is not None
            last_answers.append(answer)
            if success_check(question, answer):
                successes += 1

        latencies_ms.sort()
        return RunResult(
            label=label,
            latency_p50_ms=_percentile(latencies_ms, 0.50),
            latency_p95_ms=_percentile(latencies_ms, 0.95),
            total_input_tokens=sum(a.input_tokens for a in last_answers),
            total_output_tokens=sum(a.output_tokens for a in last_answers),
            task_success_rate=successes / len(questions) if questions else 0.0,
            answers=last_answers,
        )

    async def execute(
        self,
        scenario_name: str,
        model_config: str,
        success_criterion: str,
        questions: list[str],
        baseline: Strategy,
        treatment: Strategy,
        success_check: SuccessCheck,
    ) -> ComparisonResult:
        baseline_result = await self._run_strategy("Baseline", baseline, questions, success_check)
        treatment_result = await self._run_strategy("Treatment", treatment, questions, success_check)

        judge_scores = [
            await self._judge.score(question, b.text, t.text)
            for question, b, t in zip(
                questions, baseline_result.answers, treatment_result.answers, strict=True
            )
        ]

        return ComparisonResult(
            scenario_name=scenario_name,
            model_config=model_config,
            success_criterion=success_criterion,
            baseline=baseline_result,
            treatment=treatment_result,
            judge_scores=judge_scores,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_run_comparison.py -v`
Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add evaluation/application/run_comparison.py tests/unit/evaluation_fakes.py tests/unit/test_run_comparison.py
git commit -m "feat: add RunComparison use case

Refs Epic #144"
```

---

### Task 3: Claude judge adapter

**Files:**
- Create: `evaluation/infrastructure/claude_judge.py`
- Test: `tests/unit/test_claude_judge.py`

**Interfaces:**
- Consumes: `Judge` port, `JudgeScores` entity (Task 1).
- Produces: `ClaudeJudge`, consumed by any comparison script wiring a real judge (outside this plan's scope) and by Task 5's report-generation test via a fake.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_claude_judge.py
import json

from evaluation.infrastructure.claude_judge import ClaudeJudge


class _FakeMessage:
    def __init__(self, payload: dict) -> None:
        thinking_block = type("ThinkingBlock", (), {"type": "thinking", "thinking": "reasoning"})()
        text_block = type("TextBlock", (), {"type": "text", "text": json.dumps(payload)})()
        self.content = [thinking_block, text_block]


class _FakeMessages:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.last_call_kwargs: dict | None = None

    async def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        return _FakeMessage(self._payload)


class _FakeAnthropicClient:
    def __init__(self, payload: dict) -> None:
        self.messages = _FakeMessages(payload)


_VALID_PAYLOAD = {
    "response_a": {"coherence": 3, "relevance": 4, "completeness": 2, "groundedness": 5, "unverifiable_claims": ["claim x"]},
    "response_b": {"coherence": 5, "relevance": 5, "completeness": 5, "groundedness": 5, "unverifiable_claims": []},
}


async def test_score_parses_both_responses_from_the_judge_json():
    fake_client = _FakeAnthropicClient(_VALID_PAYLOAD)
    judge = ClaudeJudge(client=fake_client, model_id="claude-opus-5")

    scores_a, scores_b = await judge.score(query="q", response_a="a", response_b="b")

    assert scores_a.coherence == 3
    assert scores_a.unverifiable_claims == ["claim x"]
    assert scores_b.coherence == 5
    assert scores_b.unverifiable_claims == []


async def test_score_includes_the_query_and_both_responses_in_the_request():
    fake_client = _FakeAnthropicClient(_VALID_PAYLOAD)
    judge = ClaudeJudge(client=fake_client, model_id="claude-opus-5")

    await judge.score(query="What is FastAPI?", response_a="Response one.", response_b="Response two.")

    sent = fake_client.messages.last_call_kwargs
    full_prompt = str(sent["messages"])
    assert "What is FastAPI?" in full_prompt
    assert "Response one." in full_prompt
    assert "Response two." in full_prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_claude_judge.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `evaluation/infrastructure/claude_judge.py`**

```python
from __future__ import annotations

import json
from typing import Any, cast

from evaluation.domain.entities import JudgeScores
from evaluation.domain.ports import Judge

_JUDGE_SYSTEM_PROMPT = """You are an impartial judge comparing two responses to the same query.

Score EACH response independently on these four dimensions, 1-5 each, per this rubric:
- coherence: 1 = disjointed or self-contradictory, 3 = holds together but has rough transitions, 5 = reads as a single well-organized answer
- relevance: 1 = misses the actual question asked, 3 = addresses the question but includes significant off-topic material, 5 = tightly focused on what was actually asked
- completeness: 1 = leaves out information the query clearly needed, 3 = covers the main point but skips secondary detail the query implied it wanted, 5 = covers everything the query needed at the depth it implied
- groundedness: 1 = asserts specific checkable facts with no support anywhere, 3 = mixes grounded and unverifiable claims, 5 = every specific checkable claim traces to the provided context

Score both responses on all four dimensions before comparing them to each other. Also flag, per response, any specific claim you cannot verify against the query or context provided -- a real flagged claim, never a bare "seems fine" or "seems off."

Respond with ONLY this JSON shape, no other text, no markdown fencing:
{"response_a": {"coherence": <int>, "relevance": <int>, "completeness": <int>, "groundedness": <int>, "unverifiable_claims": [<str>, ...]}, "response_b": {"coherence": <int>, "relevance": <int>, "completeness": <int>, "groundedness": <int>, "unverifiable_claims": [<str>, ...]}}
"""


class ClaudeJudge(Judge):
    def __init__(self, client: Any, model_id: str) -> None:
        # Typed as Any rather than anthropic.AsyncAnthropic for the same reason
        # as ClaudeChatModel: stays substitutable by the unit tests' fake.
        self._client = client
        self._model_id = model_id

    async def score(self, query: str, response_a: str, response_b: str) -> tuple[JudgeScores, JudgeScores]:
        response = await self._client.messages.create(
            model=self._model_id,
            max_tokens=4096,
            system=_JUDGE_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}",
                }
            ],
        )
        # Same reason as ClaudeChatModel: claude-opus-5 runs adaptive thinking by
        # default, and thinking blocks lead the content list with no .text.
        text = ""
        for block in response.content:
            if block.type == "text":
                text = cast(str, block.text)
                break

        parsed = json.loads(text)
        return (
            JudgeScores(
                coherence=parsed["response_a"]["coherence"],
                relevance=parsed["response_a"]["relevance"],
                completeness=parsed["response_a"]["completeness"],
                groundedness=parsed["response_a"]["groundedness"],
                unverifiable_claims=parsed["response_a"]["unverifiable_claims"],
            ),
            JudgeScores(
                coherence=parsed["response_b"]["coherence"],
                relevance=parsed["response_b"]["relevance"],
                completeness=parsed["response_b"]["completeness"],
                groundedness=parsed["response_b"]["groundedness"],
                unverifiable_claims=parsed["response_b"]["unverifiable_claims"],
            ),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_claude_judge.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add evaluation/infrastructure/claude_judge.py tests/unit/test_claude_judge.py
git commit -m "feat: add Claude judge adapter for qualitative scoring

Refs Epic #144"
```

---

### Task 4: Markdown report renderer

**Files:**
- Create: `evaluation/infrastructure/markdown_report.py`
- Test: `tests/unit/test_markdown_report.py`

**Interfaces:**
- Consumes: `ComparisonResult`, `RunResult`, `JudgeScores` (Task 1).
- Produces: `render(result) -> str` and `render_github_comment(result) -> str`, consumed by any future comparison script and by Task 5's own end-to-end test.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_markdown_report.py
from evaluation.domain.entities import Answer, ComparisonResult, JudgeScores, RunResult
from evaluation.infrastructure.markdown_report import render, render_github_comment


def _make_result() -> ComparisonResult:
    baseline = RunResult(
        label="Baseline", latency_p50_ms=200.0, latency_p95_ms=250.0,
        total_input_tokens=100, total_output_tokens=50, task_success_rate=0.9,
        answers=[Answer(text="baseline answer", input_tokens=100, output_tokens=50)],
    )
    treatment = RunResult(
        label="Treatment", latency_p50_ms=150.0, latency_p95_ms=180.0,
        total_input_tokens=100, total_output_tokens=30, task_success_rate=0.95,
        answers=[Answer(text="treatment answer", input_tokens=100, output_tokens=30)],
    )
    scores_a = JudgeScores(coherence=4, relevance=4, completeness=3, groundedness=5, unverifiable_claims=["a stray claim"])
    scores_b = JudgeScores(coherence=5, relevance=5, completeness=5, groundedness=5)
    return ComparisonResult(
        scenario_name="Fixed Size Chunking", model_config="qwen3.5, Ollama, Q4_K_M",
        success_criterion="retrieval hit against a known-relevant chunk",
        baseline=baseline, treatment=treatment, judge_scores=[(scores_a, scores_b)],
    )


def test_render_includes_the_quantitative_table_with_a_computed_delta():
    output = render(_make_result())

    assert "Fixed Size Chunking" in output
    assert "qwen3.5, Ollama, Q4_K_M" in output
    assert "100" in output  # input tokens
    assert "50" in output  # baseline output tokens
    assert "30" in output  # treatment output tokens
    assert "-40.0%" in output  # (30-50)/50 * 100


def test_render_includes_qualitative_scores_and_flagged_claims():
    output = render(_make_result())

    assert "Coherence" in output
    assert "a stray claim" in output


def test_render_github_comment_is_non_empty_and_shorter_or_equal_to_full_report():
    result = _make_result()
    full = render(result)
    comment = render_github_comment(result)

    assert len(comment) > 0
    assert len(comment) <= len(full)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_markdown_report.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `evaluation/infrastructure/markdown_report.py`**

```python
from __future__ import annotations

from evaluation.domain.entities import ComparisonResult, JudgeScores


def _delta_pct(baseline: float, treatment: float) -> str:
    if baseline == 0:
        return "n/a"
    pct = (treatment - baseline) / baseline * 100
    return f"{pct:+.1f}%"


def _quantitative_table(result: ComparisonResult) -> list[str]:
    b, t = result.baseline, result.treatment
    output_delta = _delta_pct(b.total_output_tokens, t.total_output_tokens)
    latency_delta = _delta_pct(b.latency_p50_ms, t.latency_p50_ms)
    return [
        "## Quantitative",
        "",
        "| Run | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline |",
        "|---|---|---|---|---|---|---|",
        f"| Baseline | {result.model_config} | {b.total_input_tokens} | {b.total_output_tokens} "
        f"| {b.latency_p50_ms:.0f}ms / {b.latency_p95_ms:.0f}ms | {b.task_success_rate:.0%} | — |",
        f"| Treatment | {result.model_config} | {t.total_input_tokens} | {t.total_output_tokens} "
        f"| {t.latency_p50_ms:.0f}ms / {t.latency_p95_ms:.0f}ms | {t.task_success_rate:.0%} "
        f"| {output_delta} output tokens, {latency_delta} p50 latency |",
        "",
    ]


def _score_row(label: str, scores: JudgeScores) -> str:
    return f"| {label} | {scores.coherence} | {scores.relevance} | {scores.completeness} | {scores.groundedness} |"


def _qualitative_section(result: ComparisonResult) -> list[str]:
    lines = ["## Qualitative (per question)", ""]
    for i, (scores_a, scores_b) in enumerate(result.judge_scores, start=1):
        lines.append(f"### Question {i}")
        lines.append("")
        lines.append("| Response | Coherence | Relevance | Completeness | Groundedness |")
        lines.append("|---|---|---|---|---|")
        lines.append(_score_row("Baseline (A)", scores_a))
        lines.append(_score_row("Treatment (B)", scores_b))
        lines.append("")
        if scores_a.unverifiable_claims:
            lines.append(f"- Baseline unverifiable claims: {', '.join(scores_a.unverifiable_claims)}")
        if scores_b.unverifiable_claims:
            lines.append(f"- Treatment unverifiable claims: {', '.join(scores_b.unverifiable_claims)}")
        lines.append("")
    return lines


def render(result: ComparisonResult) -> str:
    lines = [
        f"# {result.scenario_name} — Comparison Result",
        "",
        f"**Model:** {result.model_config}",
        f"**Success criterion:** {result.success_criterion}",
        "",
        *_quantitative_table(result),
        *_qualitative_section(result),
    ]
    return "\n".join(lines)


def render_github_comment(result: ComparisonResult) -> str:
    lines = [
        f"**Model:** {result.model_config}",
        "",
        *_quantitative_table(result),
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_markdown_report.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add evaluation/infrastructure/markdown_report.py tests/unit/test_markdown_report.py
git commit -m "feat: add markdown report renderer

Refs Epic #144"
```

---

### Task 5: Scenario loader

**Files:**
- Create: `evaluation/scenarios/loader.py`
- Create: `evaluation/scenarios/_fixtures/smoke-test/queries.yaml` (a tiny fixture scenario used only by this task's own test — not a real technique scenario)
- Test: `tests/unit/test_scenario_loader.py`

**Interfaces:**
- Produces: `Scenario`, `ScenarioQuestion` entities, `load_scenario(scenario_dir) -> Scenario`, consumed by any future comparison script.

- [ ] **Step 1: Write the fixture scenario file**

```yaml
# evaluation/scenarios/_fixtures/smoke-test/queries.yaml
name: smoke-test
questions:
  - question: "What is this fixture for?"
    success_criterion: "answer mentions 'fixture' or 'test'"
  - question: "Is this a real technique scenario?"
    success_criterion: "answer says no"
```

Also create an empty `evaluation/scenarios/_fixtures/smoke-test/corpus/.gitkeep` file so the (currently empty) corpus directory exists on disk.

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/test_scenario_loader.py
from pathlib import Path

from evaluation.scenarios.loader import load_scenario

_FIXTURE_DIR = Path(__file__).parent.parent.parent / "evaluation" / "scenarios" / "_fixtures" / "smoke-test"


def test_load_scenario_reads_name_and_questions():
    scenario = load_scenario(_FIXTURE_DIR)

    assert scenario.name == "smoke-test"
    assert len(scenario.questions) == 2
    assert scenario.questions[0].question == "What is this fixture for?"
    assert scenario.questions[0].success_criterion == "answer mentions 'fixture' or 'test'"


def test_load_scenario_resolves_the_corpus_directory():
    scenario = load_scenario(_FIXTURE_DIR)

    assert scenario.corpus_dir == _FIXTURE_DIR / "corpus"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_scenario_loader.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Write `evaluation/scenarios/loader.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ScenarioQuestion:
    question: str
    success_criterion: str


@dataclass(frozen=True)
class Scenario:
    name: str
    corpus_dir: Path
    questions: list[ScenarioQuestion]


def load_scenario(scenario_dir: Path) -> Scenario:
    queries_path = scenario_dir / "queries.yaml"
    data = yaml.safe_load(queries_path.read_text(encoding="utf-8"))
    questions = [
        ScenarioQuestion(question=q["question"], success_criterion=q["success_criterion"])
        for q in data["questions"]
    ]
    return Scenario(name=data["name"], corpus_dir=scenario_dir / "corpus", questions=questions)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_scenario_loader.py -v`
Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add evaluation/scenarios/loader.py evaluation/scenarios/_fixtures/ tests/unit/test_scenario_loader.py
git commit -m "feat: add scenario loader

Refs Epic #144"
```

---

### Task 6: End-to-end wiring test

**Files:**
- Test: `tests/unit/test_evaluation_harness_end_to_end.py`

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: nothing new — this task proves the whole harness composes correctly (scenario → RunComparison → markdown report) before any real technique comparison depends on it.

- [ ] **Step 1: Write the end-to-end test**

```python
# tests/unit/test_evaluation_harness_end_to_end.py
from pathlib import Path

from evaluation.application.run_comparison import RunComparison
from evaluation.domain.entities import Answer
from evaluation.infrastructure.markdown_report import render
from evaluation.scenarios.loader import load_scenario
from tests.unit.evaluation_fakes import FakeJudge

_FIXTURE_DIR = Path(__file__).parent.parent.parent / "evaluation" / "scenarios" / "_fixtures" / "smoke-test"


async def test_full_harness_pipeline_produces_a_readable_report():
    scenario = load_scenario(_FIXTURE_DIR)

    async def baseline(question: str) -> Answer:
        return Answer(text=f"a baseline answer with no fixture awareness", input_tokens=20, output_tokens=10)

    async def treatment(question: str) -> Answer:
        return Answer(text=f"a treatment answer mentioning this is a test fixture", input_tokens=20, output_tokens=8)

    use_case = RunComparison(judge=FakeJudge(), repeat_count=2)
    result = await use_case.execute(
        scenario_name=scenario.name,
        model_config="qwen3.5, Ollama",
        success_criterion="see per-question criteria",
        questions=[q.question for q in scenario.questions],
        baseline=baseline,
        treatment=treatment,
        success_check=lambda q, a: True,
    )

    report = render(result)

    assert scenario.name in report
    assert "Quantitative" in report
    assert "Qualitative" in report
    assert len(result.judge_scores) == len(scenario.questions)
```

- [ ] **Step 2: Run the test to verify it fails, then passes**

Run: `uv run pytest tests/unit/test_evaluation_harness_end_to_end.py -v`
Expected: fails first (if `evaluation_fakes.py`'s `FakeJudge` import path or `_FIXTURE_DIR` import from another test module doesn't resolve — fix any import issue, no production code changes should be needed since Tasks 1-5 already built everything this test exercises), then `1 passed`.

- [ ] **Step 3: Run the full unit suite to confirm no regressions**

Run: `uv run pytest tests/unit/ -v`
Expected: all passing, including every test from Tasks 1-5 plus this one.

- [ ] **Step 4: Run ruff and mypy**

Run: `uv run ruff check evaluation/ tests/unit/test_evaluation_domain.py tests/unit/test_run_comparison.py tests/unit/test_claude_judge.py tests/unit/test_markdown_report.py tests/unit/test_scenario_loader.py tests/unit/test_evaluation_harness_end_to_end.py tests/unit/evaluation_fakes.py`
Run: `uv run mypy evaluation/`
Expected: both clean. Fix anything either flags before committing.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_evaluation_harness_end_to_end.py
git commit -m "test: verify the evaluation harness pipeline end to end

Refs Epic #144"
```

---

## After this plan: what it unlocks

This harness is the dependency every remaining RAG/MAG/CAG technique and combination Story needs to close its "real measured result documented" DoD item. The next sub-project is the first real consumer: a batched plan covering the RAG Chunking Strategies cluster (issues #48, #82, #84, #85, #87, #88 — Fixed Size chunking, already implemented in `FixedSizeChunker`, needs its first real measured comparison against a real scenario using this harness; the other five chunking strategies need both implementation and measurement).
