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
        rag: bool,
        cag: bool,
        mag: bool,
        questions: list[str],
        baseline: Strategy,
        treatment: Strategy,
        success_check: SuccessCheck,
        notes: str = "",
    ) -> ComparisonResult:
        baseline_result = await self._run_strategy("Baseline", baseline, questions, success_check)
        treatment_result = await self._run_strategy(
            "Treatment", treatment, questions, success_check
        )

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
            rag=rag,
            cag=cag,
            mag=mag,
            notes=notes,
            baseline=baseline_result,
            treatment=treatment_result,
            judge_scores=judge_scores,
        )
