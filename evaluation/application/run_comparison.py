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
        per_question_success_rate: list[float] = []
        total_successes = 0
        total_repeats = 0

        for question in questions:
            repeat_answers: list[Answer] = []
            question_successes = 0
            for _ in range(self._repeat_count):
                start = time.perf_counter()
                answer = await strategy(question)
                latencies_ms.append((time.perf_counter() - start) * 1000)
                repeat_answers.append(answer)
                if success_check(question, answer):
                    question_successes += 1
            # Every repeat is scored (#147) -- task success is an
            # n=repeat_count statistic per question now, not n=1. The last
            # repeat's Answer is still what gets kept for reporting and
            # judging: success_check is cheap (a plain string/pattern match,
            # no model call) and evaluating it per repeat is free, but
            # re-judging every repeat with an LLM judge would multiply judge
            # calls by repeat_count for no benefit -- the last repeat's
            # answer is a representative sample, not the statistic itself.
            last_answers.append(repeat_answers[-1])
            per_question_success_rate.append(question_successes / self._repeat_count)
            total_successes += question_successes
            total_repeats += self._repeat_count

        latencies_ms.sort()
        return RunResult(
            label=label,
            latency_p50_ms=_percentile(latencies_ms, 0.50),
            latency_p95_ms=_percentile(latencies_ms, 0.95),
            total_input_tokens=sum(a.input_tokens for a in last_answers),
            total_output_tokens=sum(a.output_tokens for a in last_answers),
            task_success_rate=total_successes / total_repeats if total_repeats else 0.0,
            answers=last_answers,
            per_question_success_rate=per_question_success_rate,
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
        reference_contexts: list[str] | None = None,
    ) -> ComparisonResult:
        baseline_result = await self._run_strategy("Baseline", baseline, questions, success_check)
        treatment_result = await self._run_strategy(
            "Treatment", treatment, questions, success_check
        )
        # One reference passage per question, or "" for every question when
        # the caller has none (#147) -- Judge.score() already treats an empty
        # reference_context as "no reference given, fall back to each arm's
        # own context," so omitting this parameter entirely reproduces every
        # pre-#147 caller's exact prior behavior.
        references = reference_contexts if reference_contexts is not None else [""] * len(
            questions
        )

        judge_scores = [
            # Each arm is judged against its OWN retrieved context for the
            # hallucination check (#148) -- scoring both against a single
            # shared context structurally penalizes whichever arm didn't
            # supply that context, even when its claims were genuinely
            # grounded in what it actually retrieved. Groundedness itself
            # uses reference_context when the caller supplies one (#147),
            # since judging groundedness against an arm's own (possibly
            # wrong) retrieval can never catch a confidently-wrong answer.
            await self._judge.score(
                question,
                b.text,
                t.text,
                context_a=b.context,
                context_b=t.context,
                reference_context=reference,
            )
            for question, b, t, reference in zip(
                questions,
                baseline_result.answers,
                treatment_result.answers,
                references,
                strict=True,
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
