import asyncio

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
        rag=True,
        cag=False,
        mag=False,
        questions=["q1", "q2"],
        baseline=baseline,
        treatment=treatment,
        success_check=success_check,
        notes="",
    )

    assert result.scenario_name == "test-scenario"
    assert call_count["baseline"] == 6  # 2 questions x 3 repeats
    assert call_count["treatment"] == 6
    assert result.baseline.task_success_rate == 1.0
    assert result.treatment.task_success_rate == 1.0
    assert result.baseline.total_output_tokens == 10  # 5 x 2 questions, last-run answers only
    assert result.treatment.total_output_tokens == 6  # 3 x 2 questions
    assert len(result.judge_scores) == 2  # one pair per question
    assert result.rag is True
    assert result.cag is False
    assert result.mag is False
    assert result.notes == ""


async def test_run_comparison_computes_latency_percentiles():
    async def instant(question: str) -> Answer:
        return Answer(text="ok", input_tokens=1, output_tokens=1)

    use_case = RunComparison(judge=FakeJudge(), repeat_count=5)
    result = await use_case.execute(
        scenario_name="latency-test", model_config="qwen3.5", success_criterion="n/a",
        rag=True, cag=False, mag=False,
        questions=["q1"], baseline=instant, treatment=instant,
        success_check=lambda q, a: True,
    )

    assert result.baseline.latency_p50_ms >= 0.0
    assert result.baseline.latency_p95_ms >= result.baseline.latency_p50_ms


async def test_run_comparison_passes_each_arms_own_context_to_the_judge():
    async def baseline(question: str) -> Answer:
        return Answer(text="base", input_tokens=1, output_tokens=1, context="baseline's own chunk")

    async def treatment(question: str) -> Answer:
        return Answer(
            text="treat", input_tokens=1, output_tokens=1, context="treatment's own chunk"
        )

    judge = FakeJudge()
    use_case = RunComparison(judge=judge, repeat_count=1)
    await use_case.execute(
        scenario_name="s", model_config="m", success_criterion="c",
        rag=True, cag=False, mag=False,
        questions=["only question"], baseline=baseline, treatment=treatment,
        success_check=lambda q, a: True,
    )

    # Each arm is judged against its OWN retrieved context, not a shared one
    # (#148) -- a baseline that retrieves something different from treatment
    # must have its claims checked against what it actually retrieved.
    assert judge.calls == [
        ("only question", "base", "treat", "baseline's own chunk", "treatment's own chunk")
    ]


async def test_run_comparison_passes_an_empty_baseline_context_through_unchanged():
    # A no-retrieval baseline (plain LLM call) has nothing of its own to be
    # judged against -- confirms the fix doesn't force a fabricated context
    # onto that arm.
    async def baseline(question: str) -> Answer:
        return Answer(text="base", input_tokens=1, output_tokens=1)

    async def treatment(question: str) -> Answer:
        return Answer(text="treat", input_tokens=1, output_tokens=1, context="retrieved chunk")

    judge = FakeJudge()
    use_case = RunComparison(judge=judge, repeat_count=1)
    await use_case.execute(
        scenario_name="s", model_config="m", success_criterion="c",
        rag=True, cag=False, mag=False,
        questions=["only question"], baseline=baseline, treatment=treatment,
        success_check=lambda q, a: True,
    )

    assert judge.calls == [("only question", "base", "treat", "", "retrieved chunk")]


async def test_run_comparison_keeps_only_the_last_repeat_runs_answer():
    call_count = {"n": 0}

    async def counting_strategy(question: str) -> Answer:
        call_count["n"] += 1
        return Answer(text=f"call number {call_count['n']}", input_tokens=1, output_tokens=1)

    use_case = RunComparison(judge=FakeJudge(), repeat_count=3)
    result = await use_case.execute(
        scenario_name="s", model_config="m", success_criterion="c",
        rag=True, cag=False, mag=False,
        questions=["only question"], baseline=counting_strategy, treatment=counting_strategy,
        success_check=lambda q, a: True,
    )

    # 3 repeats for baseline (calls 1-3) then 3 for treatment (calls 4-6); each
    # RunResult should keep only its OWN last call's answer, not the first.
    assert result.baseline.answers[0].text == "call number 3"
    assert result.treatment.answers[0].text == "call number 6"


async def test_run_comparison_pools_latency_across_every_question_and_repeat():
    # 6 calls total: 2 questions x 3 repeats. The one slow call happens during
    # the FIRST question's repeats, not the last -- deliberately, so that a
    # "reset the latency list per question" bug (which would silently keep
    # only the last question's latencies) fails this test instead of passing
    # it by coincidence.
    delays_seconds = iter([0.05, 0.001, 0.001, 0.001, 0.001, 0.001])

    async def variable_delay_strategy(question: str) -> Answer:
        await asyncio.sleep(next(delays_seconds))
        return Answer(text="ok", input_tokens=1, output_tokens=1)

    async def fast_strategy(question: str) -> Answer:
        return Answer(text="ok", input_tokens=1, output_tokens=1)

    use_case = RunComparison(judge=FakeJudge(), repeat_count=3)
    result = await use_case.execute(
        scenario_name="s", model_config="m", success_criterion="c",
        rag=True, cag=False, mag=False,
        questions=["q1", "q2"], baseline=variable_delay_strategy,
        treatment=fast_strategy,
        success_check=lambda q, a: True,
    )

    # If latency were pooled correctly across both questions, p95 must reflect
    # that one slow call from q1's first repeat. If pooling were broken (e.g.
    # the latency list reset per question, keeping only q2's all-fast
    # repeats), p95 would collapse back to ~1ms and this assertion would fail.
    assert result.baseline.latency_p95_ms > 20.0
