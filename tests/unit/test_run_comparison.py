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
