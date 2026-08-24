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
