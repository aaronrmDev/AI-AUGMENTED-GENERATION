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
    scores_a = JudgeScores(
        coherence=4, relevance=4, completeness=3, groundedness=5,
        unverifiable_claims=["a stray claim"],
    )
    scores_b = JudgeScores(coherence=5, relevance=5, completeness=5, groundedness=5)
    return ComparisonResult(
        scenario_name="Fixed Size Chunking", model_config="qwen3.5, Ollama, Q4_K_M",
        success_criterion="retrieval hit against a known-relevant chunk",
        rag=True, cag=False, mag=False, notes="fixture corpus, 1 question",
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


def test_quantitative_table_marks_only_the_treatment_row_with_the_active_paradigm():
    output = render(_make_result())

    lines = output.splitlines()
    baseline_row = next(line for line in lines if line.startswith("| Baseline"))
    treatment_row = next(line for line in lines if line.startswith("| Treatment"))

    # Baseline is always all-paradigm-off by definition.
    assert baseline_row == (
        "| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama, Q4_K_M | 100 | 50 "
        "| 200ms / 250ms | 90% | — | fixture corpus, 1 question |"
    )
    # Treatment reflects the caller-supplied rag=True, cag=False, mag=False.
    assert treatment_row.startswith("| Treatment | ✓ | ✗ | ✗ |")
    assert "fixture corpus, 1 question" in treatment_row


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


def test_render_github_comment_includes_both_quantitative_and_qualitative_sections():
    result = _make_result()
    comment = render_github_comment(result)

    assert "Quantitative" in comment
    assert "Qualitative" in comment
    assert "Coherence" in comment
    assert "a stray claim" in comment  # the unverifiable claim from _make_result()'s fixture


def test_render_github_comment_omits_the_title_and_success_criterion_line():
    result = _make_result()
    full = render(result)
    comment = render_github_comment(result)

    assert "— Comparison Result" in full
    assert "— Comparison Result" not in comment
    assert "Success criterion" not in comment


def test_render_includes_retained_answer_text_per_question():
    # #147: the answer text a report's own success/judge scores are actually
    # about was computed but never rendered -- a reader had no way to check
    # a contradiction between the scores and what was actually said.
    output = render(_make_result())

    assert "baseline answer" in output
    assert "treatment answer" in output


def test_render_omits_answer_text_gracefully_when_answers_list_is_shorter_than_questions():
    # Defensive: _qualitative_section() indexes into result.baseline.answers
    # by question position -- a mismatched (shorter) answers list must not
    # raise IndexError, just skip the answer-text lines for that question.
    baseline = RunResult(
        label="Baseline", latency_p50_ms=1.0, latency_p95_ms=1.0,
        total_input_tokens=0, total_output_tokens=0, task_success_rate=1.0,
        answers=[],  # no retained answer at all
    )
    treatment = RunResult(
        label="Treatment", latency_p50_ms=1.0, latency_p95_ms=1.0,
        total_input_tokens=0, total_output_tokens=0, task_success_rate=1.0,
        answers=[],
    )
    scores = JudgeScores(coherence=3, relevance=3, completeness=3, groundedness=3)
    result = ComparisonResult(
        scenario_name="s", model_config="m", success_criterion="c",
        rag=True, cag=False, mag=False, notes="",
        baseline=baseline, treatment=treatment, judge_scores=[(scores, scores)],
    )

    output = render(result)  # must not raise

    assert "Baseline answer:" not in output
    assert "Treatment answer:" not in output


def test_render_includes_per_question_success_rate_when_present():
    baseline = RunResult(
        label="Baseline", latency_p50_ms=200.0, latency_p95_ms=250.0,
        total_input_tokens=100, total_output_tokens=50, task_success_rate=0.6,
        answers=[Answer(text="baseline answer", input_tokens=100, output_tokens=50)],
        per_question_success_rate=[0.6],
    )
    treatment = RunResult(
        label="Treatment", latency_p50_ms=150.0, latency_p95_ms=180.0,
        total_input_tokens=100, total_output_tokens=30, task_success_rate=0.8,
        answers=[Answer(text="treatment answer", input_tokens=100, output_tokens=30)],
        per_question_success_rate=[0.8],
    )
    scores_a = JudgeScores(coherence=4, relevance=4, completeness=3, groundedness=5)
    scores_b = JudgeScores(coherence=5, relevance=5, completeness=5, groundedness=5)
    result = ComparisonResult(
        scenario_name="Fixed Size Chunking", model_config="qwen3.5, Ollama",
        success_criterion="c", rag=True, cag=False, mag=False, notes="",
        baseline=baseline, treatment=treatment, judge_scores=[(scores_a, scores_b)],
    )

    output = render(result)

    assert "baseline 60%" in output
    assert "treatment 80%" in output


def test_render_omits_per_question_success_rate_when_absent():
    # A ComparisonResult built before #147 (or by any scenario that never
    # populates per_question_success_rate) must render exactly as before --
    # no "Task success this question" line, no IndexError.
    output = render(_make_result())  # _make_result()'s RunResults default to []

    assert "Task success this question" not in output


def test_render_shows_a_dash_row_instead_of_zeros_for_a_parse_failure():
    # Regression test for #149: a JudgeScores with parse_failed=True carries
    # placeholder 0s that must never be rendered as if they were real judged
    # scores on the 1-5 rubric.
    result = _make_result()
    failed_scores = JudgeScores(
        coherence=0,
        relevance=0,
        completeness=0,
        groundedness=0,
        unverifiable_claims=["JUDGE PARSE FAILURE: JSONDecodeError: boom"],
        parse_failed=True,
    )
    result.judge_scores[0] = (failed_scores, result.judge_scores[0][1])

    output = render(result)

    lines = output.splitlines()
    baseline_score_row = next(
        line for line in lines if line.startswith("| Baseline (A)")
    )
    assert baseline_score_row == "| Baseline (A) | — | — | — | — |"
    assert "0 |" not in baseline_score_row
    assert "⚠ Baseline: JUDGE PARSE FAILURE: JSONDecodeError: boom" in output


def test_render_shows_a_dash_row_for_a_treatment_parse_failure_too():
    # The Baseline/scores_a and Treatment/scores_b branches in
    # _qualitative_section are hand-duplicated, not one shared code path --
    # this covers the symmetric case the test above doesn't touch, so a
    # copy-paste error in the scores_b branch can't slip past both silently.
    result = _make_result()
    failed_scores = JudgeScores(
        coherence=0,
        relevance=0,
        completeness=0,
        groundedness=0,
        unverifiable_claims=["JUDGE PARSE FAILURE: JSONDecodeError: boom"],
        parse_failed=True,
    )
    result.judge_scores[0] = (result.judge_scores[0][0], failed_scores)

    output = render(result)

    lines = output.splitlines()
    treatment_score_row = next(
        line for line in lines if line.startswith("| Treatment (B)")
    )
    assert treatment_score_row == "| Treatment (B) | — | — | — | — |"
    assert "0 |" not in treatment_score_row
    assert "⚠ Treatment: JUDGE PARSE FAILURE: JSONDecodeError: boom" in output


def test_render_github_comment_shows_a_dash_row_for_a_parse_failure():
    # render_github_comment() shares _qualitative_section with render(), but
    # had no direct test coverage of the parse-failure path at all.
    result = _make_result()
    failed_scores = JudgeScores(
        coherence=0,
        relevance=0,
        completeness=0,
        groundedness=0,
        unverifiable_claims=["JUDGE PARSE FAILURE: JSONDecodeError: boom"],
        parse_failed=True,
    )
    result.judge_scores[0] = (failed_scores, result.judge_scores[0][1])

    comment = render_github_comment(result)

    lines = comment.splitlines()
    baseline_score_row = next(
        line for line in lines if line.startswith("| Baseline (A)")
    )
    assert baseline_score_row == "| Baseline (A) | — | — | — | — |"
    assert "⚠ Baseline: JUDGE PARSE FAILURE: JSONDecodeError: boom" in comment
