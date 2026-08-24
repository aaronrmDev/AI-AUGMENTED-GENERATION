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
