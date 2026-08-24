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
