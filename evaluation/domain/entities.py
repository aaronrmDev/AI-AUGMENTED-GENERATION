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
