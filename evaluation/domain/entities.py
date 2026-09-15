from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Answer:
    text: str
    input_tokens: int
    output_tokens: int
    # What the model actually had available to ground its answer in -- "" for
    # a no-retrieval baseline, the retrieved chunks (or equivalent) for a
    # treatment. Defaulted so paradigms with no retrieval concept at all can
    # skip it; RunComparison passes it straight through to the judge so
    # groundedness can be checked against something real instead of guessed.
    context: str = ""


@dataclass(frozen=True)
class RunResult:
    label: str
    latency_p50_ms: float
    latency_p95_ms: float
    total_input_tokens: int
    total_output_tokens: int
    task_success_rate: float
    answers: list[Answer]
    # One entry per question, each the fraction of that question's own
    # repeats that passed success_check (#147) -- task_success_rate above
    # stays the single aggregate figure the quantitative table already
    # renders, this is the per-question breakdown a reader needs to tell
    # "every question mostly passed" apart from "one question always failed,
    # the rest always passed" behind the same aggregate number. Defaulted to
    # an empty list so existing RunResult call sites and test fixtures that
    # predate this field keep constructing validly.
    per_question_success_rate: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class JudgeScores:
    coherence: int
    relevance: int
    completeness: int
    groundedness: int
    unverifiable_claims: list[str] = field(default_factory=list)
    # True when the judge's raw response never parsed into real scores after
    # every retry (#149) -- coherence/relevance/completeness/groundedness are
    # meaningless placeholders (0) in that case, never a real judged score.
    parse_failed: bool = False


@dataclass(frozen=True)
class ComparisonResult:
    scenario_name: str
    model_config: str
    success_criterion: str
    rag: bool
    cag: bool
    mag: bool
    notes: str
    baseline: RunResult
    treatment: RunResult
    judge_scores: list[tuple[JudgeScores, JudgeScores]]
