from __future__ import annotations

import json
from typing import Any

from evaluation.domain.entities import JudgeScores
from evaluation.domain.ports import Judge
from evaluation.infrastructure._judge_prompt import JUDGE_SYSTEM_PROMPT


class OllamaJudge(Judge):
    # Deviation from the harness design (docs/evaluation/COMPARISON_METHODOLOGY.md
    # assumes a Claude judge): built only because the Anthropic account had no
    # credit balance to run ClaudeJudge live. Same model family (qwen3.5) scores
    # both the baseline and the treatment it is judging here -- a real
    # self-grading-bias risk a stronger, independent judge model doesn't have.
    # Every report produced with this judge says so explicitly in its notes
    # field; re-running with ClaudeJudge once credits exist is the correct
    # follow-up, not a silent accept of this as the final measurement.
    def __init__(self, client: Any, model_id: str) -> None:
        # Typed as Any rather than ollama.AsyncClient for the same reason as
        # OllamaChatModel: stays substitutable by the unit tests' fake.
        self._client = client
        self._model_id = model_id

    async def score(
        self, query: str, response_a: str, response_b: str, context_a: str, context_b: str
    ) -> tuple[JudgeScores, JudgeScores]:
        last_error: Exception | None = None
        for _ in range(_MAX_SCORE_ATTEMPTS):
            response = await self._client.chat(
                model=self._model_id,
                format="json",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Query: {query}\n\n"
                            f"Context for Response A (use this to judge Response A's "
                            f"groundedness -- a claim that doesn't trace to this is "
                            f"unverifiable):\n"
                            f"{context_a or '(none provided)'}\n\n"
                            f"Response A:\n{response_a}\n\n"
                            f"Context for Response B (use this to judge Response B's "
                            f"groundedness -- a claim that doesn't trace to this is "
                            f"unverifiable):\n"
                            f"{context_b or '(none provided)'}\n\n"
                            f"Response B:\n{response_b}"
                        ),
                    },
                ],
            )
            try:
                parsed = json.loads(response.message.content or "{}")
                return _to_scores(parsed)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                last_error = exc
                continue

        failure = _parse_failure(f"{type(last_error).__name__}: {last_error}")
        return failure, failure


# Bounded so one bad Ollama completion doesn't burn unlimited time/tokens, but
# high enough that transient malformed JSON (observed once in this project's
# history, see #149) gets a real chance to self-correct on retry.
_MAX_SCORE_ATTEMPTS = 3


def _parse_failure(reason: str) -> JudgeScores:
    return JudgeScores(
        coherence=0,
        relevance=0,
        completeness=0,
        groundedness=0,
        unverifiable_claims=[f"JUDGE PARSE FAILURE: {reason}"],
        parse_failed=True,
    )


def _to_scores(parsed: dict[str, Any]) -> tuple[JudgeScores, JudgeScores]:
    return (
        JudgeScores(
            coherence=parsed["response_a"]["coherence"],
            relevance=parsed["response_a"]["relevance"],
            completeness=parsed["response_a"]["completeness"],
            groundedness=parsed["response_a"]["groundedness"],
            unverifiable_claims=parsed["response_a"]["unverifiable_claims"],
        ),
        JudgeScores(
            coherence=parsed["response_b"]["coherence"],
            relevance=parsed["response_b"]["relevance"],
            completeness=parsed["response_b"]["completeness"],
            groundedness=parsed["response_b"]["groundedness"],
            unverifiable_claims=parsed["response_b"]["unverifiable_claims"],
        ),
    )
