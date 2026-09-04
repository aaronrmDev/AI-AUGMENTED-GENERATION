from __future__ import annotations

import json
from typing import Any, cast

from evaluation.domain.entities import JudgeScores
from evaluation.domain.ports import Judge
from evaluation.infrastructure._judge_prompt import JUDGE_SYSTEM_PROMPT, build_judge_user_message


class ClaudeJudge(Judge):
    def __init__(self, client: Any, model_id: str) -> None:
        # Typed as Any rather than anthropic.AsyncAnthropic for the same reason
        # as ClaudeChatModel: stays substitutable by the unit tests' fake.
        self._client = client
        self._model_id = model_id

    async def score(
        self,
        query: str,
        response_a: str,
        response_b: str,
        context_a: str,
        context_b: str,
        reference_context: str = "",
    ) -> tuple[JudgeScores, JudgeScores]:
        response = await self._client.messages.create(
            model=self._model_id,
            max_tokens=4096,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": build_judge_user_message(
                        query, response_a, response_b, context_a, context_b, reference_context
                    ),
                }
            ],
        )
        # Same reason as ClaudeChatModel: claude-opus-5 runs adaptive thinking by
        # default, and thinking blocks lead the content list with no .text.
        text = ""
        for block in response.content:
            if block.type == "text":
                text = cast(str, block.text)
                break

        parsed = json.loads(text)
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
