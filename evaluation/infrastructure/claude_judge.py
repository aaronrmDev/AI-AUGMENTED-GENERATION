from __future__ import annotations

import json
from typing import Any, cast

from evaluation.domain.entities import JudgeScores
from evaluation.domain.ports import Judge

_JUDGE_SYSTEM_PROMPT = """You are an impartial judge comparing two responses to the same query.

Score EACH response independently on these four dimensions, 1-5 each, per this rubric:
- coherence: 1 = disjointed or self-contradictory, 3 = holds together but has rough transitions, 5 = reads as a single well-organized answer
- relevance: 1 = misses the actual question asked, 3 = addresses the question but includes significant off-topic material, 5 = tightly focused on what was actually asked
- completeness: 1 = leaves out information the query clearly needed, 3 = covers the main point but skips secondary detail the query implied it wanted, 5 = covers everything the query needed at the depth it implied
- groundedness: 1 = asserts specific checkable facts with no support anywhere, 3 = mixes grounded and unverifiable claims, 5 = every specific checkable claim traces to the provided context

Score both responses on all four dimensions before comparing them to each other. Also flag, per response, any specific claim you cannot verify against the query or context provided -- a real flagged claim, never a bare "seems fine" or "seems off."

Respond with ONLY this JSON shape, no other text, no markdown fencing:
{"response_a": {"coherence": <int>, "relevance": <int>, "completeness": <int>, "groundedness": <int>, "unverifiable_claims": [<str>, ...]}, "response_b": {"coherence": <int>, "relevance": <int>, "completeness": <int>, "groundedness": <int>, "unverifiable_claims": [<str>, ...]}}
"""


class ClaudeJudge(Judge):
    def __init__(self, client: Any, model_id: str) -> None:
        # Typed as Any rather than anthropic.AsyncAnthropic for the same reason
        # as ClaudeChatModel: stays substitutable by the unit tests' fake.
        self._client = client
        self._model_id = model_id

    async def score(self, query: str, response_a: str, response_b: str) -> tuple[JudgeScores, JudgeScores]:
        response = await self._client.messages.create(
            model=self._model_id,
            max_tokens=4096,
            system=_JUDGE_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}",
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
