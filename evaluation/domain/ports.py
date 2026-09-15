from __future__ import annotations

from abc import ABC, abstractmethod

from evaluation.domain.entities import JudgeScores


class Judge(ABC):
    @abstractmethod
    async def score(
        self,
        query: str,
        response_a: str,
        response_b: str,
        context_a: str,
        context_b: str,
        reference_context: str = "",
    ) -> tuple[JudgeScores, JudgeScores]: ...
