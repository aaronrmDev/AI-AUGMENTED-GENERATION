import uuid
from datetime import datetime

from src.mag.domain.entities import ScoredEpisode
from src.mag.domain.ports import EpisodicMemoryRepository


class TemporalRetrieval:
    def __init__(self, episodic_memory_repository: EpisodicMemoryRepository) -> None:
        self._episodes = episodic_memory_repository

    async def execute(
        self,
        tenant_id: uuid.UUID,
        session_id: uuid.UUID,
        top_k: int,
        within: tuple[datetime, datetime] | None = None,
    ) -> list[ScoredEpisode]:
        if within is not None:
            start, end = within
            episodes = await self._episodes.get_by_session_in_window(
                session_id, tenant_id, start, end, top_k
            )
            # Membership in the window is binary -- see
            # get_by_session_in_window's docstring -- so every match scores
            # the same rather than inventing a graded "how in-window" signal
            # this system has no basis for.
            return [ScoredEpisode(episode=e, score=1.0) for e in episodes]

        episodes = await self._episodes.get_recent_by_session(
            session_id, tenant_id, limit=top_k
        )
        if not episodes:
            return []
        # No explicit window means recency itself is the relevance signal --
        # linear rank decay over the newest-first list gives fusion (later in
        # this batch) a graded score instead of the window branch's binary
        # 1.0, without claiming a precision this system doesn't have.
        total = len(episodes)
        return [
            ScoredEpisode(episode=e, score=(total - i) / total)
            for i, e in enumerate(episodes)
        ]
