import uuid
from collections import defaultdict
from datetime import datetime, timedelta

from src.orchestration.domain.ports import UserScopedAccessFrequencyTracker

_Key = tuple[uuid.UUID, uuid.UUID, uuid.UUID]  # (tenant_id, user_id, document_id)
_DEFAULT_RETENTION = timedelta(days=30)


class InMemoryUserScopedAccessFrequencyTracker(UserScopedAccessFrequencyTracker):
    """A real sliding-window access counter, scoped per user -- MAG's warm
    tier is personal, unlike CAG's tenant-wide shared cache, so this
    tracks (tenant_id, user_id, document_id) rather than
    InMemoryAccessFrequencyTracker's (tenant_id, document_id).

    Prunes raw access timestamps older than `retention` on every read, the
    same fixed-horizon-independent-of-any-individual-call's-window design
    InMemoryAccessFrequencyTracker (Batch D) established -- pruning
    against a per-call window instead would silently corrupt results
    whenever two callers queried with different window sizes.
    """

    def __init__(self, retention: timedelta = _DEFAULT_RETENTION) -> None:
        self._retention = retention
        self._accesses: dict[_Key, list[datetime]] = defaultdict(list)

    def record_access(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID, at: datetime
    ) -> None:
        key = (tenant_id, user_id, document_id)
        self._accesses[key].append(at)
        self._prune(key, now=at)

    def access_count(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        window: timedelta,
        now: datetime,
    ) -> int:
        key = (tenant_id, user_id, document_id)
        self._prune(key, now)
        cutoff = now - window
        return sum(1 for at in self._accesses.get(key, []) if cutoff <= at <= now)

    def most_accessed(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, n: int, window: timedelta, now: datetime
    ) -> list[uuid.UUID]:
        counts = {
            document_id: self.access_count(tenant_id, user_id, document_id, window, now)
            for (tid, uid, document_id) in list(self._accesses)
            if tid == tenant_id and uid == user_id
        }
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return [document_id for document_id, count in ranked if count > 0][:n]

    def _prune(self, key: _Key, now: datetime) -> None:
        cutoff = now - self._retention
        accesses = self._accesses.get(key)
        if accesses is None:
            return
        kept = [at for at in accesses if at >= cutoff]
        if kept:
            self._accesses[key] = kept
        else:
            del self._accesses[key]
