from collections import defaultdict
from datetime import datetime, timedelta
from typing import Generic, TypeVar

K = TypeVar("K")


class SlidingWindowCounter(Generic[K]):
    """The real sliding-window-with-retention counting algorithm shared by
    InMemoryAccessFrequencyTracker (CAG, tenant-wide) and
    InMemoryUserScopedAccessFrequencyTracker (MAG, per-user). Both are
    pure in-memory, sync, no-I/O dict counters that previously duplicated
    this exact logic near line-for-line, differing only in what their key
    tuple contains -- a review finding caught that duplication as a real
    (if low-severity) maintenance risk, since InMemoryAccessFrequencyTracker's
    own retention logic had already needed one correctness fix before this
    extraction (pruning against a per-call `window` instead of a fixed
    `retention` would silently corrupt results whenever two callers
    queried with different window sizes -- fixing that once here, instead
    of separately in each tracker, is the whole point of sharing this).
    This class doesn't know or care what K actually is; each tracker's own
    port decides its key shape.
    """

    def __init__(self, retention: timedelta) -> None:
        self._retention = retention
        self._events: dict[K, list[datetime]] = defaultdict(list)

    def record(self, key: K, at: datetime) -> None:
        self._events[key].append(at)
        self._prune(key, now=at)

    def count(self, key: K, window: timedelta, now: datetime) -> int:
        self._prune(key, now)
        cutoff = now - window
        return sum(1 for at in self._events.get(key, []) if cutoff <= at <= now)

    def keys(self) -> list[K]:
        return list(self._events)

    def _prune(self, key: K, now: datetime) -> None:
        cutoff = now - self._retention
        events = self._events.get(key)
        if events is None:
            return
        kept = [at for at in events if at >= cutoff]
        if kept:
            self._events[key] = kept
        else:
            del self._events[key]
