from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class ProbeObservation:
    arm: str
    source_key: str
    context: str
    current_marker: str
    superseded_markers: tuple[str, ...]
    served_from_cag: bool
    # Text that must never reach this probe, such as another user's personal fact.
    foreign_markers: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlacementTally:
    arm: str
    source_key: str
    probes: int
    stale_only: int
    mixed: int
    fresh_only: int
    neither: int
    cag_served: int
    stale_from_cag: int
    foreign_exposures: int

    @property
    def stale_rate(self) -> float:
        # nan, not 0.0, over zero probes: "never measured" must not read as "never stale".
        return self.stale_only / self.probes if self.probes else math.nan

    @property
    def cag_share(self) -> float:
        return self.cag_served / self.probes if self.probes else math.nan


def tally_probes(observations: Sequence[ProbeObservation]) -> list[PlacementTally]:
    groups: dict[tuple[str, str], list[ProbeObservation]] = {}
    for observation in observations:
        groups.setdefault((observation.arm, observation.source_key), []).append(observation)
    tallies: list[PlacementTally] = []
    for (arm, key), group in groups.items():
        counts = {"stale_only": 0, "mixed": 0, "fresh_only": 0, "neither": 0}
        stale_from_cag = 0
        for o in group:
            stale = any(marker in o.context for marker in o.superseded_markers)
            fresh = o.current_marker in o.context
            if stale:
                label = "mixed" if fresh else "stale_only"
            else:
                label = "fresh_only" if fresh else "neither"
            counts[label] += 1
            stale_from_cag += int(stale and o.served_from_cag)
        tallies.append(
            PlacementTally(
                arm=arm,
                source_key=key,
                probes=len(group),
                stale_only=counts["stale_only"],
                mixed=counts["mixed"],
                fresh_only=counts["fresh_only"],
                neither=counts["neither"],
                cag_served=sum(o.served_from_cag for o in group),
                stale_from_cag=stale_from_cag,
                foreign_exposures=sum(
                    any(marker in o.context for marker in o.foreign_markers) for o in group
                ),
            )
        )
    return tallies


class PreloadTracker:
    """Counts pre-loads per (arm, source), and how many were never served from."""

    def __init__(self) -> None:
        self._generations: dict[tuple[str, str], list[bool]] = {}

    def preloaded(self, arm: str, source_key: str) -> None:
        self._generations.setdefault((arm, source_key), []).append(False)

    def served(self, arm: str, source_key: str) -> None:
        generations = self._generations.get((arm, source_key))
        if generations:
            generations[-1] = True

    def counts(self, arm: str, source_key: str) -> tuple[int, int]:
        generations = self._generations.get((arm, source_key), [])
        return len(generations), generations.count(False)


@dataclass(frozen=True)
class MigrationObservation:
    source_key: str
    from_route: str
    to_route: str
    shift_at: datetime
    migrated_at: datetime

    @property
    def lag(self) -> timedelta:
        return self.migrated_at - self.shift_at


@dataclass(frozen=True)
class RouteRow:
    data_type: str
    declared_change: str
    scope: str
    source_paradigm: str
    route: str
    agrees: bool
