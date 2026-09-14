from __future__ import annotations

import math

from evaluation.domain.freshness_metrics import MigrationObservation, PlacementTally, RouteRow


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.0%}"


def render_freshness_measurements(
    routes: list[RouteRow],
    tallies: list[PlacementTally],
    preloads: dict[tuple[str, str], tuple[int, int]],
    migrations: list[MigrationObservation],
    ttl_tallies: list[PlacementTally],
    notes: str,
) -> str:
    lines = [
        "# Freshness-Aware Data Router — Measurements",
        "",
        notes,
        "",
        "## Concept 9's freshness spectrum, routed",
        "",
        "| Data type | Declared change | Scope | Source's paradigm | Route | Agrees |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r.data_type} | {r.declared_change} | {r.scope} | {r.source_paradigm} "
        f"| {r.route} | {'yes' if r.agrees else 'no'} |"
        for r in routes
    ]
    lines += [
        "",
        "## Placement ablation over 30 simulated days",
        "",
        "| Arm | Source | Probes | Stale only | Mixed | Current only | Neither | Stale rate "
        "| Served from CAG | Pre-loads | Never-served pre-loads | Foreign exposures |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in tallies:
        preloaded, unused = preloads.get((t.arm, t.source_key), (0, 0))
        lines.append(
            f"| {t.arm} | {t.source_key} | {t.probes} | {t.stale_only} | {t.mixed} "
            f"| {t.fresh_only} | {t.neither} | {_pct(t.stale_rate)} | {_pct(t.cag_share)} "
            f"| {preloaded} | {unused} | {t.foreign_exposures} |"
        )
    lines += [
        "",
        "## Migration lag",
        "",
        "| Source | Migration | Pattern shift | Migrated | Lag | Pre-loads before "
        "| Pre-loads after |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {m.source_key} | {m.from_route} → {m.to_route} "
        f"| {m.shift_at:%Y-%m-%d %H:%M} | {m.migrated_at:%Y-%m-%d %H:%M} "
        f"| {m.lag.total_seconds() / 3600:.1f} h | {m.preloads_before} | {m.preloads_after} |"
        for m in migrations
    ]
    lines += [
        "",
        "## TTL bound on a silently changed cached source",
        "",
        "| Arm | Source | Probes | Superseded text served from CAG |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {t.arm} | {t.source_key} | {t.probes} | {t.stale_from_cag} |" for t in ttl_tallies
    ]
    return "\n".join(lines) + "\n"
