from __future__ import annotations

import math

from evaluation.domain.cascade_metrics import (
    AllocatorTally,
    StaleAnswerTally,
    TierLatencySummary,
)


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.0%}"


def render_cascade_measurements(
    tiers: list[TierLatencySummary],
    tallies: list[StaleAnswerTally],
    allocators: list[AllocatorTally],
    notes: str,
) -> str:
    lines = [
        "# Orchestration Meta-Layer — Cascade Measurements",
        "",
        notes,
        "",
        "## Tier latency against Concept 5's budgets",
        "",
        "| Tier | Attempts | Outcomes | p50 ms | p95 ms | Budget ms | Within budget |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in tiers:
        outcomes = ", ".join(f"{outcome.value} {n}" for outcome, n in s.outcomes.items())
        lines.append(
            f"| {s.paradigm.value.upper()} | {s.attempts} | {outcomes} | {s.p50_ms:.2f} "
            f"| {s.p95_ms:.2f} | {s.budget_ms:.0f} | {_pct(s.within_budget_rate)} |"
        )
    lines += [
        "",
        "## Superseded frozen-cache text reaching the model, router off vs. on",
        "",
        "| Arm | Runs | Stale only | Current only | Both | Stale rate |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {t.arm} | {t.runs} | {t.stale} | {t.fresh} | {t.mixed} | {_pct(t.stale_rate)} |"
        for t in tallies
    ]
    lines += [
        "",
        "## Dynamic reallocation vs. static base slices",
        "",
        "| Window tokens | Turns | Dropped (dynamic) | Dropped (static) "
        "| Tokens used (dynamic) | Tokens used (static) |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {a.window_tokens} | {a.turns} | {a.dynamic_dropped} | {a.static_dropped} "
        f"| {a.dynamic_tokens} | {a.static_tokens} |"
        for a in allocators
    ]
    return "\n".join(lines) + "\n"
