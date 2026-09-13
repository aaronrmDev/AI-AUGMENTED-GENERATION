from __future__ import annotations

from evaluation.domain.cascade_metrics import (
    AllocatorTally,
    StaleAnswerTally,
    TierLatencySummary,
)


def render_cascade_measurements(
    tiers: list[TierLatencySummary],
    tallies: list[StaleAnswerTally],
    allocator: AllocatorTally,
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
            f"| {s.p95_ms:.2f} | {s.budget_ms:.0f} | {s.within_budget_rate:.0%} |"
        )
    lines += [
        "",
        "## Superseded frozen-cache text reaching the model, router off vs. on",
        "",
        "| Arm | Runs | Stale only | Current only | Both | Stale rate |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {t.arm} | {t.runs} | {t.stale} | {t.fresh} | {t.mixed} | {t.stale_rate:.0%} |"
        for t in tallies
    ]
    lines += [
        "",
        "## Dynamic reallocation vs. static base slices",
        "",
        "| Window tokens | Turns | Dropped (dynamic) | Dropped (static) "
        "| Tokens used (dynamic) | Tokens used (static) |",
        "|---|---|---|---|---|---|",
        f"| {allocator.window_tokens} | {allocator.turns} | {allocator.dynamic_dropped} "
        f"| {allocator.static_dropped} | {allocator.dynamic_tokens} | {allocator.static_tokens} |",
    ]
    return "\n".join(lines) + "\n"
