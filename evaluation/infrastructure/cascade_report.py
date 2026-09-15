from __future__ import annotations

import math

from evaluation.domain.cascade_metrics import (
    AllocatorTally,
    StaleAnswerTally,
    TierLatencySummary,
)


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.0%}"


def _tier_row(s: TierLatencySummary) -> str:
    """A tier's columns from Tier through Within budget, closing pipe included."""
    outcomes = ", ".join(f"{outcome.value} {n}" for outcome, n in s.outcomes.items())
    return (
        f"{s.paradigm.value.upper()} | {s.attempts} | {outcomes} | {s.p50_ms:.2f} "
        f"| {s.p95_ms:.2f} | {s.budget_ms:.0f} | {_pct(s.within_budget_rate)} |"
    )


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
    lines += [f"| {_tier_row(s)}" for s in tiers]
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


def render_retriever_measurements(
    retrievers: list[tuple[str, list[TierLatencySummary]]], notes: str
) -> str:
    """One row per tier under each RAG retriever composed behind RagTier."""
    lines = [
        "# Orchestration Meta-Layer — RAG Retrievers in a PARALLEL Route",
        "",
        notes,
        "",
        "## Tier latency against Concept 5's budgets, per RAG retriever",
        "",
        "| Retriever | Tier | Attempts | Outcomes | p50 ms | p95 ms | Budget ms | Within budget |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [f"| {name} | {_tier_row(s)}" for name, tiers in retrievers for s in tiers]
    return "\n".join(lines) + "\n"
