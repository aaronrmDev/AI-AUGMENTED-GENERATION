from __future__ import annotations

import math

from evaluation.domain.routing_metrics import RoutingMetrics, RoutingObservation
from src.orchestration.domain.entities import PARADIGM_ORDER, Paradigm


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.0%}"


def _labels(paradigms: frozenset[Paradigm]) -> str:
    return "+".join(p.value.upper() for p in PARADIGM_ORDER if p in paradigms)


def render_router_comparison(
    metrics: dict[str, RoutingMetrics],
    observations: dict[str, list[RoutingObservation]],
    notes: str,
) -> str:
    lines = [
        "# Orchestration Meta-Layer — Router Comparison",
        "",
        notes,
        "",
        "| Classifier | Queries | Exact route | Covers needed | Confident "
        "| CAG P / R | MAG P / R | RAG P / R | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in metrics.items():
        pairs = " | ".join(
            f"{_pct(m.per_paradigm[p].precision)} / {_pct(m.per_paradigm[p].recall)}"
            for p in PARADIGM_ORDER
        )
        lines.append(
            f"| {name} | {m.count} | {_pct(m.exact_match_rate)} | {_pct(m.coverage_rate)} "
            f"| {_pct(m.confident_rate)} | {pairs} "
            f"| {m.latency_p50_ms:.2f} | {m.latency_p95_ms:.2f} |"
        )
    for name, observed in observations.items():
        misrouted = [o for o in observed if o.decision.paradigms != o.expected]
        lines += ["", f"## Misrouted by {name} ({len(misrouted)})", ""]
        lines += [
            f"- {o.query!r}: expected {_labels(o.expected)}, "
            f"routed {_labels(o.decision.paradigms)} ({o.decision.mode.value})"
            for o in misrouted
        ] or ["- none"]
    return "\n".join(lines) + "\n"
