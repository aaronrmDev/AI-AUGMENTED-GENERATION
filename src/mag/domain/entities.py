from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class EpisodicMemory:
    id: uuid.UUID
    session_id: uuid.UUID
    # input, reasoning trace, tool_calls, output, outcome, actors, entities --
    # the full event, not just the final answer, so a later reflection pass
    # (Consolidation) has something to reason over rather than a single
    # opaque string.
    content: dict[str, Any]
    embedding: list[float]
    timestamp: datetime
    salience_score: float = 0.0
    # None until Consolidation reflects on this episode and extracts
    # whatever durable facts it holds -- excludes it from future
    # consolidation runs afterward, whether or not that reflection actually
    # produced a fact (a genuinely fact-free episode still shouldn't be
    # re-sent to the LLM on every later run).
    consolidated_at: datetime | None = None


@dataclass(frozen=True)
class SemanticMemory:
    id: uuid.UUID
    user_id: uuid.UUID
    fact_key: str
    fact_value: str
    embedding: list[float]
    confidence: float = 1.0
    source: str = ""
    valid_until: datetime | None = None


@dataclass(frozen=True)
class WorkingMemoryTurn:
    role: str
    content: str
    recorded_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProceduralMemory:
    id: uuid.UUID
    user_id: uuid.UUID
    task_pattern: str
    # Steps, tool sequence, whatever the caller wants to record as "how this
    # task gets done" -- deliberately unstructured beyond being JSON-shaped,
    # matching docs/database/DATABASE.md's workflow JSONB column.
    workflow: dict[str, Any] = field(default_factory=dict)
    success_rate: float = 0.0
    last_used: datetime | None = None
