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
    # (Consolidation, Batch B) has something to reason over rather than a
    # single opaque string.
    content: dict[str, Any]
    embedding: list[float]
    timestamp: datetime
    salience_score: float = 0.0


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
