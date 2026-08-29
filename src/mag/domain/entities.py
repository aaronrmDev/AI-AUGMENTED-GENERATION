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
class ScoredEpisode:
    # Pairs a full EpisodicMemory with a relevance score, matching RAG's
    # SearchResult precedent (src/rag/domain/entities.py) of carrying a
    # score alongside retrieved content rather than discarding it -- unlike
    # SearchResult, this wraps the whole entity instead of flattening
    # fields, since retrieval-strategy consumers (fusion, salience
    # weighting) need timestamp/salience_score/content, not just one field.
    # score's scale depends on which strategy produced it: cosine similarity
    # for semantic search, 0.0-1.0 heuristic relevance for the others -- see
    # the retrieval strategy query classes for what each one means.
    episode: EpisodicMemory
    score: float


@dataclass(frozen=True)
class ScoredFact:
    fact: SemanticMemory
    score: float


@dataclass(frozen=True)
class ActivatedNode:
    # The result shape for spreading activation (MAG Batch D, #76): a node
    # reached by traversing the memory graph outward from a start node, with
    # its distance-decayed relevance. node_type/properties are deliberately
    # generic rather than a typed union of six node dataclasses -- spreading
    # activation's whole point is traversing across heterogeneous node types
    # (User -> Entity -> Fact, etc.) in one pass, so the caller gets back
    # whatever the graph actually reached, not a pre-filtered single type.
    node_id: str
    node_type: str  # "User" | "Session" | "Entity" | "Concept" | "Episode" | "Fact"
    properties: dict[str, Any]
    activation: float
    hops: int


@dataclass(frozen=True)
class GatingCandidate:
    # The normalized wrapper Memory Gating (Batch E) needs to operate over
    # a mixed pool of ScoredEpisode/ScoredFact/ActivatedNode uniformly --
    # same precedent ActivatedNode already set for normalizing across
    # heterogeneous graph node types, applied here one level up, across
    # heterogeneous retrieval STRATEGIES. `origin` stays the single source
    # of truth; the other fields are a flattened, uniform read of it so
    # gating strategies never need isinstance branching on three different
    # source types. See src/mag/application/gating/_candidates.py for how
    # each source type populates these.
    content_text: str
    score: float
    salience: float
    timestamp: datetime | None
    source_type: str  # "episode" | "fact" | "graph_node"
    origin: EpisodicMemory | SemanticMemory | ActivatedNode
    embedding: list[float]


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
