# ADR-0005: LangChain and LangGraph for RAG Orchestration

Status: Superseded by [ADR-0006](0006-hand-rolled-rag-orchestration.md)

**Why this is marked superseded rather than quietly left as "Accepted":** the RAG pipeline that actually got built across six implementation batches (`src/rag/domain/`, `application/`, `infrastructure/`) never adopted LangChain or LangGraph — not `pyproject.toml`, not a single `import` anywhere under `src/`, not one commit message across 228 commits. Every technique this ADR names, including CRAG and Self-RAG, was built as a plain Python class under the same hexagonal domain/application/infrastructure layering the rest of this project uses. This document's own reasoning below is left exactly as it was written, and it is worth being direct about what searching this repository's design specs, commit history, and architecture docs for a reason turned up: nothing. No design spec, no commit message, no later doc revision states that this decision was reconsidered or why the implementation went a different way — it simply wasn't followed, and nothing in the record explains that gap. ADR-0006 records what was actually built and says so plainly, rather than retroactively inventing a rationale this repository's own history doesn't contain.

## Context

The RAG layer requires an orchestration approach for two distinct kinds of work. The first is pipeline assembly: chunking, embedding, hybrid retrieval, reranking, and context compression need to be wired together in a maintainable way, with room to swap individual stages. The second is agentic control flow: techniques like CRAG (corrective RAG, which validates and re-routes retrieval results) and Self-RAG (which gates whether retrieval happens at all) are naturally expressed as loops and conditional branches rather than a straight-line pipeline, and need a runtime that can represent that structure directly rather than approximating it with ad hoc control code.

## Decision

LangChain is used for RAG pipeline orchestration, composing the chunking, embedding, hybrid search, and reranking stages through its abstractions, while LangGraph is reserved for the agent workflows that need graph-based control flow — specifically CRAG and Self-RAG.

## Consequences

This pairing gives the project a mature, extensively integrated ecosystem for the linear parts of the RAG pipeline, reducing the amount of custom glue code needed to connect embedding models, vector stores, and rerankers. For the non-linear parts — corrective validation loops and retrieval gating — LangGraph's graph-based execution model maps directly onto the CRAG and Self-RAG patterns, avoiding the awkward workarounds that a purely linear framework would require. The trade-off is a dependency on two frameworks instead of one, with LangChain's abstractions and versioning behavior (and LangGraph's, where it diverges) becoming a long-term constraint on how the RAG layer evolves, and any change to either library's API surface propagating into this project's pipeline and agent-loop code.
