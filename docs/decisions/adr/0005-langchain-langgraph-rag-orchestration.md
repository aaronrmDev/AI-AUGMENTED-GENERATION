# ADR-0005: LangChain and LangGraph for RAG Orchestration

Status: Accepted

## Context

The RAG layer requires an orchestration approach for two distinct kinds of work. The first is pipeline assembly: chunking, embedding, hybrid retrieval, reranking, and context compression need to be wired together in a maintainable way, with room to swap individual stages. The second is agentic control flow: techniques like CRAG (corrective RAG, which validates and re-routes retrieval results) and Self-RAG (which gates whether retrieval happens at all) are naturally expressed as loops and conditional branches rather than a straight-line pipeline, and need a runtime that can represent that structure directly rather than approximating it with ad hoc control code.

## Decision

LangChain is used for RAG pipeline orchestration — chunking, embedding, hybrid search, and reranking stages are composed using its abstractions. LangGraph is used specifically for agent workflows that require graph-based control flow, namely CRAG and Self-RAG.

## Consequences

This pairing gives the project a mature, extensively integrated ecosystem for the linear parts of the RAG pipeline, reducing the amount of custom glue code needed to connect embedding models, vector stores, and rerankers. For the non-linear parts — corrective validation loops and retrieval gating — LangGraph's graph-based execution model maps directly onto the CRAG and Self-RAG patterns, avoiding the awkward workarounds that a purely linear framework would require. The trade-off is a dependency on two frameworks instead of one, with LangChain's abstractions and versioning behavior (and LangGraph's, where it diverges) becoming a long-term constraint on how the RAG layer evolves, and any change to either library's API surface propagating into this project's pipeline and agent-loop code.
