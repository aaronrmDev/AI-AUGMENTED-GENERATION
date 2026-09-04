# ADR-0002: Qdrant over Milvus/Weaviate as the Vector Database

Status: Accepted

## Context

The RAG layer needs a vector database capable of semantic search over document chunks, with metadata filtering for multi-tenant isolation and support for hybrid (vector + keyword) retrieval patterns. Milvus and Weaviate stood as viable alternatives to Qdrant, each with a mature ecosystem of its own, but the choice mattered well beyond a feature checklist: the vector store is a foundational dependency that RAG retrieval, semantic memory, and episodic memory embeddings in the MAG layer all read and write through, and the local development experience for the whole team hinges on how easy the chosen store is to run and operate.

## Decision

Qdrant is the project's vector database, selected over Milvus and Weaviate, and it backs the `documents`, `semantic_memory`, and `episodic_memory` collections described in the database schema.

## Consequences

Choosing Qdrant commits the project to a simpler operational model than Milvus's distributed architecture would require, while still getting excellent performance from its Rust-based core, and it commits the project just as much to Qdrant's own way of doing things: its approach to hybrid search and payload-based metadata filtering as the mechanism for tenant isolation at the vector layer, and its conventions for HNSW indexing — Hierarchical Navigable Small World, the graph-based approximate-nearest-neighbor algorithm that keeps similarity search fast as the corpus grows — reflected directly in the `ef_construct` and `m` parameters used elsewhere in this project's schema. Local development benefits directly: standing up Qdrant for a laptop-scale RAG pipeline is straightforward, which keeps the Phase 1 foundation work unblocked. The trade-off is that any feature unique to Milvus or Weaviate — for instance, Milvus's larger-scale distributed deployment story — is not available without a future migration.
