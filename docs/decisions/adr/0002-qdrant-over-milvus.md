# ADR-0002: Qdrant over Milvus/Weaviate as the Vector Database

Status: Accepted

## Context

The RAG layer needs a vector database capable of semantic search over document chunks, with metadata filtering for multi-tenant isolation and support for hybrid (vector + keyword) retrieval patterns. Milvus and Weaviate were both viable alternatives to Qdrant, each with mature ecosystems of their own. The choice mattered because the vector store is a foundational dependency: RAG retrieval, semantic memory, and episodic memory embeddings in the MAG layer all read and write through it, and the local development experience for the whole team hinges on how easy the chosen store is to run and operate.

## Decision

Qdrant is the project's vector database, used for the `documents`, `semantic_memory`, and `episodic_memory` collections described in the database schema. It is selected over Milvus and Weaviate.

## Consequences

Choosing Qdrant commits the project to a simpler operational model than Milvus's distributed architecture would require, while still getting excellent performance from its Rust-based core. It also commits the project to Qdrant's approach to hybrid search and payload-based metadata filtering as the mechanism for tenant isolation at the vector layer, and to Qdrant's HNSW indexing conventions (as reflected in the `ef_construct` and `m` parameters used elsewhere in this project's schema). Local development benefits directly: standing up Qdrant for a laptop-scale RAG pipeline is straightforward, which keeps the Phase 1 foundation work unblocked. The trade-off is that any feature unique to Milvus or Weaviate — for instance, Milvus's larger-scale distributed deployment story — is not available without a future migration.
