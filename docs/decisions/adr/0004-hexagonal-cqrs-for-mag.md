# ADR-0004: Hexagonal Architecture with CQRS for MAG

Status: Accepted

## Context

The MAG (Memory-Augmented Generation) layer has a workload shape that is unusual compared to typical CRUD storage: writes are frequent, small, and largely asynchronous — every interaction, tool call, and outcome can generate a new episodic memory entry — while reads are comparatively rare but far more complex, needing to combine semantic similarity, temporal recency, causal relevance, graph traversal, and salience scoring into a single ranked result. A single read/write model struggles to serve both needs well: optimizing the write path for throughput tends to hurt the read path's ability to do multi-strategy retrieval, and vice versa. The project also commits elsewhere (Section 4.3) to a Hexagonal Architecture with a framework-free domain layer, which needs a clear pattern for how memory operations plug into that structure.

## Decision

MAG memory operations use CQRS (Command Query Responsibility Segregation) within the project's Hexagonal Architecture: write models and read models for memory are separated. Writes (episodic memory capture, consolidation into semantic memory, procedural memory updates) go through a command path optimized for frequent, async persistence. Reads (multi-strategy retrieval combining semantic, temporal, causal, graph, and salience signals) go through a separate query path optimized for that complexity, independent of how the underlying data was written.

## Consequences

This separation allows the write and read paths for memory to be scaled, tuned, and evolved independently — the command side can be optimized purely for ingestion throughput without regard to query complexity, and the query side can add new retrieval strategies without touching how memories are captured. It also reinforces the domain/application/infrastructure boundaries already required by the Hexagonal Architecture rule, since commands and queries naturally map to distinct use cases in the application layer. The cost is added structural complexity: every memory-related feature now needs a command-side and a query-side implementation instead of one unified model, and keeping the two consistent (for example, ensuring a newly consolidated semantic fact is visible to the next multi-strategy read) requires deliberate synchronization rather than falling out automatically from a shared model.
