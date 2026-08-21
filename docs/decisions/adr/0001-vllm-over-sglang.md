# ADR-0001: vLLM over SGLang as the Primary Serving Engine

Status: Accepted

## Context

The system's CAG layer depends on a serving engine that can turn prefix caching, PagedAttention-style block management, and KV cache manipulation (eviction, compression, speculative decoding) into working, production-grade infrastructure rather than research prototypes. Two credible candidates existed for this role: vLLM and SGLang. Both support high-throughput LLM inference with some form of cache reuse, so the decision was less about raw capability and more about which engine the rest of the stack — deployment tooling, documentation, community support, and native prefix caching — could be built around with the least risk.

## Decision

vLLM is the primary serving engine for this project. It is the default for all standard inference paths, including the CAG cache-optimization stack described elsewhere in this project's architecture. SGLang remains an acceptable choice for specific agent workflows where its structured-program execution model and automatic KV reuse are a better fit — for example, complex multi-step agent loops — but it is not the default and is not required anywhere in the core serving path.

## Consequences

Standardizing on vLLM locks the project into its ecosystem: its native prefix caching, its PagedAttention implementation, its release cadence, and its (broader, more mature) documentation and community. This gives the team a well-trodden path for the Foundation and CAG layers, with fewer surprises during deployment and fewer gaps to fill with custom code. The trade-off is reduced flexibility — any future desire to lean more heavily on SGLang's agent-oriented features will mean running two serving engines side by side (as already permitted for specific agent workflows) rather than migrating wholesale, and any vLLM-specific limitation becomes a project-wide constraint rather than one that can be routed around by switching engines.
