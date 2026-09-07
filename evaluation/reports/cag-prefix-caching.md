# CAG Batch D: Prefix Caching — Live Measurement Report

**Scope:** #32 (story), #127 (task). See `docs/superpowers/specs/2026-09-06-cag-prefix-caching-design.md` for the design and for why this batch validates real vLLM behavior rather than implementing a mechanism of its own — vLLM's PagedAttention block manager already does the caching; this batch proves it works as CAG.md claims, on this project's own hardware.

## What CAG.md asks for, and what this batch built

CAG.md calls Prefix Caching "the highest-ROI single optimization in the whole pipeline": if a request's leading tokens match a prefix already computed for an earlier request, prefill is skipped for that stretch. `docs/testing/TESTING.md` commits this specific technique to validation against real vLLM, because the Latency-Adaptive Fallback Cascade trusts a cache hit unconditionally, and only a real serving engine can demonstrate that trust is warranted. This batch built the boundary a future orchestration layer would call through it — `CompletionServer` (`src/cag/domain/ports.py`), implemented by `VLLMCompletionClient` (`src/cag/infrastructure/vllm_completion_client.py`) against vLLM's OpenAI-compatible `/v1/completions` endpoint — and `VLLMMetricsClient` (`src/cag/infrastructure/vllm_metrics_client.py`), which reads vLLM's own Prometheus `/metrics` for the real, server-side signal of whether a hit actually happened.

## Real measurement, against a live vLLM server

Server: `vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8001 --enable-prefix-caching --gpu-memory-utilization 0.5` on this project's AMD 7900 XTX (ROCm 7.2.0, vLLM 0.28.0, `~/cag-vllm/.venv`). Three requests, after a throwaway warm-up call on unrelated content to absorb one-time engine warm-up cost (see below):

| Request | Prefix | Continuation | Latency | Prefix-cache hits (delta) |
|---|---|---|---|---|
| Cold | ~120-token shared prefix (first use) | "What is the capital of France?" | 0.2112s | 0 |
| Warm | Same shared prefix, different continuation | "What is the capital of Japan?" | 0.0938s | **144** |
| Miss | Genuinely different prefix (diverges in its first sentence) | "What is the capital of France?" | 0.1383s | 0 |

The central claim CAG.md makes about this technique — reuse across *different* requests that merely share a prefix, not just an identical repeated prompt — is confirmed directly: the warm request shares its long prefix with the cold request but asks a different question, and registered a real, server-observed 144-token cache hit while being 2.25x faster than the cold request that populated it. The genuinely different prefix (the miss) registered zero hits — no stale reuse of the cold/warm requests' cached blocks — confirming the invalidation-adjacent correctness property `docs/testing/TESTING.md` specifically named as the reason this technique can't be validated with a mock.

## A real measurement-methodology finding, disclosed rather than smoothed over

The first run of this test, with no warm-up call, measured cold=0.2153s and miss=0.0454s — the "miss," which should be no faster than "cold" since neither benefits from real prefix reuse, came in over 4x *faster*. The cause wasn't caching at all: it was request order. The cold request was the server's first real request since startup and paid a one-time engine/kernel warm-up cost (CUDA graph capture, allocator warm state) that every subsequent request, hit or miss, gets to skip. Adding a throwaway warm-up request on unrelated content before the three measured requests fixed this: cold and miss now land in the same rough neighborhood (0.21s, 0.14s) while warm is clearly fastest (0.09s) — the real caching effect, isolated from a real but unrelated confound. This is exactly the kind of measurement-honesty finding this project's evaluation reports have disclosed before (MiniCache's compression-ceiling formula, Medusa's warm-start acceptance ceiling) rather than silently reordering conditions to hide it.

## A genuine environment constraint, disclosed rather than worked around silently

This project's WSL2 instance runs in "mirrored" networking mode, which is supposed to make a WSL-bound port reachable from the Windows host directly. Empirically, it doesn't for this server: `Test-NetConnection -ComputerName <eth0-ip> -Port 8001` from native Windows PowerShell returned `False`, while `curl` to the identical address from inside WSL2 succeeded immediately. Rather than spend more time debugging Windows Firewall or WSL networking configuration — out of scope for validating a caching algorithm — this batch's integration test runs from inside WSL2 itself, using the `~/cag-vllm/.venv` Python (which already has `httpx` and `pytest` installed alongside `vllm`/`torch`), invoked directly against the test file rather than through this project's `tests/integration/conftest.py` (which needs `alembic` and this project's full dependency set, not installed in the minimal vLLM venv).

## What this batch does not do

- **Does not reimplement vLLM's own prefix-caching or PagedAttention mechanism** — that's vLLM's job; this batch validates it.
- **Does not wire into the Latency-Adaptive Fallback Cascade** — that layer isn't built yet; `CompletionServer` is a real, usable boundary for it, not the cascade itself.
- **Does not test eviction under real memory pressure** — a real, valuable next validation, but separately scoped; this batch's Definition of Done (hit reuse across different prompts, correct non-reuse on a genuinely different prefix) doesn't require it.
- **Does not run in CI** — the first CAG batch of which this is true. GitHub Actions has no ROCm GPU; the integration test skips itself cleanly with a stated reason when the server isn't reachable, rather than failing CI or silently passing.

## The numbers

Unit tests: 654 (start of batch, after Eviction) → 664 (10 new: 5 for `VLLMCompletionClient`, 5 for `VLLMMetricsClient`, both against a stubbed `httpx.MockTransport`, not a real server). Integration tests: 208 (TestContainers-backed, CI-runnable) unaffected — this batch's one real-vLLM integration test is deliberately outside that count's CI-run path, verified locally via `~/cag-vllm/.venv/bin/python -m pytest` from inside WSL2 rather than the project's own Windows-side `.venv`.
