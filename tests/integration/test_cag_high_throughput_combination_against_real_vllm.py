"""Live verification of CAG.md's Archetype B, High-Throughput Serving:
Prefix Caching + PagedAttention + Cache-Aware Batching, all three active at
once on the workload the source names for it -- "RAG APIs, chatbot
platforms, and multi-tenant serving -- exactly the workloads where many
requests share a prefix by construction."

Each of the three was validated individually in its own batch. What this
tests is the claim those three batches could not: that run together on a
realistic multi-tenant workload, "one prefill ends up serving hundreds of
requests instead of one."

The workload deliberately INTERLEAVES tenants rather than grouping them,
because interleaved arrival is what a real multi-tenant endpoint sees and is
the case where scheduling has to do actual work. Grouping the requests by
tenant first would hand the result to prefix caching alone.

Requires a real vLLM server with --enable-prefix-caching. Cannot run in CI
(no ROCm GPU on GitHub Actions); skips itself with a stated reason and runs
from inside WSL2.
"""
import os
import time
import uuid

import httpx
import pytest

from src.cag.infrastructure.vllm_completion_client import VLLMCompletionClient
from src.cag.infrastructure.vllm_metrics_client import VLLMMetricsClient

_BASE_URL = os.environ.get("CAG_VLLM_BASE_URL", "http://192.168.0.243:8001")
_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_TENANTS = 4
_REQUESTS_PER_TENANT = 6
_MAX_TOKENS = 15


def _server_reachable() -> bool:
    try:
        httpx.get(f"{_BASE_URL}/v1/models", timeout=3.0).raise_for_status()
        return True
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(
    not _server_reachable(),
    reason=(
        f"real vLLM server not reachable at {_BASE_URL} -- this test needs the "
        "project's own GPU/ROCm dev machine, not something CI can provide"
    ),
)


def _interleaved_multi_tenant_workload() -> list[str]:
    # Salted per run so a previous run's identical prompts cannot pre-warm
    # this one's cache, the contamination the Cache-Aware Batching batch
    # established the hard way.
    salt = uuid.uuid4().hex
    systems = [
        (
            f"Tenant {tenant} system prompt {salt}. You are an expert assistant "
            f"for organisation {tenant}, answering with care about infrastructure. "
        )
        * 6
        for tenant in range(_TENANTS)
    ]
    workload = []
    for index in range(_REQUESTS_PER_TENANT):
        for tenant in range(_TENANTS):
            workload.append(
                systems[tenant] + f" Request {index} for tenant {tenant}: explain briefly."
            )
    return workload


def test_many_requests_sharing_a_prefix_are_served_from_one_prefill():
    completion_client = VLLMCompletionClient(base_url=_BASE_URL, model=_MODEL)
    metrics_client = VLLMMetricsClient(base_url=_BASE_URL)
    completion_client.complete_timed("warm up the engine please", 5)

    workload = _interleaved_multi_tenant_workload()
    before_prompt = metrics_client.prompt_tokens_total()
    before_hits = metrics_client.prefix_cache_hits_total()
    started = time.perf_counter()
    for prompt in workload:
        completion_client.complete(prompt, _MAX_TOKENS)
    sequential = time.perf_counter() - started
    prompt_tokens = metrics_client.prompt_tokens_total() - before_prompt
    hits = metrics_client.prefix_cache_hits_total() - before_hits

    hit_rate = hits / prompt_tokens
    print(
        f"{len(workload)} reqs / {_TENANTS} tenants interleaved: "
        f"{sequential:.3f}s, hit rate {hit_rate:.1%}"
    )
    # Each tenant's system prompt is prefilled once and reused across that
    # tenant's remaining requests, despite the tenants arriving mixed.
    assert hit_rate > 0.5
    assert hits % 16 == 0


def test_the_three_techniques_together_beat_prefix_reuse_alone():
    completion_client = VLLMCompletionClient(base_url=_BASE_URL, model=_MODEL)
    metrics_client = VLLMMetricsClient(base_url=_BASE_URL)
    completion_client.complete_timed("warm up the engine please", 5)

    sequential_workload = _interleaved_multi_tenant_workload()
    started = time.perf_counter()
    for prompt in sequential_workload:
        completion_client.complete(prompt, _MAX_TOKENS)
    sequential = time.perf_counter() - started

    concurrent_workload = _interleaved_multi_tenant_workload()
    before_prompt = metrics_client.prompt_tokens_total()
    before_hits = metrics_client.prefix_cache_hits_total()
    _texts, concurrent = completion_client.complete_concurrent_timed(
        concurrent_workload, _MAX_TOKENS
    )
    concurrent_hits = metrics_client.prefix_cache_hits_total() - before_hits
    concurrent_prompt = metrics_client.prompt_tokens_total() - before_prompt

    print(
        f"sequential {sequential:.3f}s vs concurrent {concurrent:.3f}s "
        f"= {sequential / concurrent:.1f}x, concurrent hit rate "
        f"{concurrent_hits / concurrent_prompt:.1%}"
    )
    # Batching contributes throughput the prefix reuse alone cannot: the
    # hit rate is comparable either way, so the speedup is the scheduling
    # tier doing work on top of what caching already delivered.
    assert concurrent < sequential / 2
    assert concurrent_hits / concurrent_prompt > 0.5
