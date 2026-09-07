"""Live verification of the two claims CAG.md's Scheduling paragraph makes:
that requests sharing a prefix get batched so that prefix is computed once
for the group, and that continuous in-flight batching keeps utilisation high
without waiting for a batch to fill.

Both are true, and measuring them separately is the point -- CAG.md states
them as one bundled benefit, while at this scale they turn out to differ in
magnitude by two orders of magnitude. See
evaluation/reports/cag-cache-aware-batching.md.

Requires a real vLLM server with --enable-prefix-caching. Cannot run in CI
(no ROCm GPU on GitHub Actions), so it skips itself with a stated reason,
and runs from inside WSL2 because this machine's mirrored WSL networking
does not expose the server's port to the Windows host.
"""
import os
import uuid

import httpx
import pytest

from src.cag.infrastructure.vllm_completion_client import VLLMCompletionClient
from src.cag.infrastructure.vllm_metrics_client import VLLMMetricsClient

_BASE_URL = os.environ.get("CAG_VLLM_BASE_URL", "http://192.168.0.243:8001")
_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_MAX_TOKENS = 20
_BURST = 16

_SHARED_PREFIX = (
    "You are a careful assistant answering questions about distributed systems, "
    "consensus protocols, replication, and fault tolerance in production infrastructure. "
) * 8


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


def test_a_concurrent_burst_is_served_in_flight_rather_than_serialised():
    completion_client = VLLMCompletionClient(base_url=_BASE_URL, model=_MODEL)
    completion_client.complete_timed("warm up the engine please", 5)

    sequential_prompts = [f"{_SHARED_PREFIX} Question {i}: explain briefly." for i in range(_BURST)]
    import time

    started = time.perf_counter()
    for prompt in sequential_prompts:
        completion_client.complete(prompt, _MAX_TOKENS)
    sequential = time.perf_counter() - started

    concurrent_prompts = [
        f"{_SHARED_PREFIX} Question {100 + i}: explain briefly." for i in range(_BURST)
    ]
    texts, concurrent = completion_client.complete_concurrent_timed(
        concurrent_prompts, _MAX_TOKENS
    )

    print(
        f"\n{_BURST} sequential: {sequential:.3f}s\n"
        f"{_BURST} concurrent: {concurrent:.3f}s  speedup={sequential / concurrent:.1f}x"
    )
    assert len(texts) == _BURST
    # Measured 7.6x; asserted well below that so the test reports a real
    # regression rather than ordinary run-to-run variance.
    assert concurrent < sequential / 2


def test_a_shared_prefix_burst_reuses_blocks_while_a_distinct_prefix_burst_cannot():
    completion_client = VLLMCompletionClient(base_url=_BASE_URL, model=_MODEL)
    metrics_client = VLLMMetricsClient(base_url=_BASE_URL)
    completion_client.complete_timed("warm up the engine please", 5)

    shared_prompts = [
        f"{_SHARED_PREFIX} Distinct question {200 + i}: explain briefly." for i in range(_BURST)
    ]
    before = metrics_client.prefix_cache_hits_total()
    _, shared_latency = completion_client.complete_concurrent_timed(shared_prompts, _MAX_TOKENS)
    shared_hits = metrics_client.prefix_cache_hits_total() - before

    # Each prompt gets its OWN salt, placed at the very start. Two
    # earlier attempts got this wrong in instructive ways: with no salt
    # the prompts were byte-identical to a previous run's and hit 1632
    # tokens left cached by it, and with one salt shared across the
    # burst the identical leading "Unrelated <salt> prompt" text still
    # filled a whole 16-token block, hitting 480. Only a per-prompt
    # salt in leading position makes these genuinely share nothing --
    # "distinct" has to mean distinct from each other AND never seen by
    # this server.
    distinct_prompts = [
        f"{f'{uuid.uuid4().hex} subject {i} discussion paragraph. ' * 8} Question: explain briefly."
        for i in range(_BURST)
    ]
    before = metrics_client.prefix_cache_hits_total()
    _, distinct_latency = completion_client.complete_concurrent_timed(
        distinct_prompts, _MAX_TOKENS
    )
    distinct_hits = metrics_client.prefix_cache_hits_total() - before

    print(
        f"\nshared-prefix burst:  {shared_latency:.3f}s  cache_hit_tokens={shared_hits:.0f}\n"
        f"distinct-prefix burst: {distinct_latency:.3f}s  cache_hit_tokens={distinct_hits:.0f}"
    )

    # The robust, deterministic half of the claim: sharing is real and
    # block-granular, and requests with nothing in common share nothing.
    assert shared_hits > 0
    assert shared_hits % 16 == 0
    assert distinct_hits == 0
    assert shared_latency < distinct_latency

    # The wall-clock consequence: measured 0.166s against 0.253s, a
    # 1.52x difference. Worth recording how this number was nearly
    # missed -- an earlier version of this test reused prompts a prior
    # run had already cached, which made the "distinct" baseline
    # artificially fast and shrank the apparent benefit to about 7%.
    # The margin is large enough to assert, but the two hit-count
    # assertions above remain the load-bearing ones, since they are
    # deterministic where any timing comparison is not.
