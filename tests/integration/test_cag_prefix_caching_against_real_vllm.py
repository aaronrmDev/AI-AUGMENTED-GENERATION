"""Live verification of KV Cache Prefix Caching against a real vLLM server --
the one CAG technique this project's own testing strategy (docs/testing/
TESTING.md) commits specifically to validating against real vLLM rather than
a hand-rolled stand-in, because the claim under test is the Latency-Adaptive
Fallback Cascade's trust in a cache hit: whether the caching behavior
underneath is actually correct (a shared prefix really gets reused, a
genuinely different prefix is never served a stale hit), not something a
mock or a hand-rolled transformers cache can honestly demonstrate.

Requires a real vLLM server already running with --enable-prefix-caching
(see docs/superpowers/specs/2026-09-06-cag-prefix-caching-design.md for the
exact restart command). Unlike this project's TestContainers-backed
integration tests, this one cannot run in CI at all -- it needs the real
ROCm/vLLM stack on this project's own dev machine, not something a GitHub
Actions runner can provide -- so it skips itself cleanly, with a clear
reason, when the server isn't reachable, rather than failing CI or silently
passing.

Run from inside WSL2 against the WSL2 vLLM venv's own Python
(~/cag-vllm/.venv/bin/python -m pytest), not the project's Windows-side
.venv: this machine's WSL2 "mirrored" networking mode does not make the
vLLM server's port reachable from the Windows host side (confirmed via
Test-NetConnection returning false even though curl succeeds from inside
WSL2 itself), so the test client and the server need to share the same
network namespace.
"""
import os

import httpx
import pytest

from src.cag.infrastructure.vllm_completion_client import VLLMCompletionClient
from src.cag.infrastructure.vllm_metrics_client import VLLMMetricsClient

_BASE_URL = os.environ.get("CAG_VLLM_BASE_URL", "http://192.168.0.243:8001")
_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_MAX_TOKENS = 5

# Long enough (well over one prefix-cache block) that a real hit on it is a
# genuine, measurable prefill saving, not noise.
_SHARED_PREFIX = (
    "You are a helpful assistant with deep knowledge of world capitals, "
    "geography, history, and science. Answer questions accurately and "
    "concisely, citing relevant facts where useful. Always double-check "
    "your reasoning before answering, and prefer well-established facts "
    "over speculation. "
) * 3
_CONTINUATION_A = "What is the capital of France?"
_CONTINUATION_B = "What is the capital of Japan?"
# Diverges from _SHARED_PREFIX in its very first sentence -- a genuinely
# different prefix, not a superset or a substring of it.
_CHANGED_PREFIX = (
    "You are a terse assistant who answers in exactly one word with no "
    "explanation whatsoever, ever, under any circumstances. Answer questions "
    "accurately and concisely, citing relevant facts where useful. Always "
    "double-check your reasoning before answering, and prefer well-established "
    "facts over speculation. "
) * 3
# Unrelated to every prefix above -- exists only to absorb the one-time
# engine/kernel warm-up cost (CUDA graph capture, allocator warm state) a
# server's very first request always pays regardless of caching, which a
# first empirical run of this test showed dominates a "cold" baseline
# measured without it (cold=0.215s vs. a later, unrelated miss=0.045s on
# the same server, purely from warm-up order -- disclosed here rather than
# smoothed over by silently reordering the real conditions to hide it).
_WARM_UP_PROMPT = "The history of the Roman Empire began in "


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
        "project's own GPU/ROCm dev machine (see docs/superpowers/specs/"
        "2026-09-06-cag-prefix-caching-design.md), not something CI can provide"
    ),
)


def test_a_shared_prefix_is_reused_and_a_genuinely_different_prefix_is_not():
    completion_client = VLLMCompletionClient(base_url=_BASE_URL, model=_MODEL)
    metrics_client = VLLMMetricsClient(base_url=_BASE_URL)

    def _hits_delta(fn):
        before = metrics_client.prefix_cache_hits_total()
        result = fn()
        after = metrics_client.prefix_cache_hits_total()
        return result, after - before

    # 0. Absorb the one-time engine warm-up cost on an unrelated prompt
    # before measuring anything real -- see _WARM_UP_PROMPT's own comment.
    completion_client.complete_timed(_WARM_UP_PROMPT, _MAX_TOKENS)

    # 1. Cold: nothing cached yet for this prefix.
    (cold_text, cold_latency), cold_hits_delta = _hits_delta(
        lambda: completion_client.complete_timed(_SHARED_PREFIX + _CONTINUATION_A, _MAX_TOKENS)
    )

    # 2. Same shared prefix, a DIFFERENT continuation -- real prefix reuse
    # (not exact-string caching) is what should make this fast.
    (warm_text, warm_latency), warm_hits_delta = _hits_delta(
        lambda: completion_client.complete_timed(_SHARED_PREFIX + _CONTINUATION_B, _MAX_TOKENS)
    )

    # 3. A genuinely different prefix (diverges in its first sentence) --
    # must NOT be served a stale hit from step 1/2's cached blocks.
    (miss_text, miss_latency), miss_hits_delta = _hits_delta(
        lambda: completion_client.complete_timed(_CHANGED_PREFIX + _CONTINUATION_A, _MAX_TOKENS)
    )

    print(
        f"\ncold: latency={cold_latency:.4f}s hits_delta={cold_hits_delta:.0f} text={cold_text!r}\n"
        f"warm: latency={warm_latency:.4f}s hits_delta={warm_hits_delta:.0f} text={warm_text!r}\n"
        f"miss: latency={miss_latency:.4f}s hits_delta={miss_hits_delta:.0f} text={miss_text!r}"
    )

    # The real claim: reusing a shared prefix across different prompts
    # registers real, server-side cache hits.
    assert warm_hits_delta > 0, "a shared prefix across prompts should register a real cache hit"
    # The real invalidation claim: a genuinely different prefix must not
    # be served a stale hit from the earlier requests' cached blocks.
    assert miss_hits_delta == 0, "a genuinely different prefix must not register a cache hit"
    # The real speedup claim: the warm request (real prefix reuse) is
    # faster than both the cold request and the genuine miss.
    assert warm_latency < cold_latency
    assert warm_latency < miss_latency
