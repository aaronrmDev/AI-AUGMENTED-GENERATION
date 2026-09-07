"""Live verification that PagedAttention's block table really does let
parallel sequences from one prompt share physical memory -- CAG.md's
"memory sharing across parallel sequences that start from the same prompt"
and "efficient beam search" claims, made falsifiable against a real vLLM
server rather than restated.

Requires a real vLLM server running WITH --enable-prefix-caching (see
docs/superpowers/specs/2026-09-07-cag-paged-attention-design.md). Like the
prefix-caching batch's own test, this cannot run in CI -- GitHub Actions has
no ROCm GPU -- so it skips itself with a stated reason, and runs from inside
WSL2 because this machine's mirrored WSL networking does not expose the
server's port to the Windows host.

Why prefix caching must be ON for a test about PagedAttention, which is a
finding rather than a setup detail: measured against this same server with
prefix caching OFF, an n=8 request took 4.556s against n=1's 0.091s and
shared nothing at all. PagedAttention's block table is present either way --
it is not optional in vLLM -- so the sharing CAG.md attributes to
PagedAttention is in practice delivered by the prefix-cache layer built on
top of that block table, not by the block table alone. The full comparison
is in evaluation/reports/cag-paged-attention.md.
"""
import os

import httpx
import pytest

from src.cag.infrastructure.vllm_completion_client import VLLMCompletionClient
from src.cag.infrastructure.vllm_metrics_client import VLLMMetricsClient

_BASE_URL = os.environ.get("CAG_VLLM_BASE_URL", "http://192.168.0.243:8001")
_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_MAX_TOKENS = 20
_PARALLEL_SAMPLES = 8

# Long enough to span many whole cache blocks, so block-granular sharing
# is the dominant effect rather than a rounding artefact.
_PROMPT = (
    "The following is a detailed technical discussion of distributed systems, "
    "consensus protocols, and fault tolerance in large scale infrastructure. "
) * 12


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


def test_parallel_samples_share_one_physical_copy_of_their_prompt():
    completion_client = VLLMCompletionClient(base_url=_BASE_URL, model=_MODEL)
    metrics_client = VLLMMetricsClient(base_url=_BASE_URL)

    # Absorb one-time engine warm-up, the confound the prefix-caching
    # batch measured and corrected for.
    completion_client.complete_timed("warm up the engine please", 5)

    def _measure(n: int) -> tuple[float, float, float]:
        prompt_before = metrics_client.prompt_tokens_total()
        hits_before = metrics_client.prefix_cache_hits_total()
        texts, elapsed = completion_client.complete_many_timed(_PROMPT, _MAX_TOKENS, n)
        assert len(texts) == n
        prompt_delta = metrics_client.prompt_tokens_total() - prompt_before
        hits_delta = metrics_client.prefix_cache_hits_total() - hits_before
        return prompt_delta, hits_delta, elapsed

    single_prompt_tokens, _single_hits, single_latency = _measure(1)
    many_prompt_tokens, many_hits, many_latency = _measure(_PARALLEL_SAMPLES)

    print(
        f"\nn=1: prompt_tokens={single_prompt_tokens:.0f} latency={single_latency:.4f}s\n"
        f"n={_PARALLEL_SAMPLES}: prompt_tokens={many_prompt_tokens:.0f} "
        f"cache_hit_tokens={many_hits:.0f} latency={many_latency:.4f}s"
    )

    # vllm:prompt_tokens_total is a LOGICAL counter -- it bills every
    # branch for the full prompt, so it reads n x the prompt whether or
    # not anything was shared, and proves nothing on its own. Asserted
    # here so a future reader doesn't mistake it for evidence of sharing.
    assert many_prompt_tokens == pytest.approx(
        single_prompt_tokens * _PARALLEL_SAMPLES, rel=0.05
    )

    # The real evidence: nearly all of those logically-billed prompt
    # tokens were served from shared blocks rather than prefilled again.
    assert many_hits > 0.9 * many_prompt_tokens

    # And the consequence that matters: n parallel samples cost far less
    # than n independent prefills would.
    assert many_latency < single_latency * _PARALLEL_SAMPLES


def test_shared_prompt_tokens_land_on_whole_cache_block_boundaries():
    # Sharing is block-granular, not token-granular: whatever tail of a
    # sequence doesn't fill a complete block cannot be shared and is
    # genuinely prefilled per branch. Confirming the shared total is a
    # clean multiple of the block size is what distinguishes real
    # block-table sharing from a coincidentally large number.
    completion_client = VLLMCompletionClient(base_url=_BASE_URL, model=_MODEL)
    metrics_client = VLLMMetricsClient(base_url=_BASE_URL)
    completion_client.complete_timed("warm up the engine please", 5)

    unique_prompt = _PROMPT + " Distinguishing suffix for block boundary check."
    completion_client.complete_many_timed(unique_prompt, _MAX_TOKENS, 1)

    hits_before = metrics_client.prefix_cache_hits_total()
    completion_client.complete_many_timed(unique_prompt, _MAX_TOKENS, _PARALLEL_SAMPLES)
    shared = metrics_client.prefix_cache_hits_total() - hits_before

    print(f"\nshared tokens across {_PARALLEL_SAMPLES} branches: {shared:.0f}")
    assert shared > 0
    # vLLM's default block size is 16 tokens.
    assert shared % 16 == 0, f"{shared} shared tokens is not a whole number of 16-token blocks"
