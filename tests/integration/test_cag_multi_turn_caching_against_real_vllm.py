"""Live verification of the headline claim CAG.md makes for Multi-Turn
Caching: "near-O(1) amortized cost per turn instead of cost that grows with
every message."

Drives a real growing conversation against a real vLLM server, sending the
full accumulated history each turn as any chat application does, and asks the
structural question rather than the comfortable one. A cheaper total would
prove little -- a strategy can halve a quadratic and still be quadratic. What
distinguishes O(1) per turn is that NEWLY COMPUTED work stops growing while
the prompt keeps growing, so both are measured per turn and compared as
growth rates rather than totals.

Requires a real vLLM server with --enable-prefix-caching. Cannot run in CI
(no ROCm GPU on GitHub Actions), so it skips itself with a stated reason, and
runs from inside WSL2 because this machine's mirrored WSL networking does not
expose the server's port to the Windows host.
"""
import os
import uuid

import httpx
import pytest

from src.cag.infrastructure.vllm_completion_client import VLLMCompletionClient
from src.cag.infrastructure.vllm_metrics_client import VLLMMetricsClient

_BASE_URL = os.environ.get("CAG_VLLM_BASE_URL", "http://192.168.0.243:8001")
_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_TURNS = 8
_MAX_TOKENS = 30


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


def test_per_turn_prefill_stays_flat_while_the_conversation_grows():
    completion_client = VLLMCompletionClient(base_url=_BASE_URL, model=_MODEL)
    metrics_client = VLLMMetricsClient(base_url=_BASE_URL)
    completion_client.complete_timed("warm up the engine please", 5)

    # Salted so a previous run's identical conversation cannot pre-warm
    # this one's cache -- the same contamination that understated the
    # Cache-Aware Batching batch's result until it was caught.
    history = (
        f"Session {uuid.uuid4().hex}. You are a helpful assistant "
        "discussing distributed systems in depth. "
    )

    prompt_tokens: list[float] = []
    newly_computed: list[float] = []
    cached: list[float] = []

    for turn in range(1, _TURNS + 1):
        prompt = history + (
            f"Turn {turn}: explain one more distinct aspect of consensus "
            "protocols in detail. "
        )
        before_prompt = metrics_client.prompt_tokens_total()
        before_hits = metrics_client.prefix_cache_hits_total()
        text, _latency = completion_client.complete_timed(prompt, _MAX_TOKENS)
        turn_prompt = metrics_client.prompt_tokens_total() - before_prompt
        turn_cached = metrics_client.prefix_cache_hits_total() - before_hits

        prompt_tokens.append(turn_prompt)
        cached.append(turn_cached)
        newly_computed.append(turn_prompt - turn_cached)
        history = prompt + text + " "

    print(
        "\nturn  prompt_tokens  cached  newly_computed\n"
        + "\n".join(
            f"{i + 1:>4}{prompt_tokens[i]:>15.0f}{cached[i]:>8.0f}{newly_computed[i]:>16.0f}"
            for i in range(_TURNS)
        )
    )

    # The prompt really did grow substantially, or the test proves nothing.
    prompt_growth = prompt_tokens[-1] / prompt_tokens[0]
    assert prompt_growth > 4.0

    # Turn 1 is excluded: nothing is cached yet, so it is the cold start
    # rather than a steady-state turn, and including it would compare a
    # cache miss against cache hits.
    steady_state = newly_computed[1:]
    work_growth = max(steady_state) / min(steady_state)

    # The structural claim: newly computed work stops tracking prompt
    # length. Measured 17-31 tokens across an 8-turn conversation whose
    # prompt grew 6.5x -- a spread of one 16-token cache block, which is
    # where the block boundary happens to fall rather than growth.
    assert work_growth < prompt_growth / 2

    # And history really is being reused, not merely recomputed cheaply.
    assert cached[-1] > cached[1]
    assert cached[-1] > 0.8 * prompt_tokens[-1]
