import time

import httpx

from src.cag.domain.ports import CompletionServer


class VLLMCompletionClient(CompletionServer):
    # A real client against vLLM's OpenAI-compatible /v1/completions
    # endpoint -- the boundary a future orchestration layer's CAG tier
    # would call through. temperature=0.0 keeps completions
    # deterministic, which matters for this batch's own tests: a
    # prefix-cache hit must not change what the model outputs, only how
    # fast it arrives.
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        # An injectable httpx.Client (e.g. one built with
        # httpx.MockTransport) is what keeps this class's own unit tests
        # fast and CI-safe -- the real serving-engine behavior this
        # client talks to is only honestly verifiable against a live
        # vLLM server, which is what the integration test does.
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, prompt: str, max_tokens: int) -> str:
        response = self._client.post(
            f"{self._base_url}/v1/completions",
            json={
                "model": self._model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["text"])

    def complete_many_timed(
        self, prompt: str, max_tokens: int, n: int
    ) -> tuple[list[str], float]:
        # Parallel sampling: n completions branching from ONE prompt,
        # which is precisely the workload PagedAttention's block sharing
        # and copy-on-write exist to serve. Deliberately not on the
        # CompletionServer port -- that port models the single-answer
        # boundary an orchestration CAG tier would call, while this is a
        # serving-engine capability being measured, not composed.
        start = time.perf_counter()
        response = self._client.post(
            f"{self._base_url}/v1/completions",
            json={
                "model": self._model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 1.0,
                "n": n,
            },
        )
        response.raise_for_status()
        elapsed = time.perf_counter() - start
        return [str(choice["text"]) for choice in response.json()["choices"]], elapsed

    def complete_timed(self, prompt: str, max_tokens: int) -> tuple[str, float]:
        # Real wall-clock request latency, not a true streamed TTFT --
        # this client uses the non-streaming completions endpoint, so
        # "first token" and "last token" arrive in the same response.
        # Kept as an honest, disclosed proxy for TTFT rather than
        # mislabeled as the real thing: with max_tokens kept small (this
        # batch's tests use 5), decode time is a small, roughly constant
        # addition on top of prefill, so a genuine prefix-cache hit --
        # which skips prefill entirely -- still produces a measurably
        # shorter number here.
        start = time.perf_counter()
        text = self.complete(prompt, max_tokens)
        elapsed = time.perf_counter() - start
        return text, elapsed
