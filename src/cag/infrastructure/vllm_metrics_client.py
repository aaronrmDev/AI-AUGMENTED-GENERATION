import httpx


class VLLMMetricsClient:
    # Reads vLLM's own Prometheus-format /metrics endpoint. Real,
    # server-side ground truth for whether a prefix-cache hit actually
    # happened -- the completions response itself carries no such
    # signal, so this is the only honest way to confirm cache behavior
    # rather than inferring it purely from latency.
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=10.0)

    def _sum_metric(self, metric_name: str) -> float:
        # Prometheus counters expose a trailing "_total" in the actual
        # exposition text even when declared without it; matching both
        # spellings, and summing every labeled series (e.g. per model),
        # keeps this correct regardless of which the running vLLM
        # version emits.
        response = self._client.get(f"{self._base_url}/metrics")
        response.raise_for_status()
        candidates = (metric_name, f"{metric_name}_total")
        total = 0.0
        found = False
        for line in response.text.splitlines():
            if line.startswith("#"):
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name in candidates:
                total += float(line.rsplit(" ", 1)[-1])
                found = True
        if not found:
            raise ValueError(f"metric {metric_name!r} not found in /metrics output")
        return total

    def prefix_cache_queries_total(self) -> float:
        return self._sum_metric("vllm:prefix_cache_queries")

    def prefix_cache_hits_total(self) -> float:
        return self._sum_metric("vllm:prefix_cache_hits")
