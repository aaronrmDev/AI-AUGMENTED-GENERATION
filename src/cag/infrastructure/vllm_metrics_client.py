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
        #
        # Two assumptions, stated rather than silently relied on. The
        # value is taken as the line's last whitespace-separated field,
        # which is correct for the plain exposition format vLLM emits
        # (verified against a real running server) but would read the
        # wrong field if a line ever carried an optional trailing
        # timestamp or an OpenMetrics exemplar. And an entirely absent
        # metric raises rather than returning 0.0 -- deliberate, because
        # a silent zero would let this batch's own integration test pass
        # vacuously against a server running WITHOUT prefix caching
        # enabled, which is exactly the case that test exists to catch.
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
