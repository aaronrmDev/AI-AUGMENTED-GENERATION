import httpx
import pytest

from src.cag.infrastructure.vllm_metrics_client import VLLMMetricsClient

_SAMPLE_METRICS = """\
# HELP vllm:prefix_cache_queries_total Prefix cache queries.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 42.0
# HELP vllm:prefix_cache_hits_total Prefix cache hits.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 17.0
# HELP vllm:unrelated_metric Something else entirely.
# TYPE vllm:unrelated_metric gauge
vllm:unrelated_metric{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 999.0
"""


def _client_returning(text: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/metrics"
        return httpx.Response(200, text=text)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_reads_prefix_cache_queries_total():
    client = VLLMMetricsClient(
        base_url="http://localhost:8001", client=_client_returning(_SAMPLE_METRICS)
    )
    assert client.prefix_cache_queries_total() == pytest.approx(42.0)


def test_reads_prefix_cache_hits_total():
    client = VLLMMetricsClient(
        base_url="http://localhost:8001", client=_client_returning(_SAMPLE_METRICS)
    )
    assert client.prefix_cache_hits_total() == pytest.approx(17.0)


def test_sums_multiple_labeled_series_for_the_same_metric():
    text = (
        'vllm:prefix_cache_hits_total{model_name="a"} 3.0\n'
        'vllm:prefix_cache_hits_total{model_name="b"} 4.0\n'
    )
    client = VLLMMetricsClient(base_url="http://localhost:8001", client=_client_returning(text))
    assert client.prefix_cache_hits_total() == pytest.approx(7.0)


def test_matches_metric_name_without_a_total_suffix_too():
    # Some exposition formats declare the metric without "_total" but
    # still add the suffix in the actual series line -- and some don't.
    # Matching both spellings keeps this correct either way.
    text = 'vllm:prefix_cache_hits{model_name="a"} 5.0\n'
    client = VLLMMetricsClient(base_url="http://localhost:8001", client=_client_returning(text))
    assert client.prefix_cache_hits_total() == pytest.approx(5.0)


def test_raises_when_the_metric_is_missing():
    client = VLLMMetricsClient(base_url="http://localhost:8001", client=_client_returning(""))
    with pytest.raises(ValueError):
        client.prefix_cache_hits_total()
