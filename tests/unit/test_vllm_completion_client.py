import httpx
import pytest

from src.cag.infrastructure.vllm_completion_client import VLLMCompletionClient


def _client_with(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_complete_returns_the_first_choices_text():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/completions"
        body = request.read()
        assert b'"prompt": "hello"' in body or b'"prompt":"hello"' in body
        return httpx.Response(200, json={"choices": [{"text": " world"}]})

    client = VLLMCompletionClient(
        base_url="http://localhost:8001", model="test-model", client=_client_with(handler)
    )
    assert client.complete("hello", max_tokens=5) == " world"


def test_complete_sends_deterministic_temperature_and_the_configured_model():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.read()))
        return httpx.Response(200, json={"choices": [{"text": "x"}]})

    client = VLLMCompletionClient(
        base_url="http://localhost:8001", model="test-model", client=_client_with(handler)
    )
    client.complete("hi", max_tokens=3)
    assert captured["model"] == "test-model"
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 3


def test_complete_raises_on_a_server_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = VLLMCompletionClient(
        base_url="http://localhost:8001", model="test-model", client=_client_with(handler)
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.complete("hi", max_tokens=3)


def test_complete_timed_returns_the_text_and_a_non_negative_elapsed_time():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"text": "y"}]})

    client = VLLMCompletionClient(
        base_url="http://localhost:8001", model="test-model", client=_client_with(handler)
    )
    text, elapsed = client.complete_timed("hi", max_tokens=3)
    assert text == "y"
    assert elapsed >= 0.0


def test_base_url_trailing_slash_is_stripped():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/completions"
        return httpx.Response(200, json={"choices": [{"text": "z"}]})

    client = VLLMCompletionClient(
        base_url="http://localhost:8001/", model="test-model", client=_client_with(handler)
    )
    assert client.complete("hi", max_tokens=1) == "z"
