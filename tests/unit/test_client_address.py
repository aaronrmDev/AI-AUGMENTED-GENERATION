from starlette.requests import Request

from src.api.client_address import real_client_ip, trusted_proxy_count


def _request(*, client_host: str = "10.0.0.1", forwarded_for: str | None = None) -> Request:
    headers = []
    if forwarded_for is not None:
        headers.append((b"x-forwarded-for", forwarded_for.encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
        "client": (client_host, 12345),
    }
    return Request(scope)


def test_with_no_trusted_proxies_the_header_is_never_read(monkeypatch):
    monkeypatch.delenv("TRUSTED_PROXY_COUNT", raising=False)
    request = _request(client_host="10.0.0.1", forwarded_for="203.0.113.7")
    assert real_client_ip(request) == "10.0.0.1"


def test_with_one_trusted_proxy_the_rightmost_forwarded_entry_is_used(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_COUNT", "1")
    # client, then the one trusted proxy's own upstream address.
    request = _request(client_host="10.0.0.1", forwarded_for="203.0.113.7")
    assert real_client_ip(request) == "203.0.113.7"


def test_with_two_trusted_proxies_the_second_from_the_right_is_used(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_COUNT", "2")
    request = _request(client_host="10.0.0.1", forwarded_for="203.0.113.7, 198.51.100.9")
    assert real_client_ip(request) == "203.0.113.7"


def test_fewer_forwarded_entries_than_trusted_proxies_falls_back_to_client_host(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_COUNT", "3")
    request = _request(client_host="10.0.0.1", forwarded_for="203.0.113.7")
    assert real_client_ip(request) == "10.0.0.1"


def test_a_missing_header_falls_back_to_client_host_even_when_proxies_are_trusted(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_COUNT", "1")
    request = _request(client_host="10.0.0.1", forwarded_for=None)
    assert real_client_ip(request) == "10.0.0.1"


def test_no_client_at_all_returns_unknown(monkeypatch):
    monkeypatch.delenv("TRUSTED_PROXY_COUNT", raising=False)
    scope = {"type": "http", "method": "GET", "path": "/", "headers": [], "client": None}
    assert real_client_ip(Request(scope)) == "unknown"


def test_a_malformed_trusted_proxy_count_falls_back_to_zero_instead_of_raising(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_COUNT", "abc")
    assert trusted_proxy_count() == 0


def test_multiple_raw_forwarded_for_header_lines_are_all_honored(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_COUNT", "2")
    # Two raw header lines rather than one comma-joined line -- .get() would
    # only see the first ("203.0.113.7") and never reach the second hop.
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [
            (b"x-forwarded-for", b"203.0.113.7"),
            (b"x-forwarded-for", b"198.51.100.9"),
        ],
        "client": ("10.0.0.1", 12345),
    }
    assert real_client_ip(Request(scope)) == "203.0.113.7"
