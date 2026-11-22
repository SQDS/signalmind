"""Bounded recovery from the observed TLS reset; no requests or credentials used."""

import io
import ssl
import urllib.error

import pytest

from eval.unified import drivers


def test_connection_reset_retries_twice_then_returns_response(monkeypatch):
    calls = []

    def open_url(request, **kw):
        calls.append(request.data)
        if len(calls) < 3:
            raise urllib.error.URLError(ConnectionResetError(104, "Connection reset by peer"))
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(drivers.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(drivers.time, "sleep", lambda _: None)
    driver = drivers.OpenAIChatDriver({"model": "test"}, transport=lambda body: {})
    assert driver._post({"input": "test"}) == {"ok": True}
    assert len(calls) == 3 and len(set(calls)) == 1


@pytest.mark.parametrize("certificate_error", [False, True])
def test_retries_are_bounded_and_certificate_errors_are_not_retried(monkeypatch, certificate_error):
    count = 0

    def open_url(*args, **kw):
        nonlocal count
        count += 1
        cause = (
            ssl.SSLCertVerificationError("invalid certificate")
            if certificate_error
            else ConnectionResetError(104, "reset")
        )
        raise urllib.error.URLError(cause)

    monkeypatch.setattr(drivers.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(drivers.time, "sleep", lambda _: None)
    driver = drivers.OpenAIChatDriver({"model": "test"}, transport=lambda body: {})
    with pytest.raises(urllib.error.URLError):
        driver._post({"input": "test"})
    assert count == (1 if certificate_error else 3)
