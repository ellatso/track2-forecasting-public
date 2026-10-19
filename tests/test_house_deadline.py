"""## Executive summary (read this first)

Exercise real Linux elapsed-time limits without calling any model endpoint.
An inactivity timeout cannot stop a slow stream; this deadline can. Exhausted
budgets never start an HTTP request or expose injected credentials.
"""

import signal
import time

import pytest

from qfbench2_track_forecasting.house import House, HouseDeadline, request_deadline


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="Linux submission deadline")
def test_hard_deadline_interrupts_continuous_progress_and_restores_handler():
    previous = signal.getsignal(signal.SIGALRM)
    start = time.monotonic()
    with pytest.raises(HouseDeadline), request_deadline(0.03):
        while True:
            # A stream producing bytes regularly would never hit its inactivity timeout.
            time.sleep(0.001)
    assert time.monotonic() - start < 2
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_expired_total_budget_never_starts_network(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "http://house.invalid:9000")
    monkeypatch.setenv("MODEL_TOKEN", "synthetic-secret")
    monkeypatch.setenv("MODEL_NAME", "synthetic")
    monkeypatch.setenv("http_proxy", "http://synthetic:password@127.0.0.1:1")
    house = House()
    house.deadline = time.monotonic() - 1
    assert house.ask("test") is None
    assert house.requests == 0
    assert house.failures == ["local time budget exhausted"]


@pytest.mark.parametrize(
    "primary,secondary,wanted",
    [
        ("/injected/primary.pem", "/injected/secondary.pem", "/injected/primary.pem"),
        ("", "/injected/secondary.pem", "/injected/secondary.pem"),
        ("", "", True),
    ],
)
def test_explicit_proxy_keeps_tls_verification_and_injected_ca(
    monkeypatch, primary, secondary, wanted
):
    import requests

    monkeypatch.setenv("MODEL_ENDPOINT", "https://house.invalid:9000")
    monkeypatch.setenv("MODEL_TOKEN", "synthetic-secret")
    monkeypatch.setenv("MODEL_NAME", "synthetic")
    monkeypatch.setenv("http_proxy", "http://synthetic:password@127.0.0.1:1")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", primary)
    monkeypatch.setenv("CURL_CA_BUNDLE", secondary)
    seen = []

    def post(self, *args, **kwargs):
        seen.append((self.trust_env, kwargs.get("verify"), kwargs.get("proxies")))
        raise requests.exceptions.SSLError("synthetic failure with no credential disclosure")

    monkeypatch.setattr(requests.Session, "post", post)
    house = House()
    assert house.ask("test") is None
    assert seen == [
        (
            False,
            wanted,
            dict(
                http="http://synthetic:password@127.0.0.1:1",
                https="http://synthetic:password@127.0.0.1:1",
            ),
        )
    ]
    assert house.failures == ["House response unavailable or invalid"]


def test_unreadable_injected_ca_is_a_numeric_fallback(monkeypatch):
    import requests

    for key, value in dict(
        MODEL_ENDPOINT="https://house.invalid",
        MODEL_TOKEN="synthetic",
        MODEL_NAME="synthetic",
        http_proxy="http://a:b@127.0.0.1:1",
        REQUESTS_CA_BUNDLE="/missing/ca.pem",
    ).items():
        monkeypatch.setenv(key, value)

    def post(*args, **kwargs):
        raise OSError("synthetic unreadable CA path")

    monkeypatch.setattr(requests.Session, "post", post)
    house = House()
    assert house.ask("test") is None
    assert house.failures == ["House response unavailable or invalid"]
