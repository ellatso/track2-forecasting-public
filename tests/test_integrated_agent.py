"""## Executive summary (read this first)

Check cutoff isolation, evidence validation, shared scenarios, failure recovery, and the
real HTTP/proxy request shape without calling a live model or using market outcomes.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import numpy as np
import pandas as pd
import pytest

from qfbench2_track_forecasting import agent, cli
from qfbench2_track_forecasting.house import House

QUOTE = "The committee announced a reduction in the policy rate today."


@pytest.fixture
def unit(tmp_path):
    root = tmp_path / "unit"
    (root / "text").mkdir(parents=True)
    dates = pd.bdate_range("2020-01-01", periods=90)
    frame = pd.DataFrame(
        {"date": dates, "asset": "A", "value": 100 + np.cumsum(np.tile([0.2, -0.1, 0.1], 30))}
    )
    frame.to_parquet(root / "panel.parquet", index=False)
    asof = str(dates[-1].date())
    (root / "card.toml").write_text(
        '[task]\nid="synthetic"\n[metadata]\ncategory="T2-F4"\n[targets]\nasset_ids=["A"]\nhorizons=[21,42]\ntarget_type="level"\n'
    )
    (root / "text" / "doc.txt").write_text(QUOTE)
    (root / "text" / "corpus_index.json").write_text(
        json.dumps(
            {
                "documents": [
                    {"doc_id": "doc", "timestamp": asof, "file": "doc.txt", "doc_type": "statement"}
                ]
            }
        )
    )
    return root, frame, asof


def evidence_reply():
    return {
        "evidence": [
            {
                "doc_id": "doc",
                "quote": QUOTE,
                "assets": ["A"],
                "status": "inferred",
                "interpretation": "Potential lower future level",
            }
        ]
    }


def scenario_reply():
    return {
        "scenarios": [
            {
                "label": "policy",
                "weight": 1,
                "assets": {
                    "A": {
                        "shift_sd": -0.5,
                        "vol_scale": 1.2,
                        "tail_weight": 0.1,
                        "confidence": 0.5,
                        "evidence_ids": ["e0"],
                    }
                },
            }
        ]
    }


def test_corpus_drops_future_missing_dates_and_parent_paths(unit):
    root, _, asof = unit
    index = root / "text" / "corpus_index.json"
    parsed = json.loads(index.read_text())
    parsed["documents"] += [
        {"doc_id": "future", "timestamp": "2099-01-01", "file": "doc.txt"},
        {"doc_id": "undated", "file": "doc.txt"},
        {"doc_id": "parent", "timestamp": asof, "file": "../card.toml"},
    ]
    index.write_text(json.dumps(parsed))
    docs, excluded = agent.read_documents(root / "text", asof)
    assert [d["doc_id"] for d in docs] == ["doc"]
    assert excluded == 3


def test_fabricated_and_malformed_quotes_never_become_evidence(unit):
    root, _, asof = unit
    docs, _ = agent.read_documents(root / "text", asof)
    reply = evidence_reply()
    reply["evidence"] += [
        {
            "doc_id": "doc",
            "quote": "The future price is already guaranteed to rise.",
            "assets": ["A"],
        },
        {"doc_id": ["doc"], "quote": QUOTE, "assets": ["A"]},
        {"doc_id": "doc", "quote": QUOTE, "assets": ["UNKNOWN"]},
    ]
    valid = agent.validate_evidence(reply, docs, ["A"])
    assert len(valid) == 1 and valid[0]["id"] == "e0"


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True, "1", 10**500])
def test_invalid_model_numbers_cannot_adjust_a_forecast(invalid, unit):
    root, _, asof = unit
    docs, _ = agent.read_documents(root / "text", asof)
    ev = agent.validate_evidence(evidence_reply(), docs, ["A"])
    reply = scenario_reply()
    reply["scenarios"][0]["assets"]["A"]["shift_sd"] = invalid
    assert agent.validate_scenarios(reply, ev, ["A"], "T2-F4") == []


def test_inferred_tone_cannot_narrow_and_unknown_citations_are_ignored(unit):
    root, _, asof = unit
    docs, _ = agent.read_documents(root / "text", asof)
    ev = agent.validate_evidence(evidence_reply(), docs, ["A"])
    reply = scenario_reply()
    reply["scenarios"][0]["assets"]["A"].update(vol_scale=0.1, confidence=1, shift_sd=100)
    scenarios = agent.validate_scenarios(reply, ev, ["A"], "T2-F4")
    assert scenarios[0]["assets"]["A"]["vol_scale"] >= 1
    assert scenarios[0]["assets"]["A"]["shift_sd"] <= 0.75
    reply["scenarios"][0]["assets"]["A"]["evidence_ids"] = ["invented"]
    assert agent.validate_scenarios(reply, ev, ["A"], "T2-F4") == []


def test_joint_scenario_uses_monthly_steps_keeps_baseline_half_and_unsupported_assets():
    base = np.full((1000, 2, 2), 100.0)
    stats = {
        "step_unit": "month",
        "panel_steps": {a: {"140": 3, "160": 4} for a in ["A", "B"]},
        "step_sd": {"A": 2.0, "B": 2.0},
        "step_drift": {"A": 0.0, "B": 0.0},
        "last": {"A": 100.0, "B": 100.0},
    }
    scenarios = [
        {"weight": 1.0, "assets": {"A": {"shift_sd": 0.5, "vol_scale": 1.0, "tail_weight": 0.0}}}
    ]
    adjusted = agent.apply_scenarios(base, ["A", "B"], [140, 160], stats, scenarios, 17)
    np.testing.assert_array_equal(adjusted[:, 1, :], base[:, 1, :])
    active = adjusted[:, 0, 1] != 100
    assert 400 < active.sum() < 600
    np.testing.assert_allclose(
        (adjusted[active, 0, 0] - 100) / (adjusted[active, 0, 1] - 100), 0.75
    )
    np.testing.assert_array_equal(adjusted[~active], base[~active])


@pytest.mark.parametrize("model_valid", [False, True])
def test_end_to_end_outputs_supported_adjustment_or_exact_numeric_fallback(
    unit, tmp_path, monkeypatch, model_valid
):
    root, frame, asof = unit

    class FakeHouse:
        def __init__(self):
            self.requests, self.failures = 0, []

        def ask(self, prompt):
            self.requests += 1
            if not model_valid:
                self.failures.append("synthetic failure")
                return None
            return evidence_reply() if self.requests == 1 else scenario_reply()

    monkeypatch.setattr(agent, "House", FakeHouse)
    out = tmp_path / "out" / "forecast.parquet"
    agent.main(
        [
            "--panels",
            str(root),
            "--text",
            str(root / "text"),
            "--asof",
            asof,
            "--out",
            str(out),
            "--n-draws",
            "500",
        ]
    )
    actual = pd.read_parquet(out).value.to_numpy().reshape(500, 1, 2)
    base, _ = cli._draw({"panel": frame}, ["A"], [21, 42], asof, 500, 17)
    meta = json.loads((out.parent / "forecast_meta.json").read_text())
    assert meta["reasoning_applied"] is model_valid
    if model_valid:
        assert np.any(actual != base)
        assert meta["house_requests_attempted"] == 2
    else:
        np.testing.assert_array_equal(actual, base)
        assert meta["house_requests_attempted"] <= 3
    assert np.isfinite(actual).all()
    assert (out.parent / "forecast_rationale.md").stat().st_size > 100


def test_house_request_uses_injected_proxy_absolute_v1_bearer_and_no_redirect(monkeypatch):
    seen = []

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, dict(self.headers), body))
            raw = json.dumps(
                {"choices": [{"finish_reason": "stop", "message": {"content": '{"evidence":[]}'}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("MODEL_ENDPOINT", "http://house.invalid:9000")
        monkeypatch.setenv("MODEL_NAME", "house")
        monkeypatch.setenv("MODEL_TOKEN", "synthetic-bearer")
        monkeypatch.setenv(
            "http_proxy", f"http://synthetic:password@127.0.0.1:{server.server_port}"
        )
        monkeypatch.setenv("NO_PROXY", "*")
        house = House()
        assert house.ask("synthetic request") == {"evidence": []}
        assert seen[0][0] == "http://house.invalid:9000/v1/chat/completions"
        assert seen[0][1]["Authorization"] == "Bearer synthetic-bearer"
        assert seen[0][1]["Proxy-Authorization"].startswith("Basic ")
        assert seen[0][2]["model"] == "house"
        assert seen[0][2]["max_tokens"] <= 4000
        assert not seen[0][2]["chat_template_kwargs"]["enable_thinking"]
        assert "password" not in json.dumps(house.failures)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_missing_house_configuration_makes_no_request(monkeypatch):
    monkeypatch.delenv("MODEL_ENDPOINT", raising=False)
    house = House()
    assert house.ask("test") is None and house.requests == 0
