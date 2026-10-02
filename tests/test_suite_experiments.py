"""## Executive summary (read this first)

Verify that composite adaptation uses completed history and shared joint/tail
scores, preserves the prior marginal control, and refuses overwriting runs.
"""

import json

import numpy as np
import pandas as pd
import pytest

from experiments import adaptive as a
from experiments import batch, joint
from experiments import suite as s


def item():
    return dict(card={"scoring": {"params": {}}}, assets=["X", "Y"], horizons=[5, 20])


def history():
    return np.cumsum(np.random.default_rng(9).normal(size=(900, 2)), axis=0)


def test_composite_receives_only_completed_prefixes_and_includes_joint_tail(monkeypatch):
    values = history()
    received = []

    def fake_predict(prefix, grid, returns, family, monthly, cfg, draws, seed):
        np.testing.assert_array_equal(prefix, values[: len(prefix)])
        assert len(prefix) + int(np.max(grid)) <= len(values)
        received.append(len(prefix))
        return np.full((draws, 4), cfg["mix"])

    monkeypatch.setattr(joint, "predict", fake_predict)
    monkeypatch.setattr(
        batch,
        "raw_score",
        lambda *args: dict(marginal=1, joint=1, tail=1, weights_effective=(0.5, 0.3, 0.2)),
    )
    # Marginal loss is tied; only the joint/tail objective favours weight .5.
    monkeypatch.setattr(
        batch,
        "relative_score",
        lambda samples, *args: dict(
            composite=0.5 + 0.3 * (1 - samples[0, 0]) + 0.2 * (1 - samples[0, 0])
        ),
    )
    s._composite_cached.cache_clear()
    choice = s.choose_composite(values, [5, 20], False, item())
    assert choice["mix"] == 0.5 and not choice["fallback"]
    assert choice["inner_latest_target_index"] <= len(values) - 1
    assert len(received) == 6 * (1 + len(a.WEIGHTS))


def test_outer_future_mutation_does_not_change_composite_choice():
    values = history()
    dates = pd.bdate_range("2000-01-03", periods=len(values))
    source = dict(
        frame=pd.DataFrame(values, index=dates),
        steps=np.array([5, 20]),
        monthly=False,
        returns=False,
    )
    before, outcome, _, _ = batch.case_at(source, str(dates[700].date()))
    source["frame"].iloc[701:] += 10000
    after, outcome2, _, _ = batch.case_at(source, str(dates[700].date()))
    assert np.any(outcome != outcome2)
    assert s.choose_composite(before, [5, 20], False, item()) == s.choose_composite(
        after, [5, 20], False, item()
    )


def test_old_marginal_selector_paths_preserved():
    data = history()
    cfg = next(c for c in s.configs() if c["id"] == s.MARGINAL)
    np.testing.assert_array_equal(
        s.predictor(item())(data, [5, 20], False, "T2-F1", False, cfg, 200, 17),
        a.predict(data, [5, 20], False, "T2-F1", False, a.configs()[2], 200, 17),
    )


def test_short_composite_history_fallback_keeps_case():
    choice = s.choose_composite(history()[:320], [5, 126], False, item())
    assert choice["fallback"] and choice["mix"] == 0


def test_shortlist_keeps_all_controls_and_top_three():
    frame = pd.DataFrame(
        [
            dict(config=c["id"], cluster="X", composite=1 - index * 0.01)
            for index, c in enumerate(s.configs())
        ]
    )
    frozen = s.shortlist(frame)
    assert {s.BASE, s.MARGINAL, s.COMPOSITE, s.CALIBRATED}.issubset(frozen)
    assert len(frozen) <= 7
    assert len(s.configs()) == 21


def test_validation_never_claims_promotion_and_repeat_is_refused(tmp_path):
    frame = pd.DataFrame(
        [dict(config=s.BASE, cluster="X", group="X", origin="2024-01-02", composite=1)]
    )
    assert not s.validation_report(frame)["promotion"]
    out = tmp_path / "run"
    out.mkdir()
    (out / "plan.json").write_text("{}")
    with pytest.raises(SystemExit):
        s.main(["--run-dir", str(out)])


def test_shortlist_is_written_before_validation_targets_are_read(tmp_path, monkeypatch):
    out = tmp_path / "suite"
    monkeypatch.setattr(batch, "inventory", lambda root: ([], [], {}))
    calls = []

    def fake_run(items, origins, bounds, cfgs, seeds, draws):
        calls.append(list(origins))
        if list(origins) == list(s.VALIDATION):
            frozen = json.loads((out / "shortlist.json").read_text())
            assert (out / "validation_started.json").exists()
            assert {c["id"] for c in cfgs} == set(frozen["configs"])
        rows = []
        for cfg in cfgs:
            rows.append(
                dict(
                    config=cfg["id"],
                    group="X",
                    cluster="X",
                    origin=origins[0],
                    seed=0,
                    family="T2-F1",
                    monthly=False,
                    coverage90=0.9,
                    composite=1 if cfg["id"] == s.BASE else 0.99,
                    selected_mix=cfg["mix"],
                    inner_cases=0,
                    inner_latest_target_index=None,
                    inner_last_history_index=400,
                    inner_fallback=False,
                    inner_losses="",
                    history_sha256="test",
                )
            )
        return pd.DataFrame(rows), pd.DataFrame(), []

    monkeypatch.setattr(s, "run_cases", fake_run)
    s.main(["--run-dir", str(out)])
    assert calls == [list(a.ORIGINS), list(s.VALIDATION)]
    assert not json.loads((out / "decision.json").read_text())["promotion"]
