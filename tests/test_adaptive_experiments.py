"""## Executive summary (read this first)

Test pre-origin inner forecasts, incomplete target refusal, fallback behaviour,
seed-independent adaptation, historical gaps and immutable benchmark output.
"""

import json

import numpy as np
import pandas as pd
import pytest

from experiments import adaptive as a
from experiments import batch, joint


def history(n=900):
    return np.cumsum(np.random.default_rng(9).normal(size=(n, 2)), axis=0)


def test_inner_forecasts_never_receive_their_own_targets(monkeypatch):
    data = history()
    seen = []
    original = joint.predict

    def observed(prefix, steps, returns, family, monthly, cfg, draws, seed):
        np.testing.assert_array_equal(prefix, data[: len(prefix)])
        assert len(prefix) + int(np.max(steps)) <= len(data)
        seen.append(len(prefix))
        return original(prefix, steps, returns, family, monthly, cfg, draws, seed)

    monkeypatch.setattr(joint, "predict", observed)
    a._choose_cached.cache_clear()
    choice = a.choose_mix(data, np.array([[5, 20], [5, 20]]), False)
    assert len(seen) == 6 * len(a.WEIGHTS)
    assert choice["inner_latest_target_index"] <= len(data) - 1
    assert choice["mix"] in a.WEIGHTS


def test_inner_incomplete_future_is_refused():
    with pytest.raises(ValueError, match="unavailable future"):
        a.inner_outcome(history(100), 90, np.array([[5, 20], [5, 20]]), False)


def test_short_history_falls_back_without_removing_outer_case():
    choice = a.choose_mix(history(320), [5, 126], False)
    assert choice["fallback"] and choice["mix"] == 0 and choice["inner_cases"] == 0
    samples = a.predict(history(320), [5, 126], False, "T2-F3", False, a.configs()[2], 200, 17)
    assert samples.shape == (200, 4) and np.isfinite(samples).all()


def test_outer_future_changes_targets_but_not_adaptation():
    data = history(1500)
    dates = pd.bdate_range("2000-01-03", periods=len(data))
    item = dict(
        frame=pd.DataFrame(data, index=dates), steps=np.array([5, 20]), monthly=False, returns=False
    )
    origin = str(dates[1000].date())
    before, outcome, _, _ = batch.case_at(item, origin)
    changed = item.copy()
    changed["frame"] = item["frame"].copy()
    changed["frame"].iloc[1001:] += 100000
    after, outcome2, _, _ = batch.case_at(changed, origin)
    assert a.choose_mix(before, item["steps"], False) == a.choose_mix(after, item["steps"], False)
    assert np.all(outcome != outcome2)
    np.testing.assert_array_equal(
        a.predict(before, item["steps"], False, "T2-F3", False, a.configs()[2], 200, 17),
        a.predict(after, item["steps"], False, "T2-F3", False, a.configs()[2], 200, 17),
    )


def test_fixed_control_matches_previous_model_and_seed_does_not_select_weight():
    data = history()
    cfg = a.configs()[1]
    np.testing.assert_array_equal(
        a.predict(data, [5, 20], False, "T2-F3", False, cfg, 200, 17),
        joint.predict(data, [5, 20], False, "T2-F3", False, dict(cfg, id="previous"), 200, 17),
    )
    d0 = a.diagnostics(data, [5, 20], False, a.configs()[2])
    a.predict(data, [5, 20], False, "T2-F3", False, a.configs()[2], 200, 41)
    assert a.diagnostics(data, [5, 20], False, a.configs()[2]) == d0


def test_long_inner_gap_is_refused():
    dates = pd.bdate_range("2000-01-03", periods=900).to_numpy()
    dates[500:] += np.timedelta64(365, "D")
    item = dict(frame=pd.DataFrame(history(), index=pd.to_datetime(dates)), steps=np.array([126]))
    with pytest.raises(ValueError, match="Inner training window"):
        a.validate_history(item, str(item["frame"].index[-1].date()))


def test_protocol_is_diagnostic_and_refuses_existing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "inventory", lambda root: ([], [], {}))
    rows = []
    for cfg in a.configs():
        for seed in [0, 17, 41]:
            rows.append(
                dict(
                    config=cfg["id"],
                    cluster="X",
                    group="X",
                    source="synthetic",
                    composite=1,
                    origin="2020-01-02",
                    family="T2-F1",
                    monthly=False,
                    seed=seed,
                    coverage90=0.9,
                    selected_mix=cfg["mix"],
                    inner_cases=6,
                    inner_latest_target_index=899,
                    inner_last_history_index=899,
                    inner_losses="{}",
                    inner_fallback=False,
                    history_sha256="test",
                )
            )
    monkeypatch.setattr(a, "run_cases", lambda *args: (pd.DataFrame(rows), pd.DataFrame(), []))
    out = tmp_path / "run"
    a.main(["--run-dir", str(out)])
    plan = json.loads((out / "plan.json").read_text())
    assert not plan["rankable"] and "No independent holdout" in plan["warning"]
    assert len(pd.read_csv(out / "weight_audit.csv")) == 1
    with pytest.raises(SystemExit):
        a.main(["--run-dir", str(out)])
