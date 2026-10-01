"""## Executive summary (read this first)

Verify isolated daily comparisons, monthly grids, per-cell diagnostics and the
locked two-frequency validation protocol using synthetic histories only.
"""

import json

import numpy as np
import pandas as pd
import pytest

from experiments import batch
from experiments import frequency as f


def test_daily_changes_only_volatility():
    a, b = f.configs(False)
    assert {k for k in a if a[k] != b[k]} == {"id", "volatility"}
    assert a["drift"] == b["drift"] == 1


@pytest.mark.parametrize("model", f.configs(True))
def test_monthly_reproducible_asset_specific_grid(model):
    history = np.cumsum(np.random.default_rng(31).normal(size=(130, 2)), axis=0)
    steps = np.array([[1, 3], [2, 5]])
    a = f.predict(history, steps, False, "T2-F1", True, model, 200, 17)
    b = f.predict(history.copy(), steps, False, "T2-F1", True, model, 200, 17)
    assert a.shape == (200, 4) and np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)


def test_walk_baseline_unchanged():
    history = np.cumsum(np.random.default_rng(3).normal(size=(400, 2)), axis=0)
    cfg = f.configs(False)[0]
    np.testing.assert_array_equal(
        f.predict(history, [5, 20], False, "T2-F3", False, cfg, 200, 17),
        batch.predict(history, [5, 20], False, "T2-F3", False, cfg, 200, 17),
    )


def test_monthly_models_refuse_return_targets():
    with pytest.raises(ValueError, match="level targets"):
        f.predict(np.ones((80, 1)), [1, 2], True, "T2-F1", True, f.configs(True)[1], 200, 17)


def test_cell_diagnostics_and_boundary_refusal():
    dates = pd.date_range("2000-01-01", "2020-01-01", freq="MS")
    item = dict(
        source="synthetic",
        group="X",
        cluster='["X"]',
        assets=["X"],
        horizons=[21, 42],
        steps=np.array([[1, 2]]),
        returns=False,
        monthly=True,
        family="T2-F1",
        card={"scoring": {"params": {}}},
        frame=pd.DataFrame({"X": np.sin(np.arange(len(dates)) / 12)}, index=dates),
    )
    frame, cells, skipped = f.run_cases(
        [item], ["2015-01-03"], f.configs(True), [17], 200, "select"
    )
    assert not skipped and len(frame) == 3 and len(cells) == 6
    assert set(cells.observation_steps) == {1, 2}
    assert set(cells.horizon) == {21, 42}
    base = frame[frame.config == f.BASE].iloc[0]
    assert base.composite == pytest.approx(1)
    for row in frame.itertuples():
        assert row.composite == pytest.approx(
            row.marginal_contribution + row.joint_contribution + row.tail_contribution
        )
    frame, _, skipped = f.run_cases([item], ["2018-04-02"], f.configs(True), [17], 200, "select")
    assert frame.empty and "validation" in skipped[0]["reason"]


def test_locks_both_frequencies_and_refuses_repeat_or_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "inventory", lambda root: ([], [], {}))
    calls = []

    def run(items, origins, cfgs, seeds, draws, phase):
        calls.append((phase, [c["id"] for c in cfgs]))
        rows = [
            dict(
                config=c["id"],
                cluster=x,
                composite=0.9 if c["id"] == "daily_ewma60" else 1,
                family="T2-F1",
                monthly=c["id"].startswith("monthly"),
                seed=17,
                coverage90=0.9,
            )
            for c in cfgs
            for x in ["X", "Y"]
        ]
        return pd.DataFrame(rows), pd.DataFrame(), []

    monkeypatch.setattr(f, "run_cases", run)
    out = tmp_path / "run"
    f.main(["--run-dir", str(out)])
    locked = json.loads((out / "selected.json").read_text())
    assert locked["winners"] == {"daily": "daily_ewma60", "monthly": f.BASE}
    f.main(["--run-dir", str(out), "--phase", "holdout"])
    assert calls[-2][1] == [f.BASE, "daily_ewma60"] and calls[-1][1] == [f.BASE]
    with pytest.raises(SystemExit):
        f.main(["--run-dir", str(out), "--phase", "holdout"])
    out2 = tmp_path / "tampered"
    f.main(["--run-dir", str(out2)])
    locked = json.loads((out2 / "selected.json").read_text())
    locked["winners"]["monthly"] = "monthly_damped12"
    (out2 / "selected.json").write_text(json.dumps(locked))
    with pytest.raises(SystemExit):
        f.main(["--run-dir", str(out2), "--phase", "holdout"])
    assert not (out2 / "holdout_started.json").exists()
