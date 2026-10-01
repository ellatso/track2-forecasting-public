"""## Executive summary (read this first)

Test the isolated variance/correlation/path-clock controls, unchanged baseline,
per-case balance, and tamper/repeat guards on synthetic data.
"""

import json

import numpy as np
import pandas as pd
import pytest

from experiments import batch, frequency
from experiments import joint as j


def test_factorial_grid_and_fixed_controls():
    cfgs = j.configs()
    assert len(cfgs) == len({c["id"] for c in cfgs}) == 12
    assert all(c["drift"] == 1 and c["window"] == 300 for c in cfgs)
    assert cfgs[0]["id"] == batch.BASE


def test_clock_monotonic_and_short_horizons_unchanged():
    horizons = np.arange(1, 301)
    clock = j.variance_clock(horizons, 0.8)
    assert np.all(np.diff(clock) > 0)
    np.testing.assert_array_equal(clock[:20], horizons[:20])
    assert clock[-1] < horizons[-1]
    np.testing.assert_allclose(j.variance_clock(horizons, 1), horizons)


def test_base_preserved_and_short_clock_paths_identical():
    history = np.cumsum(np.random.default_rng(1).normal(size=(400, 2)), axis=0)
    cfg = j.configs()[0]
    np.testing.assert_array_equal(
        j.predict(history, [5, 20], False, "T2-F3", False, cfg, 200, 17),
        batch.predict(
            history, [5, 20], False, "T2-F3", False, frequency.configs(False)[0], 200, 17
        ),
    )
    a = dict(cfg, id="test", mix=0.25, shrink=0.25, power=1)
    b = dict(a, power=0.8)
    np.testing.assert_array_equal(
        j.predict(history, [5, 20], False, "T2-F3", False, a, 200, 17),
        j.predict(history, [5, 20], False, "T2-F3", False, b, 200, 17),
    )


def test_shrinkage_does_not_change_single_asset_paths():
    history = np.cumsum(np.random.default_rng(2).normal(size=(400, 1)), axis=0)
    a = dict(j.configs()[0], id="a", mix=0.25)
    b = dict(a, shrink=0.25)
    np.testing.assert_array_equal(
        j.predict(history, [5, 60], False, "T2-F2", False, a, 200, 17),
        j.predict(history, [5, 60], False, "T2-F2", False, b, 200, 17),
    )


def test_monthly_refused():
    with pytest.raises(ValueError, match="daily"):
        j.predict(np.ones((80, 1)), [1, 2], False, "T2-F1", True, j.configs()[0], 200, 17)


def test_lock_only_selected_validation_and_no_repeats(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "inventory", lambda root: ([], [], {}))
    winner = j.configs()[1]["id"]

    def fake(items, origins, cfgs, seeds, draws, phase):
        calls.append((list(origins), [c["id"] for c in cfgs]))
        return (
            pd.DataFrame(
                [
                    dict(
                        config=c["id"],
                        cluster=x,
                        composite=0.9 if c["id"] == winner else 1,
                        family="T2-F1",
                        monthly=False,
                        seed=17,
                        coverage90=0.9,
                    )
                    for c in cfgs
                    for x in ["X", "Y"]
                ]
            ),
            pd.DataFrame(),
            [],
        )

    monkeypatch.setattr(j, "run_cases", fake)
    out = tmp_path / "run"
    j.main(["--run-dir", str(out)])
    assert json.loads((out / "selected.json").read_text())["config"] == winner
    j.main(["--run-dir", str(out), "--phase", "holdout"])
    assert calls[-1] == (list(j.BOUNDS), [batch.BASE, winner])
    with pytest.raises(SystemExit):
        j.main(["--run-dir", str(out), "--phase", "holdout"])
    out2 = tmp_path / "changed"
    j.main(["--run-dir", str(out2)])
    plan = json.loads((out2 / "plan.json").read_text())
    plan["draws"] = 500
    (out2 / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(SystemExit):
        j.main(["--run-dir", str(out2), "--phase", "holdout"])
    assert not (out2 / "holdout_started.json").exists()


def test_validation_bounds_refuse_crossing_seen_2021_window():
    dates = pd.bdate_range("2000-01-03", "2021-04-01")
    item = dict(
        source="synthetic",
        group="X",
        cluster='["X"]',
        assets=["X"],
        horizons=[300],
        steps=np.array([300]),
        returns=False,
        monthly=False,
        family="T2-F1",
        card={"scoring": {"params": {}}},
        frame=pd.DataFrame({"X": np.sin(np.arange(len(dates)) / 12)}, index=dates),
    )
    frame, _, skipped = j.run_cases([item], ["2020-01-02"], j.configs(), [17], 200, "holdout")
    assert frame.empty and "outside locked period" in skipped[0]["reason"]
