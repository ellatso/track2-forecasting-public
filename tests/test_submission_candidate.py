"""## Executive summary (read this first)

Verify that daily production samples match the frozen research candidate, monthly
samples stay unchanged, and future panel mutations cannot alter forecasts.
"""

import numpy as np
import pandas as pd
import pytest

from experiments import joint
from qfbench2_track_forecasting import cli


def panel(frame):
    return (
        frame.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="asset_id", value_name="value")
    )


@pytest.mark.parametrize("returns", [False, True])
def test_daily_production_matches_frozen_research_model(returns):
    rng = np.random.default_rng(9)
    data = rng.normal(0.001, 0.01, (450, 3))
    if not returns:
        data = data.cumsum(axis=0)
    frame = pd.DataFrame(
        data, index=pd.bdate_range("2010-01-04", periods=450), columns=["X", "Y", "Z"]
    )
    asof = str(frame.index[-1].date())
    samples, _ = cli._draw(
        {"test": panel(frame)},
        list(frame.columns),
        [5, 21, 63],
        asof,
        1000,
        0,
        target_type="log_return" if returns else "level",
        drift_factor=0.5,
        variance_mix=0.5,
    )
    cfg = dict(id="candidate", window=300, drift=0.5, mix=0.5, shrink=0.0, power=1.0)
    expected = joint.predict(data, [5, 21, 63], returns, "T2-F1", False, cfg, 1000, 0)
    np.testing.assert_allclose(samples.reshape(1000, -1), expected, rtol=1e-12, atol=1e-12)


def test_monthly_candidate_is_exactly_original_baseline():
    frame = pd.DataFrame(
        {"X": np.arange(100) * 0.01 + np.sin(np.arange(100))},
        index=pd.date_range("2000-01-01", periods=100, freq="MS"),
    )
    steps = np.array([[2, 8]])
    args = ({"test": panel(frame)}, ["X"], [21, 189], "2008-04-01", 1000, 0)
    before = cli._draw(*args, panel_steps=steps)
    after = cli._draw(*args, panel_steps=steps, drift_factor=0.5, variance_mix=0.5)
    np.testing.assert_array_equal(before[0], after[0])
    assert before[1] == after[1]


def test_future_panel_rows_cannot_change_candidate():
    frame = pd.DataFrame(
        {"X": np.sin(np.arange(500) / 10)}, index=pd.bdate_range("2010-01-04", periods=500)
    )
    asof = str(frame.index[399].date())
    first = cli._draw(
        {"test": panel(frame)}, ["X"], [21], asof, 1000, 0, drift_factor=0.5, variance_mix=0.5
    )[0]
    frame.iloc[400:] += 1000000
    second = cli._draw(
        {"test": panel(frame)}, ["X"], [21], asof, 1000, 0, drift_factor=0.5, variance_mix=0.5
    )[0]
    np.testing.assert_array_equal(first, second)
