"""## Executive summary (read this first)

Verify production monthly mean changes against the frozen research calculation.
Check noise preservation, historical cutoff, and fallback for missing months.
"""

import numpy as np
import pandas as pd

from experiments import precision
from qfbench2_track_forecasting import cli
from qfbench2_track_forecasting.monthly_trend import adjust_monthly_mean


def test_production_monthly_mean_matches_frozen_research():
    history = np.random.default_rng(81).normal(0.1, 0.4, (180, 2)).cumsum(0)
    panels, assets, asof = precision.panels_for(history, True)
    steps = np.array([[2, 8], [3, 9]])
    samples, stats = cli._draw(
        panels, assets, [21, 189], asof, 4096, 0, panel_steps=steps, sampling="sobol"
    )
    changed, records = adjust_monthly_mean(samples, panels, assets, steps, asof, stats)
    cfg = next(c for c in precision.configs(True) if c["id"] == "trend12_mix0.5")
    expected = precision.predict(history, steps, False, True, cfg, 0)
    np.testing.assert_allclose(changed.reshape(4096, -1), expected, atol=1e-12)
    np.testing.assert_allclose(changed - changed.mean(0), samples - samples.mean(0), atol=1e-12)
    assert set(records) == set(assets)
    future = {name: frame.copy() for name, frame in panels.items()}
    frame = future["history"]
    for asset in assets:
        frame.loc[len(frame)] = [pd.Timestamp(asof) + pd.DateOffset(months=1), asset, 1e8]
    repeated, _ = adjust_monthly_mean(samples, future, assets, steps, asof, stats)
    np.testing.assert_array_equal(changed, repeated)


def test_recent_month_gap_preserves_original_mean():
    history = np.arange(60, dtype=float)[:, None]
    panels, assets, asof = precision.panels_for(history, True)
    frame = panels["history"]
    panels["history"] = frame.drop(frame.index[-5])
    original = np.zeros((200, 1, 1))
    stats = {"step_drift": {assets[0]: 1.0}}
    changed, records = adjust_monthly_mean(original, panels, assets, np.array([[2]]), asof, stats)
    np.testing.assert_array_equal(changed, original)
    assert records == {}
