"""## Executive summary (read this first)

Check research parity, noise preservation, cutoff safety, and gap fallback.
Use synthetic series only; no practice-unit outcomes are used.
"""

import numpy as np
import pandas as pd
import pytest

from qfbench2_track_forecasting import cli
from qfbench2_track_forecasting.mean_blend import apply_daily_mean_blend
from tests.test_submission_candidate import panel


@pytest.mark.parametrize("returns", [False, True])
def test_research_parity_and_fixed_noise(returns):
    rng = np.random.default_rng(9)
    data = rng.normal(0.001, 0.01, (450, 3))
    if not returns:
        data = data.cumsum(0)
    frame = pd.DataFrame(data, index=pd.bdate_range("2010-01-04", periods=450), columns=list("XYZ"))
    panels = {"test": panel(frame)}
    asof = str(frame.index[-1].date())
    horizons = [5, 21, 63]
    target = "log_return" if returns else "level"
    samples, stats = cli._draw(
        panels,
        list(frame.columns),
        horizons,
        asof,
        1000,
        0,
        target_type=target,
        drift_factor=0.5,
        variance_mix=0.5,
    )
    result, applied = apply_daily_mean_blend(
        samples, panels, list(frame.columns), horizons, asof, stats, target
    )
    assert applied == list(frame.columns)
    # Independent expression of the frozen research AR mean.
    recent = np.log1p(data[-120:]) if returns else data[-120:]
    x, y = recent[:-1], recent[1:]
    xc, yc = x - x.mean(0), y - y.mean(0)
    phi = (xc * yc).sum(0) / np.maximum((xc * xc).sum(0), 1e-12)
    phi = np.clip(phi, -0.5, 0.5) if returns else np.clip(phi, 0, 0.995)
    grid = np.array(horizons)[None, :]
    rate = (np.log1p(data[-300:]) if returns else np.diff(data[-300:], axis=0)).mean(0) * 0.5
    anchor = np.zeros(3) if returns else data[-1]
    old = anchor[:, None] + rate[:, None] * grid
    if returns:
        proposed = rate[:, None] * grid + (recent[-1] - rate)[:, None] * phi[:, None] * (
            1 - phi[:, None] ** grid
        ) / (1 - phi[:, None])
    else:
        equilibrium = (y.mean(0) - phi * x.mean(0)) / (1 - phi)
        proposed = equilibrium[:, None] + (anchor - equilibrium)[:, None] * phi[:, None] ** grid
    np.testing.assert_allclose(result, samples + 0.5 * (proposed - old)[None, :, :], atol=1e-12)
    np.testing.assert_allclose(result - result.mean(0), samples - samples.mean(0), atol=1e-12)
    future = frame.copy()
    future.loc[frame.index[-1] + pd.Timedelta(days=1)] = 1e6
    repeated, _ = apply_daily_mean_blend(
        samples, {"test": panel(future)}, list(frame.columns), horizons, asof, stats, target
    )
    np.testing.assert_array_equal(result, repeated)


def test_gap_series_retains_baseline():
    dates = pd.bdate_range("2000-01-03", periods=100).append(pd.DatetimeIndex(["2024-01-01"]))
    frame = pd.DataFrame({"X": np.arange(101, dtype=float)}, index=dates)
    samples = np.zeros((200, 1, 1))
    stats = {"last": {"X": 100.0}, "daily_drift": {"X": 0.5}}
    result, applied = apply_daily_mean_blend(
        samples, {"test": panel(frame)}, ["X"], [21], "2024-01-01", stats, "level"
    )
    assert applied == []
    np.testing.assert_array_equal(result, samples)
