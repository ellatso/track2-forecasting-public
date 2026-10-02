"""## Executive summary (read this first)

Check mean-only changes, exact reference controls, monthly per-asset step grids,
reversion and damped-trend direction, and future-mutation independence.
All fixtures are synthetic. No private card result is stored in this repository.
"""

import numpy as np
import pandas as pd
import pytest

from experiments import batch
from experiments import means as m
from qfbench2_track_forecasting import cli


def values():
    return np.cumsum(np.random.default_rng(37).normal(size=(420, 2)), axis=0)


@pytest.mark.parametrize("monthly", [False, True])
def test_baseline_matches_actual_submission_sampler(monthly):
    data = values()
    dates = (
        pd.date_range("2000-01-01", periods=len(data), freq="MS")
        if monthly
        else pd.bdate_range("2000-01-03", periods=len(data))
    )
    frame = pd.DataFrame(data, index=dates, columns=["A", "B"])
    panel = (
        frame.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="asset_id", value_name="value")
    )
    steps = np.array([[2, 5], [3, 6]]) if monthly else np.array([5, 20, 60])
    keys = [42, 65] if monthly else [5, 20, 60]
    expected, _ = cli._draw(
        {"synthetic": panel},
        ["A", "B"],
        keys,
        str(dates[-1].date()),
        1000,
        17,
        panel_steps=steps if monthly else None,
        drift_factor=0.5,
        variance_mix=0.5,
    )
    actual = m.predict(data, steps, False, "T2-F1", monthly, m.configs(monthly)[0], 1000, 17)
    np.testing.assert_allclose(actual, expected.reshape(1000, -1), rtol=1e-12, atol=1e-12)
    assert len(m.configs(monthly)) == 17


@pytest.mark.parametrize("monthly,returns", [(False, False), (False, True), (True, False)])
def test_all_arms_change_only_mean_with_identical_noise(monthly, returns):
    data = values()
    if returns:
        data = np.tanh(data / 100) * 0.01
    steps = np.array([[2, 5], [3, 6]]) if monthly else [5, 20, 60]
    args = (data, steps, returns, "T2-F1", monthly)
    baseline = m.predict(*args, m.configs(monthly)[0], 1000, 41)
    old = m.baseline_mean(data, steps, returns, monthly).ravel()
    for cfg in m.configs(monthly):
        actual = m.predict(*args, cfg, 1000, 41)
        new = m.forecast_mean(data, steps, returns, monthly, cfg)
        np.testing.assert_allclose(
            actual - baseline, np.broadcast_to(new - old, actual.shape), atol=1e-11, rtol=1e-10
        )
        np.testing.assert_allclose(
            np.cov(actual, rowvar=False), np.cov(baseline, rowvar=False), atol=1e-10, rtol=1e-10
        )


def test_reversion_moves_stretched_level_back():
    data = np.sin(np.arange(400) / 4)[:, None]
    data[-1] = 5
    cfg = dict(model="ar", window=120, strength=1)
    result = m.forecast_mean(data, [5, 20, 60], False, False, cfg)
    assert np.all(result < 5)
    assert abs(result[-1]) < abs(result[0])


def test_damped_trend_uses_observation_counts_and_flattens():
    data = (np.arange(400) * 0.01)[:, None]
    cfg = dict(model="trend", window=60, half_life=20, strength=1)
    forecast = m.forecast_mean(data, [5, 60, 120], False, False, cfg)
    rho = np.exp2(-1 / 20)
    expected = data[-1, 0] + 0.005 * rho * (1 - rho ** np.array([5, 60, 120])) / (1 - rho)
    np.testing.assert_allclose(forecast, expected)
    assert forecast[2] - forecast[1] < forecast[1] - data[-1, 0]


@pytest.mark.parametrize("monthly", [False, True])
def test_future_values_never_change_predictions(monthly):
    data = values()
    dates = (
        pd.date_range("2000-01-01", periods=len(data), freq="MS")
        if monthly
        else pd.bdate_range("2000-01-03", periods=len(data))
    )
    frame = pd.DataFrame(data, index=dates, columns=["A", "B"])
    index = 350
    origin = str(dates[index].date())
    steps = np.array([[2, 5], [3, 6]]) if monthly else np.array([5, 20])
    item = dict(frame=frame, steps=steps, returns=False, monthly=monthly)
    before, _, _, _ = batch.case_at(item, origin)
    changed = frame.copy()
    changed.iloc[index + 1 :] += 100000
    after, _, _, _ = batch.case_at(dict(item, frame=changed), origin)
    for cfg in m.configs(monthly):
        np.testing.assert_array_equal(
            m.predict(before, steps, False, "T2-F1", monthly, cfg, 200, 0),
            m.predict(after, steps, False, "T2-F1", monthly, cfg, 200, 0),
        )
