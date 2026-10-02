"""## Executive summary (read this first)

Verify submission-baseline equivalence, heavy tails with matched variance,
whole-path mixture membership, fixed drift and the monotone variance clock.
Synthetic data only; no market outcomes or solved cards are stored here.
"""

import numpy as np
import pandas as pd
import pytest

from experiments import distributions as d
from experiments import joint as j
from qfbench2_track_forecasting import cli


def history():
    return np.cumsum(np.random.default_rng(91).normal(size=(400, 2)), axis=0)


def test_frozen_design_and_submission_baseline():
    cfgs = d.configs()
    assert len(cfgs) == len({c["id"] for c in cfgs}) == 17
    assert cfgs[0]["id"] == d.BASE
    values = history()
    dates = pd.bdate_range("2010-01-01", periods=len(values))
    panel = (
        pd.DataFrame(values, index=dates, columns=["A", "B"])
        .rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="asset_id", value_name="value")
    )
    samples, _ = cli._draw(
        {"synthetic": panel},
        ["A", "B"],
        [5, 20, 60],
        str(dates[-1].date()),
        1000,
        0,
        drift_factor=0.5,
        variance_mix=0.5,
    )
    actual = d.predict(values, [5, 20, 60], False, "T2-F1", False, cfgs[0], 1000, 0)
    np.testing.assert_allclose(actual, samples.reshape(1000, -1), atol=1e-12, rtol=1e-12)


def test_heavier_tails_preserve_variance_and_mean():
    values = history()
    args = (values, [5, 60], False, "T2-F1", False)
    normal = d.predict(*args, d.configs()[0], 60000, 17)
    thick = d.predict(*args, dict(components=[d.component(df=5)]), 60000, 17)
    ratio = thick.var(0) / normal.var(0)
    assert np.all((ratio > 0.93) & (ratio < 1.07))
    center = d.center(values, [5, 60], False)
    assert np.all(np.abs(thick.mean(0) - center) < 0.03 * normal.std(0))
    assert np.all(
        np.quantile(np.abs(thick - center), 0.999, axis=0)
        > 1.3 * np.quantile(np.abs(normal - center), 0.999, axis=0)
    )


def test_mixture_selects_complete_draws_not_averaged_quantiles():
    values = history()
    args = (values, [5, 20, 60], False, "T2-F2", False)
    first = d.predict(*args, dict(components=[d.component()]), 1000, 41)
    second = d.predict(*args, dict(components=[d.component(df=5, power=0.8)]), 1000, 41)
    mixed = d.predict(
        *args, dict(components=[d.component(0.5), d.component(0.5, df=5, power=0.8)]), 1000, 41
    )
    first_rows = (mixed == first).all(1)
    second_rows = (mixed == second).all(1)
    assert (first_rows | second_rows).all()
    assert 400 < first_rows.sum() < 600
    assert 400 < second_rows.sum() < 600


def test_clock_changes_only_long_horizon_variance():
    values = history()
    args = (values, [5, 20], False, "T2-F3", False)
    a = d.predict(*args, dict(components=[d.component(power=0.8)]), 1000, 0)
    b = d.predict(*args, dict(components=[d.component(power=1.2)]), 1000, 0)
    np.testing.assert_array_equal(a, b)
    for power in (0.8, 1.0, 1.2):
        assert (np.diff(j.variance_clock(np.arange(1, 301), power)) > 0).all()


def test_invalid_distribution_and_monthly_refused():
    args = (history(), [5], False, "T2-F1", False)
    with pytest.raises(ValueError, match="weights"):
        d.predict(*args, dict(components=[d.component(-1)]), 200, 0)
    with pytest.raises(ValueError, match="degrees"):
        d.predict(*args, dict(components=[d.component(df=2)]), 200, 0)
    with pytest.raises(ValueError, match="daily"):
        d.predict(history(), [1], False, "T2-F1", True, d.configs()[0], 200, 0)
