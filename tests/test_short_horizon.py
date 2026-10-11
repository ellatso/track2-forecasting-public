"""## Executive summary (read this first)

Check the new research divisor against analytical and Monte Carlo expectations,
then check path sampling, unit scaling, heavy-tail variance and fixed seeds.
Synthetic data only; no practice outcome is recorded.
"""

import numpy as np
from qfbench2_common.scoring import crps
from scipy.special import ndtri

from experiments.expected_scale import normal_scales
from qfbench2_track_forecasting.short_horizon import predict


def test_normal_divisor_analytical_one_cell():
    s = normal_scales(np.array([3.0]), np.array([[0.01]]))
    np.testing.assert_allclose(s["marginal"], 0.1 / np.sqrt(np.pi))
    np.testing.assert_allclose(
        s["tail"],
        0.1 * np.exp(-0.5 * ndtri([0.01, 0.05, 0.95, 0.99]) ** 2).mean() / np.sqrt(2 * np.pi),
    )
    assert s["joint"] == 1.0


def test_joint_divisor_against_independent_monte_carlo():
    mean = np.array([0.5, 1.0, 1.1])
    cov = np.array([[0.3, 0.1, 0.08], [0.1, 0.2, 0.1], [0.08, 0.1, 0.25]])
    rng = np.random.default_rng(1)
    y = rng.multivariate_normal(mean, cov, size=300000)
    pair = np.sqrt(abs(y[:, :, None] - y[:, None, :]))
    measured = pair.var(axis=0).sum()
    np.testing.assert_allclose(normal_scales(mean, cov)["joint"], measured, rtol=0.01)


def test_v9_distribution_matches_mean_covariance_and_seed():
    x = np.random.default_rng(10).normal(size=(299, 2)) @ np.array([[0.2, 0.1], [0.0, 0.3]])
    cfg = dict(vol="v9", drift=0.5)
    y = predict(x, [2.0, 3.0], [2, 12], cfg, 8192, 0)
    np.testing.assert_array_equal(y, predict(x, [2.0, 3.0], [2, 12], cfg, 8192, 0))
    w = np.exp2(-np.arange(298, -1, -1) / 60)
    w /= w.sum()
    v = 0.5 * x.var(0, ddof=1) + 0.5 * (w[:, None] * (x - w @ x) ** 2).sum(0)
    mean = np.array([2.0, 3.0])[:, None] + 0.5 * x.mean(0)[:, None] * [2, 12]
    np.testing.assert_allclose(y.mean(0), mean, atol=0.003)
    np.testing.assert_allclose(y.var(0), v[:, None] * [2, 12], rtol=0.02)
    np.testing.assert_allclose(np.cov(y[:, 0, :], rowvar=False)[0, 1], 2 * v[0], rtol=0.04)


def test_fx_log_samples_positive_and_unit_equivariant():
    x = np.random.default_rng(12).normal(0, 0.01, (300, 2))
    cfg = dict(vol="ew20")
    one = predict(x, np.log([1.0, 150.0]), [2, 12], cfg, 1024, 17, np.array([True, True]))
    two = predict(x, np.log([2.0, 300.0]), [2, 12], cfg, 1024, 17, np.array([True, True]))
    assert np.all(one > 0)
    np.testing.assert_allclose(two, 2 * one)


def test_student_t_variance_and_tail_mass():
    x = np.random.default_rng(5).normal(size=(299, 1))
    a = predict(x, [0.0], [12], dict(vol="sample"), 16384, 0)
    b = predict(x, [0.0], [12], dict(vol="sample", df=5), 16384, 0)
    assert abs(b.var() / a.var() - 1) < 0.12
    assert (abs(b) > 3 * a.std()).mean() > (abs(a) > 3 * a.std()).mean()


def test_common_scorer_fair_crps_normal_expectation():
    rng = np.random.default_rng(100)
    x = rng.normal(size=(10000, 1))
    y = rng.normal(size=1500)
    measured = np.mean([crps.crps_marginal(x, np.array([v])) for v in y])
    np.testing.assert_allclose(
        measured, normal_scales(np.zeros(1), np.eye(1))["marginal"], rtol=0.05
    )


def panel(frame):
    return (
        frame.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="asset", value_name="value")
    )


def test_daily_container_path_matches_frozen_research_and_horizon_order():
    import pandas as pd

    from qfbench2_track_forecasting.daily_candidate import SHORT_FX, estimate

    rng = np.random.default_rng(6)
    values = np.exp(np.cumsum(rng.normal(0, 0.004, (450, 3)), axis=0)) * [1.0, 1.3, 100.0]
    frame = pd.DataFrame(
        values, index=pd.bdate_range("2010-01-01", periods=450), columns=["EUR", "GBP", "JPY"]
    )
    asof = str(frame.index[-1].date())
    actual, _ = estimate({"fx": panel(frame)}, list(frame), [12, 2, 5], asof, 4096, 17)
    exact = predict(np.diff(values[-300:], axis=0), values[-1], [2, 5, 12], SHORT_FX, 4096, 17)
    np.testing.assert_array_equal(actual[:, :, [1, 2, 0]], exact)


def test_daily_future_rows_never_enter_estimation_or_context():
    import pandas as pd

    from qfbench2_track_forecasting.daily_candidate import estimate

    rng = np.random.default_rng(7)
    values = np.exp(np.cumsum(rng.normal(0, 0.004, (2000, 3)), axis=0)) * [1.0, 1.3, 100.0]
    frame = pd.DataFrame(
        values, index=pd.bdate_range("2010-01-01", periods=2000), columns=["EUR", "GBP", "JPY"]
    )
    data = panel(frame)
    asof = str(frame.index[1700].date())
    # Target EUR has a missing middle but the current anchor remains.
    data = data[
        ~((data.asset == "EUR") & (data.date > frame.index[500]) & (data.date < frame.index[1700]))
    ]
    first, ledger = estimate({"fx": data}, ["EUR"], [2, 5, 12], asof, 4096, 0)
    assert ledger["stale"] and ledger["context_assets"] == ["GBP", "JPY"]
    modified = data.copy()
    modified.loc[modified.date > pd.Timestamp(asof), "value"] *= 10000
    second, _ = estimate({"fx": modified}, ["EUR"], [2, 5, 12], asof, 4096, 0)
    np.testing.assert_array_equal(first, second)
    assert first.std() < 0.1


def test_mixture_preserves_full_component_paths():
    x = np.random.default_rng(90).normal(size=(299, 2))
    h = [2, 5, 12]
    a = dict(vol="ew20", df=8, weight=0.75)
    b = dict(vol="v9", drift=0.5, weight=0.25)
    mixture = predict(x, [0.0, 1.0], h, dict(components=[a, b]), 512, 0)
    aa = predict(x, [0.0, 1.0], h, a, 512, 0)
    bb = predict(x, [0.0, 1.0], h, b, 512, 0)
    from_a = (mixture == aa).all(axis=(1, 2))
    from_b = (mixture == bb).all(axis=(1, 2))
    assert np.all(from_a | from_b) and 0.73 < from_a.mean() < 0.77


def test_short_rates_and_unsupported_domains_keep_v9():
    from qfbench2_track_forecasting.daily_candidate import estimate

    assert estimate({}, ["UST_2Y"], [2, 12], "2024-01-01", 4096, 0) is None
    assert estimate({}, ["UNKNOWN"], [21], "2024-01-01", 4096, 0) is None
