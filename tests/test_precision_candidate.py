"""## Executive summary (read this first)

Check low-noise normal sampling, coherent paths, fixed monthly means and M0 seeds.
Synthetic examples establish numerical behavior without any practice outcomes.
"""

import numpy as np
import pytest

from experiments import means, precision
from qfbench2_track_forecasting import cli
from qfbench2_track_forecasting.sampling import InnovationSampler


@pytest.mark.parametrize("method", ["sobol", "antithetic"])
def test_normal_sampling_is_reproducible_and_joint(method):
    first = InnovationSampler(17, 4096, 3, 2, method)
    repeat = InnovationSampler(17, 4096, 3, 2, method)
    np.testing.assert_array_equal(first.values, repeat.values)
    flattened = first.values.reshape(4096, -1)
    assert np.max(np.abs(flattened.mean(0))) < 0.01
    np.testing.assert_allclose(np.cov(flattened, rowvar=False), np.eye(6), atol=0.08)
    if method == "antithetic":
        np.testing.assert_array_equal(flattened[:2048], -flattened[2048:])
    first.standard_normal((4096, 3))
    first.standard_normal((4096, 3))
    with pytest.raises(ValueError):
        first.standard_normal((4096, 3))


def test_sobol_refuses_unbalanced_draw_count():
    with pytest.raises(ValueError):
        InnovationSampler(0, 1000, 1, 1, "sobol")


@pytest.mark.parametrize("monthly", [False, True])
def test_current_production_is_preserved(monthly):
    history = np.random.default_rng(8).normal(0.1, 0.3, (400, 2)).cumsum(0)
    steps = np.array([[2, 8], [3, 9]]) if monthly else [5, 21]
    cfg = precision.configs(monthly)[0]
    result = precision.predict(history, steps, False, monthly, cfg, 0)
    if monthly:
        original = means.predict(
            history, steps, False, "T2-F1", True, dict(model="baseline"), 1000, 0
        )
        np.testing.assert_array_equal(result, original)
    else:
        panels, assets, asof = precision.panels_for(history, False)
        original, _ = cli._draw(
            panels, assets, steps, asof, 1000, 0, drift_factor=0.5, variance_mix=0.5
        )
        np.testing.assert_array_equal(result, original.reshape(1000, -1))


def test_sobol_path_covariance():
    history = np.random.default_rng(9).normal(0, 1, (400, 1)).cumsum(0)
    panels, assets, asof = precision.panels_for(history, False)
    samples, stats = cli._draw(
        panels, assets, [5, 20], asof, 4096, 0, drift_factor=0.5, variance_mix=0.5, sampling="sobol"
    )
    covariance = np.cov(samples[:, 0, :], rowvar=False)
    sd = stats["daily_sd"][assets[0]]
    np.testing.assert_allclose(covariance, sd**2 * np.array([[5, 5], [5, 20]]), rtol=0.02)


def test_monthly_trend_changes_only_mean():
    history = np.random.default_rng(31).normal(0.1, 0.8, (180, 1)).cumsum(0)
    baseline_cfg = next(c for c in precision.configs(True) if c["id"] == "sobol4096")
    blend_cfg = next(c for c in precision.configs(True) if c["id"] == "trend12_mix0.5")
    base = precision.predict(history, np.array([[2, 8]]), False, True, baseline_cfg, 0)
    blend = precision.predict(history, np.array([[2, 8]]), False, True, blend_cfg, 0)
    np.testing.assert_allclose(blend - blend.mean(0), base - base.mean(0), atol=1e-12)
    reference_mean = means.baseline_mean(history, np.array([[2, 8]]), False, True).ravel()
    changed_mean = means.forecast_mean(
        history,
        np.array([[2, 8]]),
        False,
        True,
        dict(model="trend", window=12, half_life=12, strength=0.5),
    )
    np.testing.assert_allclose(
        blend - base, np.broadcast_to(changed_mean - reference_mean, base.shape), atol=1e-12
    )


def test_documented_m0_returns_are_raw_and_anchor_zero():
    history = np.full((300, 1), 0.01)
    samples = precision.m0_samples(history, [21], True, ["X"], "2011-01-03", "synthetic")
    assert abs(samples.mean() - 0.21) < 1e-5
    repeat = precision.m0_samples(history, [21], True, ["X"], "2011-01-03", "synthetic")
    np.testing.assert_array_equal(samples, repeat)
