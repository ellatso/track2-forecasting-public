"""## Executive summary (read this first)

Detect Sobol row aliasing in House membership and scenario labels. Synthetic
normal paths must keep both tails and independent labels, not just half the rows.
No outcomes or live model calls are used.
"""

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import qmc

from qfbench2_track_forecasting import text_scenarios


def adjusted(seed, shifts, vol=1.0):
    base = ndtri(qmc.Sobol(8, scramble=True, seed=seed).random_base2(12)).reshape(4096, 2, 4)
    mean, sd = base.mean(0), base.std(0, ddof=1)
    scenarios = [
        dict(
            weight=1 / len(shifts),
            assets={
                a: dict(shift_risk_sd=shift, vol_scale=vol, confidence=0.75, regime_changed=False)
                for a in ["A", "B"]
            },
        )
        for shift in shifts
    ]
    summary = dict(assets={a: dict(stale_fx=False) for a in ["A", "B"]})
    out = text_scenarios.apply(
        base, ["A", "B"], [1, 2, 3, 4], summary, mean, sd, sd, scenarios, seed
    )
    active = np.any(out != base, axis=(1, 2))
    return base, out, active, mean, sd


@pytest.mark.parametrize("seed", [0, 17, 41, 409])
def test_active_membership_balances_every_innovation_direction(seed):
    base, out, active, mean, _ = adjusted(seed, [0.4])
    assert active.sum() == 2048
    assert np.max(np.abs((base[active] > mean).mean(0) - 0.5)) < 0.05
    assert np.max(np.abs(base[active].mean(0) - mean)) < 0.08
    np.testing.assert_array_equal(out[~active], base[~active])
    np.testing.assert_array_equal(out, adjusted(seed, [0.4])[1])


@pytest.mark.parametrize("seed", [0, 17, 41, 409])
def test_scenario_labels_do_not_select_other_asset_or_horizon_signs(seed):
    base, out, active, mean, sd = adjusted(seed, [-0.5, 0.5])
    positive = (out[:, 0, -1] - base[:, 0, -1]) > 0
    assert 0.45 < positive[active].mean() < 0.55
    assert np.max(np.abs((base[positive] > mean).mean(0) - 0.5)) < 0.06
    negative = active & ~positive
    assert np.max(np.abs((base[negative] > mean).mean(0) - 0.5)) < 0.06
    # A shift alone must not systematically contract a marginal.
    assert np.min(out.std(0, ddof=1) / sd) > 0.97


@pytest.mark.parametrize("seed", [0, 17, 41, 409])
def test_zero_shift_widens_both_tails_without_moving_the_centre(seed):
    base, out, _, mean, sd = adjusted(seed, [0.0], vol=2.0)
    q0, q1 = (np.quantile(x, [0.01, 0.99], axis=0) for x in [base, out])
    assert np.min((mean - q1[0]) / (mean - q0[0])) > 1.4
    assert np.min((q1[1] - mean) / (q0[1] - mean)) > 1.4
    assert np.max(np.abs((out.mean(0) - mean) / sd)) < 0.04
