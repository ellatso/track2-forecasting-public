"""## Executive summary (read this first)

Check whole-path mixture behavior and input-only controls with synthetic inputs.
No published answer or measured practice result is stored in these tests.
"""

import numpy as np
import pandas as pd
import pytest

from experiments import practice_ps1 as study
from qfbench2_track_forecasting.short_horizon import predict


def inputs():
    values = 5 + np.random.default_rng(72).normal(0, 0.01, (400, 2)).cumsum(0)
    hist = pd.DataFrame(
        values, columns=["EUR", "JPY"], index=pd.bdate_range("2000-01-03", periods=400)
    )
    changes = np.diff(values[-300:], axis=0)
    base = predict(changes, values[-1], [1, 2, 4], dict(vol="v9", drift=0.5), 256, 0)
    return hist, changes, base


def test_mixture_selects_whole_paths_and_is_repeatable(monkeypatch):
    monkeypatch.setattr(study, "DRAWS", 256)
    hist, changes, base = inputs()
    other = predict(changes, hist.iloc[-1], [1, 2, 4], dict(vol="sample", drift=0), 256, 0)
    mixed = study.transformed(base, hist, changes, [1, 2, 4], "mix50", 0)
    is_base = (mixed == base).all(axis=(1, 2))
    is_other = (mixed == other).all(axis=(1, 2))
    assert (is_base | is_other).all()
    assert is_other.sum() == 128
    np.testing.assert_array_equal(
        mixed, study.transformed(base, hist, changes, [1, 2, 4], "mix50", 0)
    )


def test_bill_policy_leaves_fx_exactly_unchanged():
    hist, changes, base = inputs()
    np.testing.assert_array_equal(
        base, study.transformed(base, hist, changes, [1, 2, 4], "online_bills", 0)
    )
    with pytest.raises(ValueError, match="Unknown"):
        study.transformed(base, hist, changes, [1, 2, 4], "typo", 0)
