"""## Executive summary (read this first)

Verify factor transformations, cutoff safety and long-gap branch coverage.
Use synthetic panels; no practice answer or score is a test fixture.
"""

import numpy as np
import pandas as pd

from qfbench2_track_forecasting.daily_candidate import estimate as daily_estimate
from qfbench2_track_forecasting.factor_candidate import POLICY
from qfbench2_track_forecasting.factor_candidate import estimate as factor_estimate
from qfbench2_track_forecasting.short_horizon import predict


def panel(frame):
    return (
        frame.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="asset", value_name="value")
    )


def test_factor_uses_log1p_zero_anchor_and_research_distribution():
    rng = np.random.default_rng(61)
    rows = rng.normal(0.0005, 0.01, (450, 3))
    frame = pd.DataFrame(
        rows, columns=["MKT", "HML", "MOM"], index=pd.bdate_range("2001-01-01", periods=450)
    )
    actual, ledger = factor_estimate(
        {"f": panel(frame)}, list(frame), [63, 7, 21], str(frame.index[-1].date()), 4096, 17
    )
    research = predict(np.log1p(rows[-300:]), np.zeros(3), [7, 21, 63], POLICY, 4096, 17)
    np.testing.assert_array_equal(actual[:, :, [1, 2, 0]], research)
    assert ledger["rows"] == 300 and ledger["anchor"] == 0


def test_factor_future_rows_and_asset_labels_do_not_change_forecast():
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(
        rng.normal(0, 0.01, (450, 2)),
        columns=["MKT", "MOM"],
        index=pd.bdate_range("2001-01-01", periods=450),
    )
    asof = str(frame.index[399].date())
    before, _ = factor_estimate({"f": panel(frame)}, list(frame), [7, 21], asof, 1024, 0)
    frame.iloc[400:] = 20
    frame.columns = ["X", "Y"]
    after, _ = factor_estimate({"f": panel(frame)}, list(frame), [7, 21], asof, 1024, 0)
    np.testing.assert_array_equal(before, after)


def test_long_gap_uses_only_old_steps_current_anchor_and_current_context():
    rng = np.random.default_rng(21)
    values = np.exp(np.cumsum(rng.normal(0, 0.006, (2100, 3)), axis=0)) * [1.0, 1.3, 100]
    frame = pd.DataFrame(
        values, columns=["EUR", "GBP", "JPY"], index=pd.bdate_range("2000-01-01", periods=2100)
    )
    asof = str(frame.index[1999].date())
    frame.loc[frame.index[501:1999], "EUR"] = np.nan
    data = panel(frame).dropna()
    actual, ledger = daily_estimate({"fx": data}, ["EUR"], [126, 21, 63], asof, 4096, 0)
    assert ledger["stale"] and ledger["configuration"]["df"] == 8
    assert ledger["context_assets"] == ["GBP", "JPY"]
    anchor = values[1999, :1]
    x = np.diff(np.log(values[202:501, :1]), axis=0) * anchor
    reference = predict(x, anchor, [21, 63, 126], ledger["configuration"], 4096, 0)
    np.testing.assert_array_equal(actual[:, :, [1, 2, 0]], reference)
    data.loc[data.date > asof, "value"] *= 100000
    after, _ = daily_estimate({"fx": data}, ["EUR"], [126, 21, 63], asof, 4096, 0)
    np.testing.assert_array_equal(actual, after)


def test_long_gap_without_context_keeps_scale_ratio_one():
    dates = pd.bdate_range("2000-01-01", periods=299).append(pd.DatetimeIndex(["2010-01-04"]))
    v = np.exp(np.random.default_rng(71).normal(0, 0.005, 300).cumsum())
    data = panel(pd.DataFrame({"EUR": v}, index=dates))
    samples, ledger = daily_estimate({"fx": data}, ["EUR"], [63], "2010-01-04", 512, 0)
    assert samples.shape == (512, 1, 1)
    assert ledger["context_variance_ratio"] == 1 and ledger["configuration"]["width"] == 1
