"""## Executive summary (read this first)

Exercise the newly announced daily currencies without any historical answers.
Check cutoff isolation, authored horizon order and finite joint forecasts.
"""

import numpy as np
import pandas as pd
import pytest

from experiments.final_daily import GROUPS, ecb_to_usd
from qfbench2_track_forecasting.daily_candidate import estimate
from qfbench2_track_forecasting.text_scenarios import summarize


@pytest.mark.parametrize("asset", ["MXN", "ZAR", "KRW"])
def test_new_final_currency_receives_fx_policy_and_correct_quote(asset):
    dates = pd.bdate_range("2000-01-03", periods=420)
    values = 15 * np.exp(np.random.default_rng(19).normal(0, 0.006, 420).cumsum())
    panel = pd.DataFrame(dict(date=dates, asset=asset, value=values))
    asof = str(dates[399].date())
    result = estimate({"fx": panel}, [asset], [4, 1, 3, 2], asof, 512, 0)
    assert result is not None
    draws, ledger = result
    assert draws.shape == (512, 1, 4) and np.isfinite(draws).all()
    assert not ledger["stale"]
    panel.loc[panel.date > asof, "value"] *= 10000
    after, _ = estimate({"fx": panel}, [asset], [4, 1, 3, 2], asof, 512, 0)
    np.testing.assert_array_equal(draws, after)
    summary, *_ = summarize(
        {"fx": panel}, [asset], [4, 1, 3, 2], asof, draws, dict(target_type="level")
    )
    assert summary["assets"][asset]["unit"].startswith("foreign units per USD;")


def test_ecb_cross_rate_directions_use_one_same_day_usd_reference():
    currencies = {a for g, aa in GROUPS.items() if g.startswith("fx") for a in aa}
    row = {a: 6.0 for a in currencies if a != "EUR"}
    row["USD"] = 2.0
    data = pd.DataFrame([row], index=pd.DatetimeIndex(["2000-01-03"]))
    result = ecb_to_usd(data)
    assert result.loc["2000-01-03", "EUR"] == 2.0
    for asset in ["AUD", "GBP", "NZD"]:
        assert result.loc["2000-01-03", asset] == 1 / 3
    for asset in ["JPY", "CHF", "MXN", "ZAR", "KRW", "INR", "BRL"]:
        assert result.loc["2000-01-03", asset] == 3.0
