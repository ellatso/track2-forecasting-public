"""## Executive summary (read this first)

Check release-unit arithmetic and publication-vintage isolation on synthetic data.
No forecast-quality claim, historical release answer or live API is involved.
"""

import numpy as np
import pandas as pd
import pytest

from qfbench2_track_forecasting.release_units import latest_vintage, transform_draws


def test_quarterly_growth_is_compounded_and_not_multiplied_by_four():
    history = pd.Series([100.0], index=pd.PeriodIndex(["2000Q1"], freq="Q"))
    result = transform_draws([101.0], history, pd.Period("2000Q2"), metric="annualized_quarter_pct")
    np.testing.assert_allclose(result, [4.060401])
    assert result[0] != 4.0


def test_monthly_and_yearly_inflation_use_different_base_periods():
    history = pd.Series([100.0, 102.0], index=pd.PeriodIndex(["1999-02", "2000-01"], freq="M"))
    target = pd.Period("2000-02", freq="M")
    monthly = transform_draws([103.0], history, target, metric="period_pct")
    yearly = transform_draws([103.0], history, target, metric="year_pct")
    np.testing.assert_allclose(monthly, [100 / 102])
    np.testing.assert_allclose(yearly, [3.0])


def test_later_revisions_and_future_periods_cannot_change_available_vintage():
    records = pd.DataFrame(
        dict(
            period=["2000-01", "2000-01", "2000-02"],
            released_at=["2000-02-15", "2000-03-15", "2000-03-15"],
            value=[100, 101, 102],
        )
    )
    first = latest_vintage(records, "2000-02-20", frequency="M")
    assert first.to_dict() == {pd.Period("2000-01"): 100}
    records["value"] = records.value.astype(object)
    records.loc[records.released_at > "2000-02-20", "value"] = "invalid future data"
    after = latest_vintage(records, "2000-02-20", frequency="M")
    pd.testing.assert_series_equal(first, after)


def test_conflicting_same_timestamp_is_rejected():
    records = pd.DataFrame(
        dict(period=["2000-01"] * 2, released_at=["2000-02-15"] * 2, value=[100, 101])
    )
    with pytest.raises(ValueError, match="Conflicting"):
        latest_vintage(records, "2000-02-20", frequency="M")


def test_publication_dates_are_required_even_when_values_look_plausible():
    with pytest.raises(ValueError, match="publication dates"):
        latest_vintage(
            pd.DataFrame(dict(period=["2000-01"], value=[100])), "2000-02-20", frequency="M"
        )


def test_missing_comparison_month_cannot_be_replaced_with_nearest_month():
    history = pd.Series([100.0], index=pd.PeriodIndex(["2000-01"], freq="M"))
    with pytest.raises(ValueError, match="matching comparison period"):
        transform_draws([101.0], history, pd.Period("2000-03", freq="M"), metric="period_pct")


def test_count_scale_requires_an_explicit_conversion():
    result = transform_draws([200000.0], None, None, metric="count", unit_scale=0.001)
    np.testing.assert_array_equal(result, [200.0])
    with pytest.raises(ValueError, match="negative"):
        transform_draws([-1.0], None, None, metric="count")


def test_no_released_observations_returns_an_empty_history():
    records = pd.DataFrame(dict(period=["2000-01"], released_at=["2000-02-15"], value=[100]))
    result = latest_vintage(records, "1999-12-31", frequency="M")
    assert result.empty and isinstance(result.index, pd.PeriodIndex)
