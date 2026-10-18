"""## Executive summary (read this first)

Test deployment dispatch and marginal preservation using synthetic series only.
Do not infer forecast quality or record realized practice outcomes here.
"""

import numpy as np
import pandas as pd
import pytest

from experiments.production_audit import couple, forecast


def test_final_monday_origins_never_roll_back_to_friday():
    from experiments.final_daily import select_origins

    dates = pd.bdate_range("2000-01-03", "2000-02-01")
    dates = dates[dates != pd.Timestamp("2000-01-17")]
    origins = select_origins("2000-01-04", "2000-02-01", 1, 0, dates)
    assert list(origins) == list(pd.to_datetime(["2000-01-10", "2000-01-24", "2000-01-31"]))
    assert (origins.weekday == 0).all()
    with pytest.raises(ValueError):
        select_origins("2000-01-03", "2000-02-01", 0)


@pytest.mark.parametrize(
    "assets",
    [
        ["EUR", "JPY"],
        ["UST_7Y", "UST_20Y"],
        ["UST_2Y", "EUR"],
    ],
)
def test_real_dispatch_and_coupling_preserve_all_marginals(assets):
    x = np.random.default_rng(27).normal(0, 0.01, (420, len(assets))).cumsum(0) + 5
    frame = pd.DataFrame(x, index=pd.bdate_range("2000-01-03", periods=420), columns=assets)
    base = forecast(frame, [1, 2, 3, 4], 512, 17)
    for mode in ["ew60", "shrink50"]:
        changed = forecast(frame, [1, 2, 3, 4], 512, 17, coupling=mode)
        np.testing.assert_array_equal(np.sort(base, axis=0), np.sort(changed, axis=0))
        assert not np.array_equal(base, changed)
    assert base.shape == (512, len(assets), 4)


def test_forecast_uses_production_function_not_a_recreated_sampler(monkeypatch):
    calls = []

    def spy(panels, card, path, asof, draws, seed):
        calls.append((panels, card, asof, seed))
        return np.full((draws, 1, 4), 123.0), {}, False, None

    monkeypatch.setattr("experiments.production_audit.numeric_forecast", spy)
    frame = pd.DataFrame({"EUR": [1.0]}, index=pd.to_datetime(["2000-01-03"]))
    assert (forecast(frame, [1, 2, 3, 4], 512, 17) == 123).all()
    assert calls[0][2:] == ("2000-01-03", 17)
    assert calls[0][1]["targets"]["target_frequency"] == "daily"


def test_rank_coupling_rejects_missing_values_and_shape_mismatch():
    a = np.ones((512, 2, 4))
    with pytest.raises(ValueError):
        couple(a, a[:256])
    with pytest.raises(ValueError):
        couple(a, a * np.nan)


def test_width_experiment_preserves_centre_and_scales_standard_deviation():
    x = np.random.default_rng(28).normal(0, 0.01, (420, 1)).cumsum(0) + 5
    frame = pd.DataFrame(x, index=pd.bdate_range("2000-01-03", periods=420), columns=["EUR"])
    base = forecast(frame, [1, 2, 3, 4], 512, 17)
    changed = forecast(frame, [1, 2, 3, 4], 512, 17, width=0.8)
    np.testing.assert_allclose(changed.mean(0), base.mean(0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(changed.std(0), base.std(0) * 0.8, rtol=1e-10)


def test_online_calibration_uses_prior_window_for_each_error():
    from experiments.production_audit import prequential_width

    # This calculation checks the temporal split, not fit-to-outcome residuals.
    rng = np.random.default_rng(700)
    changes = rng.normal(0, 0.01, 450)
    frame = pd.DataFrame(
        {"UST_2Y": 5 + np.r_[0, changes.cumsum()]}, index=pd.bdate_range("2000-01-03", periods=451)
    )
    errors = []
    for i in range(len(changes) - 60, len(changes)):
        past = changes[i - 299 : i]
        w = np.exp2(-np.arange(298, -1, -1) / 60)
        w /= w.sum()
        variance = 0.5 * past.var(ddof=1) + 0.5 * np.sum(w * (past - w @ past) ** 2)
        errors.append((changes[i] - 0.5 * past.mean()) ** 2 / variance)
    wanted = np.clip(np.sqrt(np.mean(errors)), 0.8, 1.2)
    np.testing.assert_allclose(prequential_width(frame), [wanted], rtol=1e-10)
    assert np.isfinite(forecast(frame, [1, 2, 3, 4], 512, 17, online_width=True)).all()


def test_extended_official_source_reader_keeps_other_tenors_and_common_dates(tmp_path):
    import zipfile

    from experiments.final_daily import GROUPS, panels

    currencies = sorted({a for g, aa in GROUPS.items() if g.startswith("fx") for a in aa} - {"EUR"})
    csv = pd.DataFrame(
        6.0, index=pd.Index(["2000-01-03", "2000-01-04"], name="Date"), columns=["USD"] + currencies
    ).to_csv()
    with zipfile.ZipFile(tmp_path / "ecb-history.zip", "w") as z:
        z.writestr("eurofxref-hist.csv", csv)
    fields = ["3MONTH", "6MONTH", "1YEAR", "2YEAR", "5YEAR", "7YEAR", "10YEAR", "20YEAR", "30YEAR"]
    xml = (
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata" '
        'xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices">'
        "<entry><content><m:properties><d:NEW_DATE>2000-01-03</d:NEW_DATE>"
        + "".join(f"<d:BC_{f}>5.0</d:BC_{f}>" for f in fields)
        + "</m:properties></content></entry></feed>"
    )
    (tmp_path / "treasury-2000.xml").write_text(xml)
    frames, _ = panels(tmp_path, pd.Timestamp("2000-01-04"), extended=True)
    assert list(frames["rates_other"]) == ["UST_7Y", "UST_20Y"]
    assert list(frames["mixed_major"]) == ["UST_2Y", "UST_10Y", "EUR", "JPY"]
    assert list(frames["mixed_major"].index) == [pd.Timestamp("2000-01-03")]
    assert len(frames["fx_major"]) == 2  # No Treasury holiday was filled into a mixed case.
