"""## Executive summary (read this first)

Check cited event dates and unchanged forecasts before a future event.
Use synthetic paths and quotations, never historical answers or model calls.
"""

import numpy as np
import pytest

from qfbench2_track_forecasting import text_scenarios
from qfbench2_track_forecasting.event_timing import observation_horizon


def fixture():
    base = np.random.default_rng(13).normal(size=(512, 2, 4)).cumsum(2)
    assets, horizons = ["UST_2Y", "EUR"], [4, 1, 3, 2]
    mean, sd = base.mean(0), base.std(0)
    summary = {"assets": {a: {"stale_fx": False} for a in assets}}
    return base, assets, horizons, summary, mean, sd


def reply(date="2000-01-06"):
    return dict(
        scenarios=[
            dict(
                weight=1,
                assets={
                    "UST_2Y": dict(
                        shift_risk_sd=0.4,
                        vol_scale=1.4,
                        confidence=0.5,
                        regime_changed=False,
                        evidence_ids=["e0"],
                        activation_date=date,
                    )
                },
            )
        ]
    )


def test_future_event_cannot_change_any_pre_event_path():
    base, assets, horizons, summary, mean, sd = fixture()
    spec = dict(
        shift_risk_sd=0.2,
        vol_scale=1.4,
        confidence=0.5,
        regime_changed=False,
        event_horizon=3,
    )
    scenarios = [dict(weight=1, assets={"UST_2Y": spec})]
    out = text_scenarios.apply(base, assets, horizons, summary, mean, sd, sd, scenarios, 0)
    np.testing.assert_array_equal(out[:, :, [1, 3]], base[:, :, [1, 3]])
    np.testing.assert_array_equal(out[:, 1], base[:, 1])
    assert ((out == base).all(axis=(1, 2))).sum() == 256
    assert not np.array_equal(out[:, 0, [0, 2]], base[:, 0, [0, 2]])


@pytest.mark.parametrize(
    "quote",
    [
        "The decision is scheduled for 2000-01-06, and its outcome is unknown.",
        "The decision is scheduled for January 6, 2000, and its outcome is unknown.",
        "The decision is scheduled for 6 January 2000, and its outcome is unknown.",
        "The meeting is scheduled for January 5–6, 2000; the decision is on its final day.",
    ],
)
def test_activation_date_requires_a_verbatim_cited_calendar_date(quote):
    ev = [dict(id="e0", assets=["UST_2Y"], quote=quote)]
    sc = text_scenarios.validate(reply(), ev, ["UST_2Y"], [4, 1, 3, 2], asof="2000-01-03")
    assert sc[0]["assets"]["UST_2Y"]["event_horizon"] == 3


@pytest.mark.parametrize(
    "date,quote",
    [
        ("2000-01-06", "There is a scheduled decision; no date was supplied."),
        ("2000-01-03", "A decision is scheduled for 2000-01-03."),
        ("2000-01-20", "A decision is scheduled for 2000-01-20."),
        ("2000-99-06", "A decision is scheduled for 2000-99-06."),
    ],
)
def test_invalid_past_outside_window_or_invented_event_date_cannot_adjust(date, quote):
    ev = [dict(id="e0", assets=["UST_2Y"], quote=quote)]
    assert (
        text_scenarios.validate(reply(date), ev, ["UST_2Y"], [1, 2, 3, 4], asof="2000-01-03") == []
    )


def test_afternoon_us_event_enters_treasury_same_day_and_ecb_fx_the_next_day():
    spec = dict(
        activation_date="2000-01-05",
        activation_time="14:00",
        activation_timezone="America/New_York",
    )
    evidence = [
        dict(quote="A decision is scheduled for January 5, 2000 at 2:00 p.m. Eastern Time.")
    ]
    assert observation_horizon(spec, evidence, "2000-01-03", [1, 2, 3, 4], "UST_2Y") == 2
    assert observation_horizon(spec, evidence, "2000-01-03", [1, 2, 3, 4], "EUR") == 3


def test_unquoted_intraday_time_is_not_inferred_from_the_event_name():
    spec = dict(
        activation_date="2000-01-05",
        activation_time="14:00",
        activation_timezone="America/New_York",
    )
    evidence = [dict(quote="A decision is scheduled for January 5, 2000; the hour is unspecified.")]
    with pytest.raises(ValueError, match="jointly quoted"):
        observation_horizon(spec, evidence, "2000-01-03", [1, 2, 3, 4], "EUR")


def test_summer_us_release_after_ecb_fix_enters_next_day():
    spec = dict(
        activation_date="2000-07-05",
        activation_time="08:30",
        activation_timezone="America/New_York",
    )
    evidence = [dict(quote="Release on July 5, 2000 at 8:30 a.m. Eastern Time.")]
    # 12:30 UTC is after the 12:10 UTC summer ECB fix.
    assert observation_horizon(spec, evidence, "2000-07-03", [1, 2, 3, 4], "EUR") == 3


def test_us_europe_dst_mismatch_can_put_early_us_release_before_ecb_fix():
    spec = dict(
        activation_date="2010-11-03",
        activation_time="08:30",
        activation_timezone="America/New_York",
    )
    evidence = [dict(quote="Release on November 3, 2010 at 8:30 a.m. Eastern Time.")]
    assert observation_horizon(spec, evidence, "2010-11-01", [1, 2, 3, 4], "EUR") == 2
