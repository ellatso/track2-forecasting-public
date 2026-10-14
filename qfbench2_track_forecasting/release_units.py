"""## Executive summary (read this first)

Prepare official-release research without guessing a new task-card schema.
Select one series' actually released vintage at a cutoff. Transform hypothetical
level draws to explicit index, count, percentage-change or annualized-quarter units.
These helpers do not nowcast a release, infer seasonal adjustment, fetch data,
or change the current forecast CLI. New-card adaptation awaits its published contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def latest_vintage(
    records, asof, *, frequency, period_col="period", release_col="released_at", value_col="value"
):
    """Return one series' last published values, not a current revised-data download.

    A date-only asof admits that UTC calendar day. Publication timestamps are
    required; an observation period alone is not proof of when data was available.
    """
    required = {period_col, release_col, value_col}
    if not required.issubset(records.columns):
        raise ValueError("Explicit observation periods and publication dates are required")
    cutoff = pd.Timestamp(asof, tz="UTC").normalize() + pd.Timedelta(days=1)
    d = records.copy()
    d["published"] = pd.to_datetime(d[release_col], utc=True, errors="coerce", format="mixed")
    if d.published.isna().any():
        raise ValueError("Missing or ambiguous publication timestamp")
    d = d[d.published < cutoff].copy()
    if "asset" in d and d.asset.nunique() > 1:
        raise ValueError("Select one release series before choosing its vintage")
    if d.empty:
        return pd.Series([], index=pd.PeriodIndex([], freq=frequency), dtype=float, name="value")
    d["observation"] = pd.PeriodIndex(d[period_col], freq=frequency)
    ended = d.observation.map(lambda p: p.end_time.tz_localize("UTC")) < cutoff
    d = d[ended].copy()
    d["number"] = pd.to_numeric(d[value_col], errors="coerce")
    if not np.isfinite(d.number.to_numpy()).all():
        raise ValueError("Nonfinite released observation")
    duplicates = d.groupby(["observation", "published"]).number.nunique()
    if (duplicates > 1).any():
        raise ValueError("Conflicting values for the same published vintage")
    d = d.sort_values("published").drop_duplicates("observation", keep="last")
    return pd.Series(
        d.number.to_numpy(), index=pd.PeriodIndex(d.observation, freq=frequency), name="value"
    ).sort_index()


def transform_draws(draws, history, target_period, *, metric, unit_scale=1.0):
    """Transform a distribution of levels; never substitute a point estimate for it.

    The caller must choose the matching SA/NSA series and vintage. A monthly YoY
    uses the same month one year earlier; it is not a monthly change times twelve.
    GDP annualization compounds a quarter-to-quarter ratio to the fourth power.
    """
    x = np.asarray(draws, float)
    if metric not in ("level", "count", "period_pct", "year_pct", "annualized_quarter_pct"):
        raise ValueError("Unsupported release unit; do not guess")
    if x.size == 0 or not np.isfinite(x).all() or not np.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError("Finite draws and a positive explicit unit scale are required")
    if metric in ("level", "count"):
        if metric == "count" and (x < 0).any():
            raise ValueError("Count draws cannot be negative")
        result = x * unit_scale
    else:
        if not isinstance(history.index, pd.PeriodIndex) or not isinstance(
            target_period, pd.Period
        ):
            raise ValueError("Growth transforms require explicit observation periods")
        if history.index.has_duplicates or history.index.freq != target_period.freq:
            raise ValueError("Ambiguous or mismatched observation periods")
        freq = target_period.freqstr
        lag, power = 1, 1
        if metric == "annualized_quarter_pct":
            if not freq.startswith("Q"):
                raise ValueError("GDP quarterly annualization requires quarterly periods")
            power = 4
        elif metric == "year_pct":
            if freq.startswith("M"):
                lag = 12
            elif freq.startswith("Q"):
                lag = 4
            else:
                raise ValueError("Year changes require monthly or quarterly periods")
        elif metric != "period_pct":
            raise ValueError("Unsupported release unit; do not guess")
        denominator = history.get(target_period - lag, np.nan)
        if not np.isfinite(denominator) or denominator <= 0 or (x <= 0).any():
            raise ValueError("A positive, released matching comparison period is required")
        result = 100 * np.expm1(power * (np.log(x) - np.log(denominator))) * unit_scale
    if not np.isfinite(result).all():
        raise ValueError("Release conversion overflowed")
    return result
