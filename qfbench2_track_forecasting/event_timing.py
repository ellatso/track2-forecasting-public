"""## Executive summary (read this first)

Activate a research scenario only on or after a cited future calendar date.
Dates must occur in an already verified quotation. The date alone does not prove
the causal interpretation or scenario probability. No external calendar is read.
"""

from __future__ import annotations

import datetime as dt
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .daily_candidate import FX

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def date_in_quote(date, quote):
    """Require the full year; do not infer an undated calendar from model memory."""
    if not isinstance(quote, str):
        return False
    d, y = date.day, date.year
    month = MONTHS[date.month - 1]
    m = rf"(?:{month}|{month[:3]}\.?)"
    patterns = [
        re.escape(date.isoformat()),
        rf"{m}\s+0?{d}(?:st|nd|rd|th)?\s*,?\s*{y}",
        rf"0?{d}(?:st|nd|rd|th)?\s+{m}\s*,?\s*{y}",
        rf"{m}\s+\d{{1,2}}\s*[-–—]\s*0?{d}(?:st|nd|rd|th)?\s*,?\s*{y}",
    ]
    return any(re.search(rf"(?<!\w){p}(?!\w)", quote, re.I) for p in patterns)


def activation_horizon(value, cited_evidence, asof, horizons):
    """Return a positive daily business step, or reject an unsupported event date."""
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Invalid event date")
    date = dt.date.fromisoformat(value)
    cutoff = dt.date.fromisoformat(asof)
    last = (pd.Timestamp(cutoff) + pd.offsets.BDay(max(horizons))).date()
    if not cutoff < date <= last:
        raise ValueError("Event is not inside the future forecast window")
    if not any(date_in_quote(date, e.get("quote")) for e in cited_evidence):
        raise ValueError("Event date is absent from verified quotations")
    effective = np.busday_offset(np.datetime64(date), 0, roll="forward")
    return int(np.busday_count(np.datetime64(cutoff) + 1, effective + 1))


def observation_horizon(spec, cited_evidence, asof, horizons, asset):
    """Use an explicitly selected ECB/UST source-clock profile, never infer a source.

    ECB uses 14:10 Europe/Berlin (13:10 UTC in the announced Final week).
    Treasury uses approximately 15:30 New York time. The event's clock and zone
    must occur with its date in a verified quote. Missing evidence rejects it.
    """
    activation_horizon(spec["activation_date"], cited_evidence, asof, horizons)
    clock, zone = spec.get("activation_time"), spec.get("activation_timezone")
    if not isinstance(clock, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", clock):
        raise ValueError("A quoted event time is required for a source-clock adjustment")
    aliases = {
        "UTC": ["UTC", "GMT"],
        "America/New_York": ["America/New_York", "Eastern Time"],
        "Europe/Berlin": ["Europe/Berlin", "Central European Time"],
    }
    if zone not in aliases:
        raise ValueError("Unsupported or unquoted event timezone")
    event = dt.datetime.fromisoformat(spec["activation_date"] + "T" + clock)
    tz = ZoneInfo(zone)
    if event.replace(tzinfo=tz, fold=0).utcoffset() != event.replace(tzinfo=tz, fold=1).utcoffset():
        raise ValueError("Ambiguous or nonexistent local event time")
    event = event.replace(tzinfo=tz)
    hour, minute = event.hour, event.minute
    times = [re.escape(clock)]
    suffix = "a" if hour < 12 else "p"
    times.append(rf"0?{hour % 12 or 12}:{minute:02d}\s*{suffix}\.?\s*m\.?")
    if minute == 0:
        times.append(rf"0?{hour % 12 or 12}\s*{suffix}\.?\s*m\.?")
    zones = aliases[zone] + [event.tzname()]
    supported = False
    for e in cited_evidence:
        quote = e.get("quote", "")
        if (
            date_in_quote(event.date(), quote)
            and any(re.search(rf"(?<!\w){p}(?!\w)", quote, re.I) for p in times)
            and any(re.search(rf"(?<!\w){re.escape(z)}(?!\w)", quote, re.I) for z in zones)
        ):
            supported = True
    if not supported:
        raise ValueError("The event date, clock and timezone are not jointly quoted")
    for step in range(1, max(horizons) + 1):
        date = (pd.Timestamp(asof) + pd.offsets.BDay(step)).date()
        if asset in FX:
            observation = dt.datetime.combine(date, dt.time(14, 10), ZoneInfo("Europe/Berlin"))
        elif asset.startswith("UST_"):
            observation = dt.datetime.combine(date, dt.time(15, 30), ZoneInfo("America/New_York"))
        else:
            raise ValueError("No observation-clock contract for this asset")
        if observation >= event:
            return step
    raise ValueError("Event happens after the final forecast observation")


def adjust_path(base, mean, multiplier, shift, horizons, start):
    """Keep earlier draws exact; scale only post-event increments of each joint path.

    When a reported interval straddles the event, divide its variance by business
    steps. This is a Brownian interpolation assumption, not recovered intraday data.
    Mean impact is a persistent level jump at the event, not a pre-event trend.
    """
    h = np.asarray(horizons, float)
    order = np.argsort(h)
    hh = h[order]
    previous = np.r_[0, hh[:-1]]
    fraction = np.clip((hh - np.maximum(previous, start - 1)) / (hh - previous), 0, 1)
    residual = (base - mean)[:, order]
    increments = np.diff(np.pad(residual, ((0, 0), (1, 0))), axis=1)
    scales = np.sqrt(1 + fraction * (np.asarray(multiplier)[order] ** 2 - 1))
    paths = mean[order] + np.cumsum(increments * scales, axis=1) + shift * (hh >= start)
    result = base.copy()
    affected = hh >= start
    result[:, order[affected]] = paths[:, affected]
    return result
