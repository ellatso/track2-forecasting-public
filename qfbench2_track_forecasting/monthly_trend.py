"""## Executive summary (read this first)

Blend monthly baseline means equally with a recent 12-observation damped trend.
Keep covariance and paths unchanged. Fit only consecutive pre-cutoff observations.
Authored horizon keys stay unchanged; adjustment uses explicit monthly steps.
"""

import numpy as np

from . import cli


def adjust_monthly_mean(samples, panels, assets, steps, asof, stats, strength=0.5):
    if not 0 <= strength <= 1:
        raise ValueError("Invalid monthly trend strength")
    result = samples.copy()
    records = {}
    damping = np.exp2(-1 / 12)
    for ai, asset in enumerate(assets):
        history = cli._monthly_series(cli._series(panels, asset, asof)).iloc[-12:]
        if len(history) < 12 or np.any(np.diff(history.index.asi8) != 1):
            continue
        values = history.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            continue
        time = np.arange(12, dtype=float)
        time -= time.mean()
        slope = float(np.dot(time, values) / np.dot(time, time))
        span = np.asarray(steps[ai], dtype=float)
        cumulative = damping * (1 - damping**span) / (1 - damping)
        adjustment = strength * (slope * cumulative - stats["step_drift"][asset] * span)
        result[:, ai, :] += adjustment
        records[asset] = dict(
            slope=slope,
            strength=strength,
            monthly_steps=span.astype(int).tolist(),
            mean_adjustment=adjustment.tolist(),
            history_start=str(history.index[0]),
            history_end=str(history.index[-1]),
        )
    return result, records
