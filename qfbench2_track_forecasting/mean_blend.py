"""## Executive summary (read this first)

Blend half the existing daily mean with a 120-observation autoregressive mean.
Reuse production noise unchanged. Skip series with gaps or insufficient history.
Monthly forecasts never call this helper. No fitted parameters are packaged.
"""

import numpy as np
import pandas as pd

from . import cli


def apply_daily_mean_blend(samples, panels, assets, horizons, asof, stats, target_type):
    """Fit only past observations, then translate each marginal without changing noise."""
    grid = np.asarray(horizons, dtype=float)
    returns = target_type == "log_return"
    result = samples.copy()
    applied = []
    for ai, asset in enumerate(assets):
        history = cli._series(panels, asset, asof).iloc[-120:]
        if len(history) < 30:
            continue
        dates = pd.DatetimeIndex(pd.to_datetime(history.index))
        gaps = np.diff(dates.to_numpy()).astype("timedelta64[D]").astype(float)
        if dates.has_duplicates or not np.isfinite(gaps).all() or np.any(gaps <= 0):
            continue
        if np.any(gaps > max(10 * float(np.median(gaps)), 5)):
            continue
        values = history.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (returns and np.any(values <= -1)):
            continue
        values = np.log1p(values) if returns else values
        x, y = values[:-1], values[1:]
        xc, yc = x - x.mean(), y - y.mean()
        phi = float(np.dot(xc, yc) / max(float(np.dot(xc, xc)), 1e-12))
        phi = float(np.clip(phi, -0.5, 0.5) if returns else np.clip(phi, 0, 0.995))
        power = phi**grid
        rate = stats["daily_drift"][asset]
        anchor = stats["last"][asset]
        old = anchor + rate * grid
        if returns:
            proposed = rate * grid + (values[-1] - rate) * phi * (1 - power) / (1 - phi)
        else:
            equilibrium = (y.mean() - phi * x.mean()) / (1 - phi)
            proposed = equilibrium + (anchor - equilibrium) * power
        result[:, ai, :] += 0.5 * (proposed - old)
        applied.append(asset)
    return result, applied
