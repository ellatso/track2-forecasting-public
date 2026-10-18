"""## Executive summary (read this first)

Evaluate the deployed numeric function, including its domain dispatch and fallback.
Optional rank coupling changes dependence only: every marginal draw is preserved.
This is research, not a change to the submitted forecaster or a House quality test.
"""

from pathlib import Path

import numpy as np

from qfbench2_track_forecasting.candidate import numeric_forecast
from qfbench2_track_forecasting.short_horizon import predict


def forecast(history, horizons, draws, seed, *, coupling=None, width=1.0, online_width=False):
    """Use only this origin's prefix, through the actual production dispatch."""
    panel = (
        history.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="asset", value_name="value")
    )
    card = dict(
        targets=dict(
            asset_ids=list(history),
            horizons=list(horizons),
            target_type="level",
            target_frequency="daily",
        )
    )
    samples, *_ = numeric_forecast(
        {"history": panel},
        card,
        Path("card.toml"),
        str(history.index[-1].date()),
        draws,
        seed,
    )
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Positive finite research width required")
    if online_width:
        width = prequential_width(history)[None, :, None]
    if online_width or width != 1.0:
        centre = samples.mean(axis=0, keepdims=True)
        samples = centre + width * (samples - centre)
    if coupling is None:
        return samples
    if coupling not in ("ew60", "shrink50"):
        raise ValueError("Unknown research coupling")
    # Mirror the production prefix/gap filter when constructing the rank template.
    import pandas as pd

    from qfbench2_track_forecasting.cli import _diff_without_gaps

    changes = (
        pd.DataFrame({a: _diff_without_gaps(history[a].iloc[-300:]) for a in history})
        .dropna()
        .to_numpy()
    )
    cfg = dict(vol="sample", drift=0.0)
    if coupling == "ew60":
        cfg["corr"] = 60
    else:
        cfg["shrink"] = 0.5
    template = predict(changes, np.zeros(len(history.columns)), horizons, cfg, draws, seed)
    return couple(samples, template)


def prequential_width(history):
    """Use 60 past one-step-ahead errors; each variance predates its own outcome.

    Restrict this experiment to fresh daily FX/Treasury levels. The mean and
    variance below reproduce the corresponding deployed one-step marginals.
    """
    import pandas as pd

    from qfbench2_track_forecasting.cli import _diff_without_gaps
    from qfbench2_track_forecasting.daily_candidate import FX

    x = pd.DataFrame({a: _diff_without_gaps(history[a]) for a in history}).dropna().to_numpy()
    fx = all(a in FX for a in history)
    if not fx and not all(a in FX or a.startswith("UST_") for a in history):
        raise ValueError("Online calibration only supports daily FX/Treasury levels")
    errors = []
    for i in range(max(30, len(x) - 60), len(x)):
        past = x[max(0, i - 299) : i]
        mean = past.mean(0)
        v9 = 0.5 * past.var(0, ddof=1)
        w = np.exp2(-np.arange(len(past) - 1, -1, -1) / 60.0)
        w /= w.sum()
        v9 += 0.5 * (w[:, None] * (past - w @ past) ** 2).sum(0)
        variance, centre = v9, 0.5 * mean
        if fx:
            w = np.exp2(-np.arange(len(past) - 1, -1, -1) / 20.0)
            w /= w.sum()
            ew20 = (w[:, None] * (past - w @ past) ** 2).sum(0) / (1 - w @ w)
            variance = 0.25 * v9 + 0.75 * ew20 + 0.25 * 0.75 * (0.5 * mean) ** 2
            centre = 0.125 * mean
        errors.append((x[i] - centre) ** 2 / np.maximum(variance, 1e-20))
    if len(errors) < 30:
        return np.ones(len(history.columns))
    return np.clip(np.sqrt(np.mean(errors, axis=0)), 0.8, 1.2)


def couple(samples, template):
    """Reorder each cell by a path template without changing its empirical distribution."""
    samples, template = np.asarray(samples), np.asarray(template)
    if samples.shape != template.shape or samples.ndim != 3:
        raise ValueError("Matching draw x asset x horizon arrays required")
    if not np.isfinite(samples).all() or not np.isfinite(template).all():
        raise ValueError("Finite arrays required")
    order = np.argsort(template, axis=0, kind="stable")
    result = np.empty_like(samples)
    np.put_along_axis(result, order, np.sort(samples, axis=0), axis=0)
    return result
