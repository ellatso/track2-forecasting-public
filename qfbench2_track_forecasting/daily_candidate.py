"""## Executive summary (read this first)

Apply the frozen daily policy to this card's own dated panels. Short-rate forecasts
retain V9. Short FX forecasts mix V9 with an EWMA20 Student-t distribution. A
missing-middle FX history can adapt its scale from other current FX series in this
same input, including long horizons with Student-t8 paths. Preserve authored
horizons and fall back on unsupported card types.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import cli
from .short_horizon import ew_cov, predict

FX = frozenset(
    ("AUD", "CAD", "CHF", "DKK", "EUR", "GBP", "JPY", "NOK", "NZD", "SEK", "CNY", "BRL", "INR")
)
SHORT_FX = {
    "id": "fx_guard75",
    "components": [
        {"vol": "v9", "drift": 0.5, "weight": 0.25},
        {"vol": "ew20", "drift": 0.0, "df": 8, "weight": 0.75},
    ],
}
LONG = {"id": "garch_t8", "vol": "garch", "drift": 0.0, "df": 8}


def estimate(panels, assets, horizons, asof, draws, seed):
    """Return None for domains left to V9; never read another card or a target value."""
    fx = all(a in FX for a in assets)
    rates = all(a.startswith("UST_") for a in assets)
    if not (fx or rates):
        return None
    if rates and max(horizons) <= 12:
        return None
    hist = {a: cli._series(panels, a, asof).iloc[-300:] for a in assets}
    if any(s.index.has_duplicates or not np.isfinite(s.to_numpy()).all() for s in hist.values()):
        raise ValueError("Ambiguous or nonfinite daily history")
    changes = pd.DataFrame({a: cli._diff_without_gaps(s) for a, s in hist.items()}).dropna()
    if len(changes) < 30:
        return None
    anchor = np.array([hist[a].iloc[-1] for a in assets])
    order = np.argsort(horizons)
    sorted_h = np.asarray(horizons)[order]
    last_step = pd.Timestamp(changes.index[-1])
    stale = (pd.Timestamp(asof) - last_step).days > 45
    ledger = {
        "method": "GARCH t8" if max(horizons) > 12 else "25% V9 + 75% EWMA20 t8",
        "rows": len(changes),
        "history_end": str(last_step.date()),
        "stale": stale,
        "anchor": {a: float(anchor[j]) for j, a in enumerate(assets)},
    }
    if stale:
        if not fx:
            return None
        if any((s <= 0).any() for s in hist.values()):
            return None
        logs = pd.DataFrame(
            {a: cli._diff_without_gaps(np.log(s)) for a, s in hist.items()}
        ).dropna()
        logchanges = logs.to_numpy() * anchor
        ratios = []
        used = []
        old_start = pd.Timestamp(logs.index[0])
        old_end = pd.Timestamp(logs.index[-1])
        available = set()
        for frame in panels.values():
            col = cli._asset_col(frame)
            if col:
                available.update(map(str, frame[col].unique()))
        for asset in sorted((available & FX) - set(assets)):
            s = cli._series(panels, asset, asof)
            if s.index.has_duplicates or (s <= 0).any():
                continue
            if (pd.Timestamp(asof) - pd.Timestamp(s.index[-1])).days > 7:
                continue
            r = cli._diff_without_gaps(np.log(s))
            dates = pd.to_datetime(r.index)
            old = r[(dates >= old_start) & (dates <= old_end)].dropna()
            recent = r.iloc[-299:].dropna()
            if len(old) < 60 or len(recent) < 60:
                continue
            if (pd.Timestamp(asof) - pd.Timestamp(recent.index[-1])).days > 7:
                continue
            old_var = float(ew_cov(old.to_numpy()[:, None], 20)[0, 0])
            current_var = float(ew_cov(recent.to_numpy()[:, None], 20)[0, 0])
            if old_var > 1e-12 and np.isfinite(current_var):
                ratios.append(current_var / old_var)
                used.append(asset)
        ratio = float(np.clip(np.median(ratios), 0.25, 4.0)) if len(ratios) >= 2 else 1.0
        cfg = {"vol": "v9", "drift": 0.0, "width": float(np.sqrt(0.5 + 0.5 * ratio))}
        if max(horizons) > 12:
            cfg["df"] = 8
        samples = predict(logchanges, anchor, sorted_h, cfg, draws, seed)
        ledger.update(
            method="gap-safe FX rescaling with 50% context adaptation"
            + (", t8" if max(horizons) > 12 else ""),
            context_assets=used,
            context_variance_ratio=ratio,
            configuration=cfg,
        )
    else:
        cfg = SHORT_FX if max(horizons) <= 12 else LONG
        samples = predict(changes.to_numpy(), anchor, sorted_h, cfg, draws, seed)
        ledger["configuration"] = cfg
    result = np.empty_like(samples)
    result[:, :, order] = samples
    return result, ledger
