"""## Executive summary (read this first)

Forecast cumulative factor log returns with conditional variance and t8 paths.
Keep the half-strength drift and 300-row window. Read only this task's dated
panels; never use a card identifier or an outcome to choose the distribution.
"""

import numpy as np
import pandas as pd

from . import cli
from .short_horizon import predict
from .targets import log_return_steps

POLICY = {"vol": "garch", "df": 8, "drift": 0.5, "window": 300}


def estimate(panels, assets, horizons, asof, draws, seed):
    """Return the forecast and its transparent, input-only configuration ledger."""
    hist = {asset: cli._series(panels, asset, asof).iloc[-300:] for asset in assets}
    if any(s.index.has_duplicates or not np.isfinite(s.to_numpy()).all() for s in hist.values()):
        raise ValueError("Ambiguous or nonfinite return history")
    steps = pd.DataFrame(
        {a: pd.Series(log_return_steps(s), index=s.index) for a, s in hist.items()}
    ).dropna()
    order = np.argsort(horizons)
    samples = predict(
        steps.to_numpy(), np.zeros(len(assets)), np.asarray(horizons)[order], POLICY, draws, seed
    )
    result = np.empty_like(samples)
    result[:, :, order] = samples
    ledger = {
        "method": "factor half drift, GARCH t8",
        "rows": len(steps),
        "history_end": str(steps.index[-1]),
        "configuration": POLICY,
        "step_transform": "log1p(simple return)",
        "anchor": 0.0,
    }
    return result, ledger
