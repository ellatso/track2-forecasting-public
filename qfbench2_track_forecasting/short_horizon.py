"""## Executive summary (read this first)

Forecast daily level targets from this input's dated history. Shrink means,
estimate conditional covariance, and sample whole correlated paths. Configuration
is explicit so historical comparisons and the container use the same function.
No outcomes, unit identities, dates or fitted offline weights select a model.
"""

from functools import lru_cache

import numpy as np
from scipy.special import ndtri
from scipy.stats import chi2, qmc


@lru_cache(maxsize=128)
def innovations(draws, nassets, nhorizons, seed, df):
    u = qmc.Sobol(nassets * nhorizons + 2, scramble=True, seed=seed).random_base2(
        int(np.log2(draws))
    )
    u = np.clip(u, 1e-12, 1 - 1e-12)
    z = ndtri(u[:, : nassets * nhorizons]).reshape(draws, nhorizons, nassets)
    scale = np.sqrt((df - 2) / chi2.ppf(u[:, -2], df)) if df else np.ones(draws)
    return z, scale, u[:, -1]


def ew_cov(x, half_life):
    w = np.exp2(-np.arange(len(x) - 1, -1, -1) / half_life)
    w /= w.sum()
    xc = x - w @ x
    return (xc * w[:, None]).T @ xc / max(1 - float(w @ w), 1e-6)


def positive_corr(cov, shrink=0.0):
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-20))
    corr = np.nan_to_num(cov / np.outer(sd, sd), nan=0.0)
    corr = (1 - shrink) * corr + shrink * np.eye(len(sd))
    w, v = np.linalg.eigh(corr)
    corr = (v * np.maximum(w, 1e-8)) @ v.T
    d = np.sqrt(np.diag(corr))
    return corr / np.outer(d, d)


def predict(changes, anchor, horizons, cfg, draws=4096, seed=0, log_mask=None):
    """Return draws x assets x horizons, from finite, date-aligned increments."""
    if "components" in cfg:
        weights = np.asarray([part["weight"] for part in cfg["components"]], float)
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Invalid mixture weights")
        _, _, selector = innovations(draws, len(anchor), len(horizons), seed + 7927, 0)
        indices = np.searchsorted(np.cumsum(weights), selector, side="right")
        out = None
        for k, part in enumerate(cfg["components"]):
            member = predict(changes, anchor, horizons, part, draws, seed, log_mask)
            if out is None:
                out = np.empty_like(member)
            out[indices == k] = member[indices == k]
        return out
    x = np.ascontiguousarray(changes[-cfg.get("window", 299) :], dtype=float)
    anchor = np.asarray(anchor, float)
    horizons = np.asarray(horizons, int)
    if (
        len(x) < 30
        or not np.isfinite(x).all()
        or np.any(horizons <= 0)
        or np.any(np.diff(horizons) <= 0)
    ):
        raise ValueError("Need finite history and increasing positive horizons")
    if draws < 200 or draws & (draws - 1):
        raise ValueError("Draw count must be a power of two, at least 256")
    sample = np.atleast_2d(np.cov(x, rowvar=False))
    kind = cfg.get("vol", "blend60")
    if kind == "v9":
        w = np.exp2(-np.arange(len(x) - 1, -1, -1) / 60.0)
        w /= w.sum()
        centered = x - w @ x
        cov = 0.5 * sample + 0.5 * ((centered * w[:, None]).T @ centered)
    elif kind == "sample":
        cov = sample
    elif kind.startswith("ew"):
        cov = ew_cov(x, float(kind[2:]))
    elif kind.startswith("blend"):
        cov = 0.5 * sample + 0.5 * ew_cov(x, float(kind[5:]))
    elif kind == "har":
        cov = 0.2 * sample + 0.3 * ew_cov(x, 20) + 0.5 * ew_cov(x, 5)
    elif kind == "garch":
        v = np.maximum(np.diag(sample), 1e-20)
        state = v.copy()
        for row in x - x.mean(0):
            state = 0.02 * v + 0.06 * row**2 + 0.92 * state
        cov = sample.copy()
        sd = np.sqrt(np.maximum(state, 1e-20))
        cov = positive_corr(ew_cov(x, 60)) * np.outer(sd, sd)
    else:
        raise ValueError(kind)
    variance = np.maximum(np.diag(cov), 1e-20)
    corr_cov = sample if cfg.get("corr", "sample") == "sample" else ew_cov(x, float(cfg["corr"]))
    corr = positive_corr(corr_cov, cfg.get("shrink", 0.0))
    root = np.linalg.cholesky(corr)
    drift = x.mean(0) * cfg.get("drift", 0.0)
    means = horizons[:, None] * drift
    mean_kind = cfg.get("mean", "drift")
    if mean_kind == "ar":
        centered = x - x.mean(0)
        phi = np.sum(centered[1:] * centered[:-1], 0) / (
            np.sum(centered[:-1] ** 2, 0) + len(x) * np.diag(sample) * 0.1 + 1e-20
        )
        phi = np.clip(phi, -0.5, 0.5) * cfg.get("strength", 0.5)
        means += x[-1] * (phi * (1 - phi ** horizons[:, None]) / (1 - phi))
    elif mean_kind == "trend":
        drift2 = x[-20:].mean(0) * cfg.get("strength", 0.1)
        means += horizons[:, None] * np.clip(
            drift2, -0.1 * np.sqrt(variance), 0.1 * np.sqrt(variance)
        )
    z, scale, selector = innovations(draws, len(anchor), len(horizons), seed, cfg.get("df", 0))
    if cfg.get("df", 0) and cfg.get("tail_weight", 1.0) < 1:
        scale = np.where(selector < cfg["tail_weight"], scale, 1.0)
    cumulative = horizons[:, None] * variance
    if kind == "garch" or cfg.get("revert"):
        rho = 0.98 if kind == "garch" else np.exp2(-1.0 / cfg["revert"])
        long = np.maximum(np.diag(sample), 1e-20)
        cumulative = horizons[:, None] * long + (variance - long) * (
            1 - rho ** horizons[:, None]
        ) / (1 - rho)
    interval = np.diff(np.pad(cumulative, ((1, 0), (0, 0))), axis=0)
    noise = np.cumsum((z @ root.T) * np.sqrt(np.maximum(interval, 0))[None, :, :], axis=1)
    if cfg.get("vol_mixture"):
        noise *= np.where(selector < 0.8, 0.8, np.sqrt((1 - 0.8 * 0.8**2) / 0.2))[:, None, None]
    noise *= scale[:, None, None] * cfg.get("width", 1.0)
    out = anchor[None, None, :] + means[None, :, :] + noise
    if log_mask is not None and np.any(log_mask):
        out[:, :, log_mask] = np.exp(out[:, :, log_mask])
    return out.transpose(0, 2, 1)
