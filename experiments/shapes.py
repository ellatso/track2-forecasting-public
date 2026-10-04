"""## Executive summary (read this first)

Change only the uncertainty around the V9 mean. Compare widths, Student-t full
path scales, monotone variance clocks and whole-path mixtures. All estimates use
only the passed history. The forecast never receives outcomes or a practice ID.
"""

import numpy as np

from . import means, precision

_cache = {}


def configs(monthly):
    rows = [dict(id="v9")]
    rows += [dict(id=f"width{x:g}", width=x) for x in (0.7, 0.85, 0.95, 1.05, 1.15, 1.3)]
    rows += [dict(id=f"t{df}", df=df) for df in (3, 5, 8, 16)]
    rows += [dict(id=f"tailmix{x:g}", df=5, tail_weight=x) for x in (0.25, 0.5)]
    rows += [dict(id=f"t5_width{x:g}", df=5, width=x) for x in (0.85, 1.15)]
    if not monthly:
        rows += [dict(id=f"clock{x:g}", power=x) for x in (0.7, 0.85, 1.15, 1.3)]
        rows += [dict(id=f"t5_clock{x:g}", power=x, df=5) for x in (0.85, 1.15)]
        rows += [dict(id=f"volmix{x:g}", mix=x) for x in (0.0, 1.0)]
        rows += [dict(id=f"ewma{x}", half_life=x) for x in (20, 120)]
        rows += [dict(id=f"revert{x}", revert=x) for x in (21, 63)]
        rows += [
            dict(id="volmixture", vol_mixture=True),
            dict(id="clockmixture", clock_mixture=True),
        ]
    return rows


def v9(history, steps, returns, monthly, seed, prepared):
    key = (history[-300:].tobytes(), np.asarray(steps).tobytes(), returns, monthly, seed)
    if key not in _cache:
        _cache.clear()
        cfg = dict(draws=4096, sampling="sobol", drift=0.5, trend=0.5 if monthly else 0.0)
        samples = precision.predict(history, steps, returns, monthly, cfg, seed, prepared)
        if monthly:
            mean = means.forecast_mean(
                history,
                steps,
                returns,
                monthly,
                dict(model="trend", window=12, half_life=12, strength=0.5),
            )
        else:
            mean = means.baseline_mean(history, steps, returns, monthly).ravel()
        _cache[key] = (samples, mean)
    return _cache[key]


def ewvar(x, half_life):
    weights = np.exp2(-np.arange(len(x) - 1, -1, -1) / half_life)
    weights /= weights.sum()
    center = weights @ x
    return weights @ ((x - center) ** 2)


def clock_noise(noise, steps, cfg, sample_var, base_var):
    horizons = np.asarray(steps, dtype=float)
    if cfg.get("revert"):
        rho = np.exp2(-1.0 / cfg["revert"])
        cumulative = rho * (1 - rho**horizons) / (1 - rho)
        clock = (
            sample_var[:, None] * horizons + (base_var - sample_var)[:, None] * cumulative
        ) / base_var[:, None]
    else:
        clock = np.maximum(
            0.0,
            np.where(horizons <= 20.0, horizons, 20.0 * (horizons / 20.0) ** cfg.get("power", 1.0)),
        )
        clock = np.broadcast_to(clock, noise.shape[1:])
    interval = np.diff(np.r_[0.0, horizons])
    clock_interval = np.diff(np.pad(clock, ((0, 0), (1, 0))), axis=1)
    innovations = np.diff(np.pad(noise, ((0, 0), (0, 0), (1, 0))), axis=2)
    return np.cumsum(innovations * np.sqrt(clock_interval / interval), axis=2)


def predict(history, steps, returns, monthly, cfg, seed, prepared=None):
    samples, mean = v9(history, steps, returns, monthly, seed, prepared)
    if cfg["id"] == "v9":
        return samples
    shape = (len(samples), history.shape[1], np.asarray(steps).shape[-1])
    noise = (samples - mean).reshape(shape).copy()
    if not monthly:
        x = np.log1p(history[-300:]) if returns else np.diff(history[-300:], axis=0)
        sample_var = x.var(0, ddof=1)
        base_var = np.maximum(0.5 * sample_var + 0.5 * ewvar(x, 60.0), 1e-20)
        if cfg.get("power") or cfg.get("revert"):
            noise = clock_noise(noise, steps, cfg, sample_var, base_var)
        if "mix" in cfg or "half_life" in cfg:
            mix = cfg.get("mix", 0.5)
            var = (1 - mix) * sample_var + mix * ewvar(x, cfg.get("half_life", 60.0))
            noise *= np.sqrt(np.maximum(var, 1e-20) / base_var)[None, :, None]
        if cfg.get("vol_mixture"):
            choose = (
                np.random.default_rng(np.random.SeedSequence([seed, 10117])).random(len(noise))
                < 0.5
            )
            ratio = np.where(choose[:, None], sample_var, ewvar(x, 60.0)) / base_var
            noise *= np.sqrt(np.maximum(ratio, 1e-20))[:, :, None]
        if cfg.get("clock_mixture"):
            choose = (
                np.random.default_rng(np.random.SeedSequence([seed, 10117])).random(len(noise))
                < 0.5
            )
            lo = clock_noise(noise, steps, dict(power=0.85), sample_var, base_var)
            hi = clock_noise(noise, steps, dict(power=1.15), sample_var, base_var)
            noise = np.where(choose[:, None, None], lo, hi)
    if cfg.get("df"):
        df = cfg["df"]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 10118, df]))
        scale = np.sqrt((df - 2) / rng.chisquare(df, len(noise)))
        if "tail_weight" in cfg:
            chosen = np.random.default_rng(np.random.SeedSequence([seed, 10117])).random(len(noise))
            scale = np.where(chosen < cfg["tail_weight"], scale, 1.0)
        noise *= scale[:, None, None]
    proposed = mean + noise.reshape(samples.shape) * cfg.get("width", 1.0)
    if "ensemble_weight" in cfg:
        selector = np.random.default_rng(np.random.SeedSequence([seed, 10119])).random(len(noise))
        return np.where((selector < cfg["ensemble_weight"])[:, None], proposed, samples)
    return proposed
