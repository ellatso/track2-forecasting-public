"""## Executive summary (read this first)

Compare an unchanged baseline, a fixed variance mixture, and a mixture chosen
using only completed forecasts inside each case's pre-origin history. All outer
periods are historical diagnostics already explored in earlier rounds. Monthly
forecasts stay unchanged. No official submission is produced or claimed improved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from qfbench2_common.scoring import crps

from . import batch
from . import frequency as f
from . import joint as j

ROOT = batch.ROOT
BASE = batch.BASE
FIXED = "fixed_mix50_corr25"
ADAPTIVE = "adaptive_mix_corr25"
WEIGHTS = (0.0, 0.125, 0.25, 0.375, 0.5)
INNER_DRAWS = 256
INNER_SEED = 71
ORIGINS = (
    *f.SELECTION,
    "2018-04-02",
    "2019-04-01",
    "2020-01-02",
    "2021-01-04",
    "2022-01-03",
    "2023-01-03",
)
BOUNDS = {o: (o[:4] + "-01-01", str(int(o[:4]) + 1) + "-01-01") for o in ORIGINS}


def configs():
    base = dict(id=BASE, window=300, drift=1.0, mix=0.0, shrink=0.0, power=1.0)
    return [base, dict(base, id=FIXED, mix=0.5, shrink=0.25), dict(base, id=ADAPTIVE, shrink=0.25)]


def grid_for(steps, assets):
    grid = np.asarray(steps, dtype=int)
    if grid.ndim == 1:
        grid = np.tile(grid, (assets, 1))
    if grid.shape[0] != assets or np.any(grid < 1):
        raise ValueError("Invalid observation grid")
    return grid


def inner_outcome(history, origin_index, grid, returns):
    values = np.asarray(history, dtype=float)
    if origin_index + int(grid.max()) >= len(values):
        raise ValueError("Inner target enters unavailable future")
    result = np.empty(grid.shape)
    for asset, column in np.ndindex(grid.shape):
        span = int(grid[asset, column])
        if returns:
            future = values[origin_index + 1 : origin_index + span + 1, asset]
            if np.any(future <= -1):
                raise ValueError("Invalid historical return")
            result[asset, column] = np.log1p(future).sum()
        else:
            result[asset, column] = values[origin_index + span, asset]
    return result.ravel()


@lru_cache(maxsize=64)
def _choose_cached(history_bytes, shape, flat_steps, step_shape, returns):
    # Cache pure numerical inputs, not unit handles or another card's state.
    history = np.frombuffer(history_bytes, dtype=np.float64).reshape(shape)
    grid = np.asarray(flat_steps, dtype=int).reshape(step_shape)
    longest = int(grid.max())
    origins = [len(history) - 1 - longest - 21 * k for k in range(6)]
    origins = [t for t in origins if t >= 299]
    fallback = dict(
        mix=0.0,
        inner_cases=len(origins),
        inner_latest_target_index=None,
        inner_last_history_index=len(history) - 1,
        inner_losses={},
        fallback=True,
    )
    if len(origins) < 3:
        return fallback
    losses = {weight: [] for weight in WEIGHTS}
    for origin in origins:
        prefix = history[: origin + 1]
        outcome = inner_outcome(history, origin, grid, returns)
        recent = prefix[-300:]
        increments = np.log1p(recent) if returns else np.diff(recent, axis=0)
        scale = (np.maximum(increments.std(0, ddof=1), 1e-10)[:, None] * np.sqrt(grid)).ravel()
        for weight in WEIGHTS:
            cfg = dict(configs()[1], id="inner", mix=weight)
            samples = j.predict(prefix, grid, returns, "T2-F1", False, cfg, INNER_DRAWS, INNER_SEED)
            normalized = [
                crps.crps_marginal(samples[:, cell : cell + 1], outcome[cell : cell + 1])
                / scale[cell]
                for cell in range(len(outcome))
            ]
            losses[weight].append(float(np.mean(normalized)))
    means = {weight: float(np.mean(scores)) for weight, scores in losses.items()}
    if not all(np.isfinite(v) for v in means.values()):
        raise ValueError("Nonfinite inner loss")
    best = min(WEIGHTS, key=lambda weight: (means[weight], weight))
    # Prefer the smaller weight on effectively tied numerical scores.
    best = next(weight for weight in WEIGHTS if means[weight] <= means[best] + 1e-8)
    return dict(
        mix=best,
        inner_cases=len(origins),
        inner_latest_target_index=max(origins) + longest,
        inner_last_history_index=len(history) - 1,
        inner_losses=means,
        fallback=False,
    )


def choose_mix(history, steps, returns):
    history = np.ascontiguousarray(history, dtype=np.float64)
    grid = grid_for(steps, history.shape[1])
    return _choose_cached(
        history.tobytes(), history.shape, tuple(grid.ravel()), grid.shape, returns
    )


def predict(history, steps, returns, family, monthly, cfg, draws, seed):
    if monthly:
        raise ValueError("Adaptive round compares daily models only")
    history = np.ascontiguousarray(history, dtype=np.float64)
    chosen = dict(cfg)
    if cfg["id"] == ADAPTIVE:
        chosen["mix"] = choose_mix(history, steps, returns)["mix"]
    return j.predict(history, steps, returns, family, False, chosen, draws, seed)


def diagnostics(history, steps, returns, cfg):
    result = dict(
        selected_mix=cfg["mix"],
        inner_cases=0,
        inner_latest_target_index=None,
        inner_last_history_index=len(history) - 1,
        inner_losses="",
        inner_fallback=False,
    )
    if cfg["id"] == ADAPTIVE:
        chosen = choose_mix(history, steps, returns)
        result.update(
            selected_mix=chosen["mix"],
            inner_cases=chosen["inner_cases"],
            inner_latest_target_index=chosen["inner_latest_target_index"],
            inner_losses=json.dumps(chosen["inner_losses"], sort_keys=True),
            inner_fallback=chosen["fallback"],
        )
    result["history_sha256"] = hashlib.sha256(
        np.asarray(history, dtype=np.float64).tobytes()
    ).hexdigest()
    return result


def validate_history(item, actual):
    # Inner forecasts need a longer uninterrupted prefix than the outer 300-row walk.
    frame = item["frame"].loc[:actual]
    needed = 300 + int(np.max(item["steps"])) + 21 * 5
    dates = frame.index[-needed:].to_numpy()
    gaps = np.diff(dates).astype("timedelta64[D]").astype(int)
    if len(gaps) and gaps.max() > 10:
        raise ValueError("Inner training window spans a historical gap")


def run_cases(items, seeds, draws):
    return f.run_cases(
        items,
        ORIGINS,
        configs(),
        seeds,
        draws,
        "holdout",
        predictor=predict,
        baseline_cfg=configs()[0],
        validation_bounds=BOUNDS,
        prediction_diagnostics=diagnostics,
        history_validator=validate_history,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 17, 41])
    args = parser.parse_args(argv)
    out = args.run_dir.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error("Results must stay outside the public repository")
    if not 200 <= args.draws <= 20000 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use 200-20000 draws and nonempty unique seeds")
    if out.exists() and any(out.iterdir()):
        parser.error("An existing diagnostic run cannot be overwritten")
    items, skipped, inputs = batch.inventory(ROOT)
    items = [i for i in items if not i["monthly"]]
    code = {
        str(p.relative_to(ROOT)): batch.sha(p)
        for p in [
            Path(__file__),
            Path(f.__file__),
            Path(j.__file__),
            Path(batch.__file__),
            *sorted((ROOT / "qfbench2_track_forecasting").glob("*.py")),
        ]
    }
    code["shared_crps"] = batch.sha(Path(crps.__file__))
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(
        created_utc=datetime.now(UTC).isoformat(),
        configs=configs(),
        origins=ORIGINS,
        validation_bounds=BOUNDS,
        seeds=args.seeds,
        draws=args.draws,
        inner_weights=WEIGHTS,
        inner_draws=INNER_DRAWS,
        inner_seed=INNER_SEED,
        inner_spacing=21,
        inner_max_cases=6,
        code_sha256=code,
        input_sha256=inputs,
        environment=batch.audit(ROOT),
        rankable=False,
        monthly_policy="Unchanged baseline; monthly forecasts not evaluated here",
        warning=(
            "All outer dates are previously explored historical diagnostics. "
            "No independent holdout. No House calls."
        ),
    )
    f.write_json(out / "plan.json", plan)
    frame, cells, omitted = run_cases(items, args.seeds, args.draws)
    f.write_json(out / "coverage.json", dict(inventory=skipped, exclusions=omitted))
    if frame.empty:
        raise SystemExit("No eligible cases; inspect coverage report")
    frame.to_csv(out / "diagnostic_cases.csv", index=False)
    cells.to_csv(out / "diagnostic_cells.csv", index=False)
    batch.save_diagnostics(frame, out, "diagnostic")
    summary = batch.summarize(frame)
    summary.to_csv(out / "diagnostic_summary.csv", index=False)
    period = []
    for year, subset in frame.groupby(frame.origin.str[:4]):
        table = batch.summarize(subset)
        table.insert(0, "year", year)
        period.append(table)
    pd.concat(period, ignore_index=True).to_csv(out / "diagnostic_by_year.csv", index=False)
    adaptive_rows = frame[frame.config == ADAPTIVE].drop_duplicates(["group", "origin"])
    adaptive_rows[
        [
            "group",
            "cluster",
            "source",
            "origin",
            "selected_mix",
            "inner_cases",
            "inner_latest_target_index",
            "inner_last_history_index",
            "inner_losses",
            "inner_fallback",
            "history_sha256",
        ]
    ].to_csv(out / "weight_audit.csv", index=False)
    intervals = {name: batch.paired_interval(frame, name) for name in [FIXED, ADAPTIVE]}
    f.write_json(out / "paired_diagnostics.json", intervals)
    print(summary.to_string(index=False), flush=True)
    print("Diagnostic only; no independent holdout, winner promotion or submission.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
