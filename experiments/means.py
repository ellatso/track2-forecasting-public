"""## Executive summary (read this first)

Compare mean reversion and damped trends with the current submission means.
Change only conditional means: reuse identical joint Gaussian noise for every arm.
Evaluate daily and monthly separately on explored historical snapshots. Monthly
step counts retain the original explicit mapping; historical release vintages are
unavailable, so monthly results are observation-index diagnostics, not causal OOS.
Import shared organizer scoring. Save results outside GitHub; do not auto-submit.
"""

from __future__ import annotations

import argparse
import itertools
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from qfbench2_common.scoring import crps

from qfbench2_track_forecasting import cli

from . import adaptive as a
from . import batch
from . import distributions as d
from . import frequency as f
from . import joint as j

BASE = "current_reference"
ORIGINS = d.ORIGINS
BOUNDS = d.BOUNDS


def configs(monthly):
    rows = [dict(id=BASE, model="baseline"), dict(id="zero_drift", model="zero")]
    ar_windows = (60, 120) if monthly else (120, 300)
    for window, strength in itertools.product(ar_windows, (0.25, 0.5, 1.0)):
        rows.append(
            dict(id=f"ar{window}_blend{strength:g}", model="ar", window=window, strength=strength)
        )
    windows = (12, 24) if monthly else (60, 120)
    half_lives = (3, 6, 12) if monthly else (20, 60, 120)
    for window, half_life in itertools.product(windows, half_lives):
        rows.append(
            dict(
                id=f"trend{window}_half{half_life}",
                model="trend",
                window=window,
                half_life=half_life,
                strength=1.0,
            )
        )
    for half_life in half_lives:
        rows.append(
            dict(
                id=f"trend{windows[0]}_half{half_life}_blend50",
                model="trend",
                window=windows[0],
                half_life=half_life,
                strength=0.5,
            )
        )
    return rows


def current_config(monthly):
    if monthly:
        return dict(
            id="monthly_reference",
            window=300,
            drift=1.0,
            spread=1.0,
            policy="fixed",
            volatility="sample",
        )
    return dict(id="daily_reference", window=300, drift=0.5, mix=0.5, shrink=0.0, power=1.0)


def baseline_mean(history, steps, returns, monthly):
    values = np.asarray(history[-300:], dtype=float)
    increments = np.log1p(values) if returns else np.diff(values, axis=0)
    drift = increments.mean(0) * (1.0 if monthly else 0.5)
    grid = a.grid_for(steps, history.shape[1])
    anchor = np.zeros(history.shape[1]) if returns else history[-1]
    return anchor[:, None] + drift[:, None] * grid


def forecast_mean(history, steps, returns, monthly, cfg):
    history = np.ascontiguousarray(history, dtype=float)
    grid = a.grid_for(steps, history.shape[1])
    base = baseline_mean(history, steps, returns, monthly)
    anchor = np.zeros(history.shape[1]) if returns else history[-1]
    if cfg["model"] == "baseline":
        return base.ravel()
    if cfg["model"] == "zero":
        return np.broadcast_to(anchor[:, None], grid.shape).ravel()
    recent = np.asarray(history[-cfg["window"] :], dtype=float)
    if len(recent) < min(cfg["window"], 30):
        raise ValueError("Insufficient mean-model history")
    values = np.log1p(recent) if returns else recent
    if cfg["model"] == "ar":
        x, y = values[:-1], values[1:]
        xc, yc = x - x.mean(0), y - y.mean(0)
        phi = (xc * yc).sum(0) / np.maximum((xc * xc).sum(0), 1e-12)
        phi = np.clip(phi, -0.5, 0.5) if returns else np.clip(phi, 0, 0.995)
        power = phi[:, None] ** grid
        if returns:
            # Factor returns revert toward the current baseline mean return rate.
            rate = np.log1p(history[-300:]).mean(0) * (1 if monthly else 0.5)
            proposed = rate[:, None] * grid + (values[-1] - rate)[:, None] * phi[:, None] * (
                1 - power
            ) / (1 - phi[:, None])
        else:
            intercept = y.mean(0) - phi * x.mean(0)
            equilibrium = intercept / (1 - phi)
            proposed = equilibrium[:, None] + (anchor - equilibrium)[:, None] * power
    elif cfg["model"] == "trend":
        if returns:
            slope = values.mean(0)
        else:
            time = np.arange(len(values), dtype=float)
            time -= time.mean()
            slope = (time[:, None] * values).sum(0) / (time * time).sum()
        slope *= 1.0 if monthly else 0.5
        damping = np.exp2(-1 / cfg["half_life"])
        cumulative = damping * (1 - damping**grid) / (1 - damping)
        proposed = anchor[:, None] + slope[:, None] * cumulative
    else:
        raise ValueError("Unknown mean model")
    strength = cfg["strength"]
    if not 0 <= strength <= 1:
        raise ValueError("Invalid mean blend")
    return ((1 - strength) * base + strength * proposed).ravel()


def predict(history, steps, returns, family, monthly, cfg, draws, seed):
    history = np.ascontiguousarray(history, dtype=float)
    reference = current_config(monthly)
    if monthly:
        if returns:
            raise ValueError("Monthly targets must be levels")
        # The actual CLI draws noise once per monthly transition, including months
        # without a target. Reuse it exactly rather than changing the seed stream.
        grid = a.grid_for(steps, history.shape[1])
        assets = [f"asset{i}" for i in range(history.shape[1])]
        dates = pd.date_range("1900-01-01", periods=len(history), freq="MS")
        panel = (
            pd.DataFrame(history, index=dates, columns=assets)
            .rename_axis("date")
            .reset_index()
            .melt(id_vars="date", var_name="asset_id", value_name="value")
        )
        samples, _ = cli._draw(
            {"synthetic_dates": panel},
            assets,
            list(range(1, grid.shape[1] + 1)),
            str(dates[-1].date()),
            draws,
            seed,
            panel_steps=grid,
        )
        base = samples.reshape(draws, -1)
    else:
        base = j.predict(history, steps, returns, family, False, reference, draws, seed)
    if cfg["model"] == "baseline":
        return base
    old = baseline_mean(history, steps, returns, monthly).ravel()
    new = forecast_mean(history, steps, returns, monthly, cfg)
    result = base + (new - old)[None, :]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite mean forecast")
    return result


def save(frame, cells, out, frequency):
    prefix = frequency + "_"
    frame.to_csv(out / (prefix + "cases.csv"), index=False)
    cells.to_csv(out / (prefix + "cells.csv"), index=False)
    batch.summarize(frame).to_csv(out / (prefix + "summary.csv"), index=False)
    batch.save_diagnostics(frame, out, frequency)
    years = []
    for year, group in frame.groupby(frame.origin.str[:4]):
        table = batch.summarize(group)
        table.insert(0, "year", year)
        years.append(table)
    pd.concat(years).to_csv(out / (prefix + "by_year.csv"), index=False)
    frame.groupby(["config", "cluster"]).composite.mean().unstack(0).to_csv(
        out / (prefix + "by_basket.csv")
    )
    cells["crps_ratio"] = cells.marginal_crps / cells.baseline_crps.replace(0, np.nan)
    cells.groupby(["config", "horizon", "observation_steps"]).crps_ratio.mean().to_csv(
        out / (prefix + "by_horizon.csv")
    )
    frame.groupby("config")[
        ["marginal_ratio", "joint_ratio", "tail_ratio", "coverage90"]
    ].mean().to_csv(out / (prefix + "components.csv"))
    compatible = frame.copy()
    compatible.loc[compatible.config == BASE, "config"] = batch.BASE
    f.write_json(
        out / (prefix + "paired.json"),
        {c: batch.paired_interval(compatible, c) for c in frame.config.unique() if c != BASE},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 17, 41])
    parser.add_argument("--quick-check", action="store_true")
    args = parser.parse_args(argv)
    out = args.run_dir.resolve()
    if out == batch.ROOT or batch.ROOT in out.parents:
        parser.error("Keep results outside the public repository")
    if (
        not 200 <= args.draws <= 20000
        or not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or min(args.seeds) < 0
    ):
        parser.error("Use 200-20000 draws and unique nonnegative seeds")
    if out.exists() and any(out.iterdir()):
        parser.error("Use a fresh run directory")
    items, inventory, inputs = batch.inventory(batch.ROOT)
    seeds, draws = ([0], 200) if args.quick_check else (args.seeds, args.draws)
    origins = ORIGINS[:1] if args.quick_check else ORIGINS
    paths = [
        Path(__file__),
        Path(a.__file__),
        Path(batch.__file__),
        Path(d.__file__),
        Path(f.__file__),
        Path(j.__file__),
        *sorted((batch.ROOT / "qfbench2_track_forecasting").glob("*.py")),
    ]
    code = {str(p.relative_to(batch.ROOT)): batch.sha(p) for p in paths}
    code["shared_crps"] = batch.sha(Path(crps.__file__))
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(
        created_utc=datetime.now(UTC).isoformat(),
        daily_configs=configs(False),
        monthly_configs=configs(True),
        seeds=seeds,
        draws=draws,
        origins=origins,
        bounds=BOUNDS,
        input_sha256=inputs,
        code_sha256=code,
        environment=batch.audit(batch.ROOT),
        quick_check=args.quick_check,
        rankable=False,
        promotion=False,
        warning=(
            "Explored snapshot diagnostics only. Monthly release vintages are unavailable. "
            "Volatility and joint noise remain fixed; only means change. "
            "No House calls or submission."
        ),
    )
    f.write_json(out / "plan.json", plan)
    coverage, decision = dict(inventory=inventory), {}
    for frequency, monthly in [("daily", False), ("monthly", True)]:
        selected = [i for i in items if i["monthly"] == monthly]
        frames, cell_frames, omitted = [], [], []
        for index, item in enumerate(selected):
            print(f"{frequency} {index+1}/{len(selected)}: {item['source']}", flush=True)
            frame, cells, excluded = f.run_cases(
                [item],
                origins,
                configs(monthly),
                seeds,
                draws,
                "holdout",
                predictor=predict,
                baseline_cfg=configs(monthly)[0],
                validation_bounds=BOUNDS,
            )
            if not frame.empty:
                frames.append(frame)
                cell_frames.append(cells)
                frame.to_csv(out / f"checkpoint-{frequency}-{index:02d}.csv", index=False)
            omitted.extend(excluded)
        coverage[frequency] = omitted
        if not frames:
            raise ValueError(f"No eligible {frequency} cases")
        frame, cells = (
            pd.concat(frames, ignore_index=True),
            pd.concat(cell_frames, ignore_index=True),
        )
        if not (
            frame.groupby(["group", "origin", "seed"]).config.nunique() == len(configs(monthly))
        ).all():
            raise ValueError("Unequal candidate coverage")
        save(frame, cells, out, frequency)
        decision[frequency] = dict(
            winner=str(batch.summarize(frame).iloc[0].config),
            cases=len(frame[["group", "origin"]].drop_duplicates()),
            baskets=frame.cluster.nunique(),
            methods=len(configs(monthly)),
            promotion=False,
        )
        print(batch.summarize(frame).to_string(index=False), flush=True)
    f.write_json(out / "coverage.json", coverage)
    if (
        any(batch.sha(batch.ROOT / p) != v for p, v in code.items() if p != "shared_crps")
        or any(batch.sha(batch.ROOT / p) != v for p, v in inputs.items())
        or batch.sha(Path(crps.__file__)) != code["shared_crps"]
    ):
        raise ValueError("Source or inputs changed during run")
    f.write_json(out / "decision.json", decision)
    print("Complete; only historical diagnostics, no submission change.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
