"""## Executive summary (read this first)

Compare 17 daily distribution candidates against the submitted half-drift model.
Freeze drift, random seeds, draws and dates. Student-t means a heavy-tailed
variance-normalized Gaussian scale mixture; one scale is shared by each full path.
All dates, including January 2024, are explored diagnostics. Monthly stays unchanged.
Import organizer scoring. Keep results outside GitHub; never automatically submit.
"""

from __future__ import annotations

import argparse
import itertools
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from qfbench2_common.scoring import crps

from . import adaptive as a
from . import batch
from . import frequency as f
from . import joint as j

BASE = "current_halfdrift_mix50"
ORIGINS = (*a.ORIGINS, "2024-01-02")
BOUNDS = {o: (o[:4] + "-01-01", str(int(o[:4]) + 1) + "-01-01") for o in ORIGINS}


def component(weight=1.0, df=None, power=1.0, mix=0.5):
    return dict(weight=weight, df=df, power=power, mix=mix)


def configs():
    result = []
    for df, power in itertools.product((None, 5, 8), (1.0, 0.8, 1.2)):
        name = (
            BASE
            if df is None and power == 1
            else f"{'normal' if df is None else 't'+str(df)}_clock{power:g}"
        )
        result.append(dict(id=name, components=[component(df=df, power=power)]))
    for mix, weight in itertools.product((0.0, 1.0), (0.25, 0.5)):
        result.append(
            dict(
                id=f"vol{mix:g}_mixture{weight:g}",
                components=[component(1 - weight), component(weight, mix=mix)],
            )
        )
    for weight in (0.25, 0.5):
        result.append(
            dict(
                id=f"tail5_mixture{weight:g}",
                components=[component(1 - weight), component(weight, df=5)],
            )
        )
    result.append(
        dict(
            id="clock_mixture50", components=[component(0.5, power=0.8), component(0.5, power=1.2)]
        )
    )
    result.append(
        dict(
            id="joint_mixture",
            components=[
                component(0.5),
                component(0.25, df=5, power=0.8),
                component(0.25, df=8, power=1.2),
            ],
        )
    )
    return result


def center(history, steps, returns):
    recent = np.asarray(history[-300:], dtype=float)
    increments = np.log1p(recent) if returns else np.diff(recent, axis=0)
    drift = increments.mean(0) * 0.5
    grid = a.grid_for(steps, history.shape[1])
    anchor = np.zeros(history.shape[1]) if returns else history[-1]
    return (anchor[:, None] + drift[:, None] * grid).ravel()


def predict(history, steps, returns, family, monthly, cfg, draws, seed):
    if monthly:
        raise ValueError("Distribution suite compares daily models only")
    parts = cfg["components"]
    weights = np.array([p["weight"] for p in parts])
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid mixture weights")
    mean = center(history, steps, returns)
    output = None
    selector = np.random.default_rng(np.random.SeedSequence([seed, 2901])).random(draws)
    lower = 0.0
    for part, weight in zip(parts, weights, strict=True):
        numerical = dict(
            id="distribution_component",
            window=300,
            drift=0.5,
            mix=part["mix"],
            shrink=0.0,
            power=part["power"],
        )
        samples = j.predict(history, steps, returns, family, False, numerical, draws, seed)
        df = part["df"]
        if df is not None:
            if df <= 2:
                raise ValueError("Student-t degrees of freedom must exceed two")
            # Normalize theoretical variance, rather than widening the model accidentally.
            # The same positive scale acts on every asset and horizon in a draw.
            scale_rng = np.random.default_rng(np.random.SeedSequence([seed, 2902, int(df)]))
            scale = np.sqrt((df - 2) / scale_rng.chisquare(df, size=draws))
            samples = mean + (samples - mean) * scale[:, None]
        if output is None:
            output = np.empty_like(samples)
        mask = (selector >= lower) & (selector < lower + weight)
        if part is parts[-1]:
            mask = selector >= lower
        output[mask] = samples[mask]
        lower += weight
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite distribution sample")
    return output


def save(frame, cells, out):
    frame.to_csv(out / "diagnostic_cases.csv", index=False)
    cells.to_csv(out / "diagnostic_cells.csv", index=False)
    batch.summarize(frame).to_csv(out / "diagnostic_summary.csv", index=False)
    batch.save_diagnostics(frame, out, "diagnostic")
    years = []
    for year, subset in frame.groupby(frame.origin.str[:4]):
        table = batch.summarize(subset)
        table.insert(0, "year", year)
        years.append(table)
    pd.concat(years).to_csv(out / "diagnostic_by_year.csv", index=False)
    frame.groupby(["config", "cluster"]).composite.mean().unstack(0).to_csv(
        out / "diagnostic_by_basket.csv"
    )
    frame.groupby("config")[
        ["marginal_ratio", "joint_ratio", "tail_ratio", "coverage90"]
    ].mean().to_csv(out / "components.csv")
    cells["crps_ratio"] = cells.marginal_crps / cells.baseline_crps.replace(0, np.nan)
    cells.groupby(["config", "horizon"]).crps_ratio.mean().to_csv(out / "by_horizon.csv")
    compatible = frame.copy()
    compatible.loc[compatible.config == BASE, "config"] = batch.BASE
    f.write_json(
        out / "paired.json",
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
    items, inventory_exclusions, inputs = batch.inventory(batch.ROOT)
    items = [i for i in items if not i["monthly"]]
    cfgs = configs()
    seeds, draws = ([0], 200) if args.quick_check else (args.seeds, args.draws)
    origins = ORIGINS[:1] if args.quick_check else ORIGINS
    paths = [
        Path(__file__),
        Path(a.__file__),
        Path(batch.__file__),
        Path(f.__file__),
        Path(j.__file__),
        *sorted((batch.ROOT / "qfbench2_track_forecasting").glob("*.py")),
    ]
    code = {str(p.relative_to(batch.ROOT)): batch.sha(p) for p in paths}
    code["shared_crps"] = batch.sha(Path(crps.__file__))
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(
        created_utc=datetime.now(UTC).isoformat(),
        configs=cfgs,
        baseline=BASE,
        drift=0.5,
        window=300,
        shrinkage=0.0,
        draws=draws,
        seeds=seeds,
        origins=origins,
        bounds=BOUNDS,
        input_sha256=inputs,
        code_sha256=code,
        environment=batch.audit(batch.ROOT),
        quick_check=args.quick_check,
        rankable=False,
        promotion=False,
        monthly="unchanged baseline; not evaluated",
        warning=(
            "All dates have been explored, including 2024. Fixed diagnostic comparison only. "
            "No House calls, image change or submission."
        ),
    )
    f.write_json(out / "plan.json", plan)
    frames, cell_frames, omitted = [], [], []
    for index, item in enumerate(items):
        print(f"Basket {index+1}/{len(items)}: {item['source']}", flush=True)
        frame, cells, excluded = f.run_cases(
            [item],
            origins,
            cfgs,
            seeds,
            draws,
            "holdout",
            predictor=predict,
            baseline_cfg=cfgs[0],
            validation_bounds=BOUNDS,
            history_validator=a.validate_history,
        )
        if not frame.empty:
            frames.append(frame)
            cell_frames.append(cells)
            frame.to_csv(out / f"checkpoint-{index:02d}.csv", index=False)
        omitted.extend(excluded)
    f.write_json(out / "coverage.json", dict(inventory=inventory_exclusions, exclusions=omitted))
    if not frames:
        raise SystemExit("No eligible cases")
    frame, cells = pd.concat(frames, ignore_index=True), pd.concat(cell_frames, ignore_index=True)
    balanced = frame.groupby(["group", "origin", "seed"]).config.nunique()
    if not (balanced == len(cfgs)).all():
        raise ValueError("Unequal candidate coverage")
    if (
        any(batch.sha(batch.ROOT / p) != v for p, v in code.items() if p != "shared_crps")
        or any(batch.sha(batch.ROOT / p) != v for p, v in inputs.items())
        or batch.sha(Path(crps.__file__)) != code["shared_crps"]
    ):
        raise ValueError("Inputs or source changed during the run")
    save(frame, cells, out)
    f.write_json(
        out / "decision.json",
        dict(
            status="explored historical diagnostics",
            winner=str(batch.summarize(frame).iloc[0].config),
            promotion=False,
            cases=frame[["group", "origin"]].drop_duplicates().shape[0],
            baskets=frame.cluster.nunique(),
            methods=len(cfgs),
            warning=(
                "No untouched OOS remains in this batch. "
                "No automatic promotion or score guarantee."
            ),
        ),
    )
    print(batch.summarize(frame).to_string(index=False), flush=True)
    print("Complete. No image or competition submission changed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
