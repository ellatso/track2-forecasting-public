"""## Executive summary (read this first)

Compare daily variance blending, correlation shrinkage and long-horizon variance
accumulation without changing drift, seeds or draws. Lock the selection winner
before evaluating 2020/2022 historical origins. Monthly forecasts stay unchanged.
All metrics come from shared organizer scoring; no competition upload is created.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from qfbench2_common.scoring import crps

from . import batch
from . import frequency as f

ROOT = batch.ROOT
BASE = batch.BASE
SELECTION = f.SELECTION
BOUNDS = {"2020-01-02": ("2020-01-01", "2021-01-01"), "2022-01-03": ("2022-01-01", "2023-01-01")}


def configs():
    result = []
    for mix, shrink, power in itertools.product((0.0, 0.25, 0.5), (0.0, 0.25), (1.0, 0.8)):
        result.append(
            dict(
                id=BASE
                if (mix, shrink, power) == (0, 0, 1)
                else f"mix{mix:g}_corr{shrink:g}_clock{power:g}",
                window=300,
                drift=1.0,
                mix=mix,
                shrink=shrink,
                power=power,
            )
        )
    return result


def variance_clock(horizon, power):
    # Business observations, not calendar days. Short horizons preserve Brownian variance.
    horizon = np.asarray(horizon, dtype=float)
    return np.where(horizon <= 20, horizon, 20 * (horizon / 20) ** power)


def predict(history, steps, returns, family, monthly, cfg, draws, seed):
    if monthly:
        raise ValueError("Third round compares daily models only")
    if cfg["id"] == BASE:
        return batch.predict(
            history, steps, returns, family, False, f.configs(False)[0], draws, seed
        )
    recent = np.asarray(history[-cfg["window"] :], dtype=float)
    increments = np.log1p(recent) if returns else np.diff(recent, axis=0)
    if len(increments) < 30 or not np.isfinite(increments).all():
        raise ValueError("Insufficient finite increments")
    drift = increments.mean(0) * cfg["drift"]
    sample_var = increments.var(0, ddof=1)
    weights = np.exp2(-np.arange(len(increments) - 1, -1, -1) / 60.0)
    weights /= weights.sum()
    center = (increments * weights[:, None]).sum(0)
    recent_var = ((increments - center) ** 2 * weights[:, None]).sum(0)
    sd = np.maximum(np.sqrt((1 - cfg["mix"]) * sample_var + cfg["mix"] * recent_var), 1e-10)
    corr = np.nan_to_num(np.atleast_2d(np.corrcoef(increments, rowvar=False)), nan=0)
    np.fill_diagonal(corr, 1)
    corr = (1 - cfg["shrink"]) * corr + cfg["shrink"] * np.eye(len(sd))
    eig, vec = np.linalg.eigh(corr)
    positive = (vec * np.maximum(eig, 1e-8)) @ vec.T
    norm = np.sqrt(np.diag(positive))
    positive /= np.outer(norm, norm)
    chol = np.linalg.cholesky(positive)
    grid = np.asarray(steps, dtype=int)
    if grid.ndim == 1:
        grid = np.tile(grid, (len(sd), 1))
    result = np.empty((draws, len(sd), grid.shape[1]))
    rng = np.random.default_rng(seed)
    noise_path = np.zeros((draws, len(sd)))
    anchor = np.zeros(len(sd)) if returns else history[-1]
    previous_clock = 0.0
    for horizon in np.unique(grid):
        clock = float(variance_clock(horizon, cfg["power"]))
        noise_path += (
            (rng.normal(size=noise_path.shape) @ chol.T) * sd * np.sqrt(clock - previous_clock)
        )
        for asset, column in np.argwhere(grid == horizon):
            result[:, asset, column] = anchor[asset] + drift[asset] * horizon + noise_path[:, asset]
        previous_clock = clock
    return result.reshape(draws, -1)


def run_cases(items, origins, cfgs, seeds, draws, phase):
    return f.run_cases(
        items,
        origins,
        cfgs,
        seeds,
        draws,
        phase,
        predictor=predict,
        baseline_cfg=configs()[0],
        selection_end=f.BOUNDARY,
        validation_bounds=BOUNDS,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=["select", "holdout"], default="select")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 17, 41])
    args = parser.parse_args(argv)
    out = args.run_dir.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error("Results must stay outside the public repository")
    if not 200 <= args.draws <= 20000 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use 200-20000 draws and nonempty unique seeds")
    items, skipped, inputs = batch.inventory(ROOT)
    items = [i for i in items if not i["monthly"]]
    code = {
        str(p.relative_to(ROOT)): batch.sha(p)
        for p in [
            Path(__file__),
            Path(f.__file__),
            Path(batch.__file__),
            *sorted((ROOT / "qfbench2_track_forecasting").glob("*.py")),
        ]
    }
    code["shared_crps"] = batch.sha(Path(crps.__file__))
    plan_path = out / "plan.json"
    if args.phase == "select":
        if out.exists() and any(out.iterdir()):
            parser.error("Use a fresh run directory")
        out.mkdir(parents=True, exist_ok=True)
        plan = dict(
            created_utc=datetime.now(UTC).isoformat(),
            configs=configs(),
            seeds=args.seeds,
            draws=args.draws,
            selection_origins=SELECTION,
            holdout_origins=list(BOUNDS),
            validation_bounds=BOUNDS,
            code_sha256=code,
            input_sha256=inputs,
            environment=batch.audit(ROOT),
            rankable=False,
            monthly_policy="Unchanged baseline; not evaluated or promoted in this round",
            warning=(
                "Fixed historical origins; public history explored before. "
                "Not untouched OOS. No House calls."
            ),
        )
        f.write_json(plan_path, plan)
        cfgs = plan["configs"]
    else:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        locked = json.loads((out / "selected.json").read_text(encoding="utf-8"))
        summary_path = out / "selection_summary.csv"
        if (
            locked["plan_sha256"] != batch.sha(plan_path)
            or plan["code_sha256"] != code
            or plan["input_sha256"] != inputs
            or plan["environment"]["python"] != sys.version
            or plan["environment"]["packages"] != batch.audit(ROOT)["packages"]
            or locked["summary_sha256"] != batch.sha(summary_path)
            or locked["config"] != f.choose(pd.read_csv(summary_path))
        ):
            parser.error("Source, plan, environment, input or selection lock changed")
        if (out / "holdout_started.json").exists():
            parser.error("Validation already started; read existing results")
        f.write_json(out / "holdout_started.json", locked)
        cfgs = [c for c in plan["configs"] if c["id"] in (BASE, locked["config"])]
    frame, cells, omitted = run_cases(
        items,
        plan["selection_origins"] if args.phase == "select" else plan["holdout_origins"],
        cfgs,
        plan["seeds"],
        plan["draws"],
        args.phase,
    )
    f.write_json(out / (args.phase + "_coverage.json"), dict(inventory=skipped, exclusions=omitted))
    if frame.empty:
        raise SystemExit("No eligible cases; inspect coverage report")
    prefix = "selection" if args.phase == "select" else "holdout"
    frame.to_csv(out / (prefix + "_cases.csv"), index=False)
    cells.to_csv(out / (prefix + "_cells.csv"), index=False)
    batch.save_diagnostics(frame, out, prefix)
    summary = batch.summarize(frame)
    summary.to_csv(out / (prefix + "_summary.csv"), index=False)
    if args.phase == "select":
        f.write_json(
            out / "selected.json",
            dict(
                config=f.choose(summary),
                plan_sha256=batch.sha(plan_path),
                summary_sha256=batch.sha(out / "selection_summary.csv"),
            ),
        )
    else:
        report = batch.paired_interval(frame, locked["config"])
        report["winner"] = locked["config"]
        report["improved"] = bool(
            locked["config"] != BASE
            and summary.set_index("config").loc[locked["config"], "local_relative_loss"] < 1
        )
        f.write_json(out / "validation.json", report)
    print(summary.to_string(index=False), flush=True)
    print("Results:", out, "No competition submission was created.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
