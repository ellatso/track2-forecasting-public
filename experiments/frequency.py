"""## Executive summary (read this first)

Select daily and monthly models separately on historical selection dates. Daily
comparisons change volatility only. Monthly alternatives use level reversion or
an attenuating trend. Lock both winners before evaluating new historical dates.
Outputs are local diagnostics, never official scores or untouched-data evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from qfbench2_common.scoring import crps

from . import batch

ROOT = batch.ROOT
BASE = batch.BASE
SELECTION = ("2011-01-03", "2013-01-02", "2015-01-02", "2017-01-03")
VALIDATION = ("2018-04-02", "2019-04-01")
BOUNDARY = "2018-04-01"
END = "2020-01-01"


def configs(monthly):
    base = dict(
        id=BASE,
        window=300,
        drift=1.0,
        spread=1.0,
        policy="fixed",
        volatility="sample",
        model="walk",
    )
    if not monthly:
        return [base, dict(base, id="daily_ewma60", volatility="ewma60")]
    return [
        base,
        dict(base, id="monthly_reversion120", window=120, model="reversion"),
        dict(base, id="monthly_damped12", window=120, model="damped", damping=0.8),
    ]


def predict(history, steps, returns, family, monthly, cfg, draws, seed):
    if cfg["model"] == "walk":
        return batch.predict(history, steps, returns, family, monthly, cfg, draws, seed)
    if not monthly or returns:
        raise ValueError("Monthly alternatives require level targets")
    recent = np.asarray(history[-cfg["window"] :], dtype=float)
    if len(recent) < 60 or not np.isfinite(recent).all():
        raise ValueError("Monthly model needs 60 finite level observations")
    grid = np.asarray(steps, dtype=int)
    if grid.ndim == 1:
        grid = np.tile(grid, (recent.shape[1], 1))
    state = np.tile(recent[-1], (draws, 1))
    result = np.empty((draws, recent.shape[1], grid.shape[1]))
    if cfg["model"] == "reversion":
        x, y = recent[:-1], recent[1:]
        xc, yc = x - x.mean(0), y - y.mean(0)
        phi = np.clip((xc * yc).sum(0) / np.maximum((xc * xc).sum(0), 1e-12), 0, 0.98)
        intercept = y.mean(0) - phi * x.mean(0)
        residual = y - (intercept + phi * x)
    else:
        increments = np.diff(recent, axis=0)
        residual = increments - increments.mean(0)
        time = np.arange(12, dtype=float)
        time -= time.mean()
        slope = (time[:, None] * recent[-12:]).sum(0) / (time * time).sum()
    sd = np.maximum(residual.std(0, ddof=1), 1e-10)
    corr = np.nan_to_num(np.atleast_2d(np.corrcoef(residual, rowvar=False)), nan=0.0)
    np.fill_diagonal(corr, 1)
    eig, vec = np.linalg.eigh(corr)
    positive = (vec * np.maximum(eig, 1e-8)) @ vec.T
    positive /= np.outer(np.sqrt(np.diag(positive)), np.sqrt(np.diag(positive)))
    chol = np.linalg.cholesky(positive)
    rng = np.random.default_rng(seed)
    for step in range(1, int(grid.max()) + 1):
        noise = (rng.normal(size=state.shape) @ chol.T) * sd
        if cfg["model"] == "reversion":
            state = intercept + phi * state + noise
        else:
            state = state + slope * cfg["damping"] ** step + noise
        for asset, column in np.argwhere(grid == step):
            result[:, asset, column] = state[:, asset]
    return result.reshape(draws, -1)


def run_cases(
    items,
    origins,
    cfgs,
    seeds,
    draws,
    phase,
    *,
    predictor=None,
    baseline_cfg=None,
    selection_end=BOUNDARY,
    validation_bounds=None,
    prediction_diagnostics=None,
    history_validator=None,
):
    predictor = predict if predictor is None else predictor
    rows, cells, excluded = [], [], []
    for i, item in enumerate(items):
        print(f"[{i+1}/{len(items)}] {item['source']}", flush=True)
        for origin in origins:
            try:
                history, outcome, actual, future = batch.case_at(item, origin)
                if phase == "select" and future >= selection_end:
                    raise ValueError("selection outcome enters new validation period")
                if phase == "holdout":
                    lower, upper = (
                        (BOUNDARY, END) if validation_bounds is None else validation_bounds[origin]
                    )
                    if actual < lower or future >= upper:
                        raise ValueError("validation outcome outside locked period")
                if history_validator is not None:
                    history_validator(item, actual)
                pending, pending_cells = [], []
                for seed in seeds:
                    base = predictor(
                        history,
                        item["steps"],
                        item["returns"],
                        item["family"],
                        item["monthly"],
                        configs(item["monthly"])[0] if baseline_cfg is None else baseline_cfg,
                        draws,
                        seed,
                    )
                    baseline = batch.raw_score(
                        base, outcome, item["card"], item["assets"], item["horizons"]
                    )
                    for cfg in cfgs:
                        samples = predictor(
                            history,
                            item["steps"],
                            item["returns"],
                            item["family"],
                            item["monthly"],
                            cfg,
                            draws,
                            seed,
                        )
                        raw = batch.raw_score(
                            samples, outcome, item["card"], item["assets"], item["horizons"]
                        )
                        metric = batch.relative_score(samples, outcome, item["card"], baseline)
                        common = dict(
                            config=cfg["id"],
                            group=item["group"],
                            cluster=item["cluster"],
                            source=item["source"],
                            family=item["family"],
                            monthly=item["monthly"],
                            origin=actual,
                            outcome_end=future,
                            seed=seed,
                        )
                        if prediction_diagnostics is not None:
                            common.update(
                                prediction_diagnostics(history, item["steps"], item["returns"], cfg)
                            )
                        components = {}
                        for k, weight in zip(
                            ("marginal", "joint", "tail"),
                            baseline["weights_effective"],
                            strict=True,
                        ):
                            ratio = raw[k] / baseline[k] if weight else 0.0
                            components[k + "_ratio"] = ratio
                            components[k + "_contribution"] = weight * ratio
                            components[k + "_baseline"] = baseline[k]
                        pending.append(
                            dict(
                                common,
                                **metric,
                                **components,
                                coverage90=float(
                                    np.mean(
                                        (outcome >= np.quantile(samples, 0.05, axis=0))
                                        & (outcome <= np.quantile(samples, 0.95, axis=0))
                                    )
                                ),
                            )
                        )
                        grid = np.asarray(item["steps"])
                        if grid.ndim == 1:
                            grid = np.tile(grid, (len(item["assets"]), 1))
                        for asset, name in enumerate(item["assets"]):
                            for column, horizon in enumerate(item["horizons"]):
                                j = asset * len(item["horizons"]) + column
                                # Import the organizer implementation; do not recreate CRPS.
                                pending_cells.append(
                                    dict(
                                        common,
                                        asset=name,
                                        horizon=horizon,
                                        observation_steps=int(grid[asset, column]),
                                        marginal_crps=crps.crps_marginal(
                                            samples[:, j : j + 1], outcome[j : j + 1]
                                        ),
                                        baseline_crps=crps.crps_marginal(
                                            base[:, j : j + 1], outcome[j : j + 1]
                                        ),
                                        predicted_mean=float(samples[:, j].mean()),
                                        q05=float(np.quantile(samples[:, j], 0.05)),
                                        q95=float(np.quantile(samples[:, j], 0.95)),
                                    )
                                )
                rows.extend(pending)
                cells.extend(pending_cells)
            except (Exception, SystemExit) as exc:
                excluded.append(dict(source=item["source"], origin=origin, reason=str(exc)))
    return pd.DataFrame(rows), pd.DataFrame(cells), excluded


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def choose(summary):
    # Equal loss retains the baseline; avoid promotion on negligible numerical ties.
    best = summary.iloc[0]
    return str(best["config"]) if best.local_relative_loss < 1 - 1e-8 else BASE


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=["select", "holdout"], default="select")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 17, 41])
    args = parser.parse_args(argv)
    out = args.run_dir.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error("Results must remain outside the public repository")
    if not 200 <= args.draws <= 20000 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use 200-20000 draws and nonempty unique seeds")
    items, skipped, inputs = batch.inventory(ROOT)
    code = {
        str(p.relative_to(ROOT)): batch.sha(p)
        for p in [
            Path(__file__),
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
            daily_configs=configs(False),
            monthly_configs=configs(True),
            seeds=args.seeds,
            draws=args.draws,
            selection_origins=SELECTION,
            holdout_origins=VALIDATION,
            selection_outcome_end_exclusive=BOUNDARY,
            holdout_outcome_end_exclusive=END,
            code_sha256=code,
            input_sha256=inputs,
            environment=batch.audit(ROOT),
            rankable=False,
            warning=(
                "New locked origins; public history was explored before. "
                "Not untouched OOS. House not evaluated."
            ),
        )
        write_json(plan_path, plan)
        locked = dict(plan_sha256=batch.sha(plan_path), winners={}, summary_sha256={})
    else:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        locked = json.loads((out / "selected.json").read_text(encoding="utf-8"))
        if (
            locked["plan_sha256"] != batch.sha(plan_path)
            or plan["code_sha256"] != code
            or plan["input_sha256"] != inputs
            or plan["environment"]["packages"] != batch.audit(ROOT)["packages"]
            or plan["environment"]["python"] != sys.version
        ):
            parser.error("Plan, code, inputs or environment changed; validation refused")
        for freq in ["daily", "monthly"]:
            path = out / f"selection_{freq}_summary.csv"
            if (
                batch.sha(path) != locked["summary_sha256"][freq]
                or choose(pd.read_csv(path)) != locked["winners"][freq]
            ):
                parser.error("Selection lock changed; validation refused")
        if (out / "holdout_started.json").exists():
            parser.error("Validation already started; read existing results")
        write_json(out / "holdout_started.json", locked)
    exclusions = {}
    validation = {}
    for freq, monthly in [("daily", False), ("monthly", True)]:
        cfgs = plan[freq + "_configs"]
        if args.phase == "holdout":
            cfgs = [c for c in cfgs if c["id"] in [BASE, locked["winners"][freq]]]
        frame, cells, omitted = run_cases(
            [i for i in items if i["monthly"] == monthly],
            plan["selection_origins"] if args.phase == "select" else plan["holdout_origins"],
            cfgs,
            plan["seeds"],
            plan["draws"],
            args.phase,
        )
        exclusions[freq] = omitted
        if frame.empty:
            write_json(
                out / (args.phase + "_coverage.json"),
                dict(inventory=skipped, exclusions=exclusions),
            )
            raise SystemExit(f"No eligible {freq} cases; see coverage report")
        prefix = ("selection" if args.phase == "select" else "holdout") + "_" + freq
        frame.to_csv(out / (prefix + "_cases.csv"), index=False)
        cells.to_csv(out / (prefix + "_cells.csv"), index=False)
        batch.save_diagnostics(frame, out, prefix)
        summary = batch.summarize(frame)
        summary.to_csv(out / (prefix + "_summary.csv"), index=False)
        print(freq, summary.to_string(index=False), flush=True)
        if args.phase == "select":
            locked["winners"][freq] = choose(summary)
            locked["summary_sha256"][freq] = batch.sha(out / (prefix + "_summary.csv"))
        else:
            winner = locked["winners"][freq]
            validation[freq] = batch.paired_interval(frame, winner)
            validation[freq]["winner"] = winner
            validation[freq]["improved"] = bool(
                winner != BASE
                and summary.set_index("config").loc[winner, "local_relative_loss"] < 1
            )
    write_json(
        out / (args.phase + "_coverage.json"), dict(inventory=skipped, exclusions=exclusions)
    )
    if args.phase == "select":
        write_json(out / "selected.json", locked)
    else:
        write_json(out / "validation.json", validation)
    print("Results:", out, "No competition submission was created.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
