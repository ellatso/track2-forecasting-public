"""## Executive summary (read this first)

Separate variance mixing and correlation shrinkage in a paired four-arm daily
experiment. Use the previous adaptive round's dates, coverage filter and seeds.
Report conditional effects and interaction with equal asset-basket weighting.
All periods are previously explored diagnostics, not independent validation.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from qfbench2_common.scoring import crps

from . import adaptive as a
from . import batch
from . import frequency as f
from . import joint as j

BASE = batch.BASE
VOL = "variance_only"
CORR = "correlation_only"
BOTH = "variance_and_correlation"
KEYS = ["group", "cluster", "origin", "seed"]
METRICS = ["composite", "marginal_contribution", "joint_contribution", "tail_contribution"]


def configs():
    base = dict(id=BASE, window=300, drift=1.0, mix=0.0, shrink=0.0, power=1.0)
    return [
        base,
        dict(base, id=VOL, mix=0.5),
        dict(base, id=CORR, shrink=0.25),
        dict(base, id=BOTH, mix=0.5, shrink=0.25),
    ]


def predict(history, steps, returns, family, monthly, cfg, draws, seed):
    return j.predict(
        np.ascontiguousarray(history, dtype=np.float64),
        steps,
        returns,
        family,
        monthly,
        cfg,
        draws,
        seed,
    )


def effects(frame):
    """Paired differences; negative means lower loss. Reject incomplete comparisons."""
    if frame.duplicated([*KEYS, "config"]).any():
        raise ValueError("Duplicate paired case")
    wide = frame.pivot(index=KEYS, columns="config", values=METRICS)
    if set(frame.config) != {BASE, VOL, CORR, BOTH} or wide.isna().any().any():
        raise ValueError("Incomplete four-arm paired case")
    contrasts = {
        "variance_without_shrinkage": {VOL: 1, BASE: -1},
        "shrinkage_without_variance": {CORR: 1, BASE: -1},
        "variance_with_shrinkage": {BOTH: 1, CORR: -1},
        "shrinkage_with_variance": {BOTH: 1, VOL: -1},
        "combined_vs_baseline": {BOTH: 1, BASE: -1},
        "interaction": {BOTH: 1, VOL: -1, CORR: -1, BASE: 1},
    }
    tables = []
    for label, weights in contrasts.items():
        table = pd.DataFrame(
            {
                metric: sum(weight * wide[metric, cfg] for cfg, weight in weights.items())
                for metric in METRICS
            }
        ).reset_index()
        table["effect"] = label
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def summarize_effects(frame):
    # Match batch.summarize: mean cases within each basket, then equal basket weights.
    baskets = frame.groupby(["effect", "cluster"])[METRICS].mean().reset_index()
    result = baskets.groupby("effect")[METRICS].mean().reset_index()
    result["asset_baskets"] = baskets.groupby("effect").size().to_numpy()
    return result


def run_cases(items, seeds, draws):
    return f.run_cases(
        items,
        a.ORIGINS,
        configs(),
        seeds,
        draws,
        "holdout",
        predictor=predict,
        baseline_cfg=configs()[0],
        validation_bounds=a.BOUNDS,
        history_validator=a.validate_history,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 17, 41])
    args = parser.parse_args(argv)
    out = args.run_dir.resolve()
    if out == batch.ROOT or batch.ROOT in out.parents:
        parser.error("Results must stay outside the public repository")
    if not 200 <= args.draws <= 20000 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use 200-20000 draws and nonempty unique seeds")
    if out.exists() and any(out.iterdir()):
        parser.error("An existing diagnostic run cannot be overwritten")
    items, skipped, inputs = batch.inventory(batch.ROOT)
    items = [item for item in items if not item["monthly"]]
    code = {
        str(path.relative_to(batch.ROOT)): batch.sha(path)
        for path in [
            Path(__file__),
            Path(a.__file__),
            Path(f.__file__),
            Path(j.__file__),
            Path(batch.__file__),
            *sorted((batch.ROOT / "qfbench2_track_forecasting").glob("*.py")),
        ]
    }
    code["shared_crps"] = batch.sha(Path(crps.__file__))
    out.mkdir(parents=True, exist_ok=True)
    f.write_json(
        out / "plan.json",
        dict(
            created_utc=datetime.now(UTC).isoformat(),
            configs=configs(),
            origins=a.ORIGINS,
            validation_bounds=a.BOUNDS,
            draws=args.draws,
            seeds=args.seeds,
            code_sha256=code,
            input_sha256=inputs,
            environment=batch.audit(batch.ROOT),
            rankable=False,
            monthly_policy="Unchanged baseline; monthly forecasts not evaluated",
            warning=(
                "Previously explored historical diagnostics. "
                "No independent holdout or House calls."
            ),
        ),
    )
    frame, cells, omitted = run_cases(items, args.seeds, args.draws)
    f.write_json(out / "coverage.json", dict(inventory=skipped, exclusions=omitted))
    if frame.empty:
        raise SystemExit("No eligible cases; inspect coverage")
    paired = effects(frame)
    frame.to_csv(out / "diagnostic_cases.csv", index=False)
    cells.to_csv(out / "diagnostic_cells.csv", index=False)
    batch.save_diagnostics(frame, out, "diagnostic")
    summary = batch.summarize(frame)
    summary.to_csv(out / "diagnostic_summary.csv", index=False)
    paired.to_csv(out / "effect_cases.csv", index=False)
    summarize_effects(paired).to_csv(out / "effect_summary.csv", index=False)
    years, year_effects = [], []
    for year, subset in frame.groupby(frame.origin.str[:4]):
        table = batch.summarize(subset)
        table.insert(0, "year", year)
        years.append(table)
        table = summarize_effects(effects(subset))
        table.insert(0, "year", year)
        year_effects.append(table)
    pd.concat(years, ignore_index=True).to_csv(out / "diagnostic_by_year.csv", index=False)
    pd.concat(year_effects, ignore_index=True).to_csv(out / "effect_by_year.csv", index=False)
    f.write_json(
        out / "paired_diagnostics.json",
        {cfg: batch.paired_interval(frame, cfg) for cfg in [VOL, CORR, BOTH]},
    )
    print(summary.to_string(index=False), flush=True)
    print(summarize_effects(paired).to_string(index=False), flush=True)
    print("Diagnostic only; no winner promotion or competition submission.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
