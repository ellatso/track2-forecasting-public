"""## Executive summary (read this first)

Run one frozen daily research suite: 18 drift/variance/correlation settings, the
prior marginal selector, a shared-composite selector, and a calibrated numeric
control. Freeze a shortlist on explored dates before evaluating January 2024.
Monthly models and official images remain unchanged. Results stay outside GitHub.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from qfbench2_common.scoring import crps

from . import adaptive as a
from . import batch
from . import frequency as f
from . import joint as j

BASE = batch.BASE
MARGINAL = "dynamic_marginal"
COMPOSITE = "dynamic_composite"
CALIBRATED = "calibrated_numeric_control"
VALIDATION = {"2024-01-02": ("2024-01-01", "2025-01-01")}


def configs():
    rows = []
    for drift, mix, shrink in itertools.product((0.0, 0.5, 1.0), (0.0, 0.5, 1.0), (0.0, 0.25)):
        name = (
            BASE if (drift, mix, shrink) == (1, 0, 0) else f"d{drift:g}_mix{mix:g}_corr{shrink:g}"
        )
        rows.append(dict(id=name, window=300, drift=drift, mix=mix, shrink=shrink, power=1.0))
    base = next(c for c in rows if c["id"] == BASE)
    rows += [
        dict(base, id=MARGINAL, shrink=0.25),
        dict(base, id=COMPOSITE, shrink=0.25),
        dict(base, id=CALIBRATED),
    ]
    return rows


def baseline_config():
    return next(c for c in configs() if c["id"] == BASE)


@lru_cache(maxsize=64)
def _composite_cached(data, shape, steps, grid_shape, returns, scoring_json, assets, horizons):
    history = np.frombuffer(data, dtype=np.float64).reshape(shape)
    grid = np.array(steps).reshape(grid_shape)
    longest = int(grid.max())
    origins = [len(history) - 1 - longest - 21 * k for k in range(6)]
    origins = [o for o in origins if o >= 299]
    result = dict(
        mix=0.0,
        inner_cases=len(origins),
        fallback=True,
        inner_latest_target_index=None,
        inner_last_history_index=len(history) - 1,
        inner_losses={},
    )
    if len(origins) < 3:
        return result
    card = dict(scoring=json.loads(scoring_json))
    bases, records = [], {w: [] for w in a.WEIGHTS}
    for origin in origins:
        prefix = history[: origin + 1]
        outcome = a.inner_outcome(history, origin, grid, returns)
        samples = j.predict(
            prefix, grid, returns, "T2-F1", False, baseline_config(), a.INNER_DRAWS, a.INNER_SEED
        )
        bases.append(batch.raw_score(samples, outcome, card, assets, horizons))
        for weight in a.WEIGHTS:
            cfg = dict(baseline_config(), id="inner", mix=weight, shrink=0.25)
            samples = j.predict(
                prefix, grid, returns, "T2-F1", False, cfg, a.INNER_DRAWS, a.INNER_SEED
            )
            records[weight].append((samples, outcome))
    # Average baseline scales over completed inner cases, avoiding a tiny single-case divisor.
    # Delegate effective weights and composite math entirely to the existing shared scorer.
    baseline = dict(bases[0])
    for component in ("marginal", "joint", "tail"):
        baseline[component] = float(np.mean([b[component] for b in bases]))
    if any(
        w and baseline[k] <= 1e-12
        for k, w in zip(("marginal", "joint", "tail"), baseline["weights_effective"], strict=True)
    ):
        return result
    losses = {
        w: float(
            np.mean(
                [batch.relative_score(s, y, card, baseline)["composite"] for s, y in records[w]]
            )
        )
        for w in a.WEIGHTS
    }
    if not all(np.isfinite(v) for v in losses.values()):
        raise ValueError("Nonfinite inner composite loss")
    best = min(losses.values())
    result.update(
        mix=next(w for w in a.WEIGHTS if losses[w] <= best + 1e-8),
        fallback=False,
        inner_losses=losses,
        inner_latest_target_index=max(origins) + longest,
    )
    return result


def choose_composite(history, steps, returns, item):
    history = np.ascontiguousarray(history, dtype=np.float64)
    grid = a.grid_for(steps, history.shape[1])
    return _composite_cached(
        history.tobytes(),
        history.shape,
        tuple(grid.ravel()),
        grid.shape,
        returns,
        json.dumps(item["card"].get("scoring", {}), sort_keys=True),
        tuple(item["assets"]),
        tuple(item["horizons"]),
    )


def predictor(item):
    def predict(history, steps, returns, family, monthly, cfg, draws, seed):
        if monthly:
            raise ValueError("Suite compares daily models only")
        history = np.ascontiguousarray(history, dtype=np.float64)
        if cfg["id"] == CALIBRATED:
            calibrated = next(c for c in batch.configs() if c["id"] == "calibrated_policy")
            return batch.predict(history, steps, returns, family, False, calibrated, draws, seed)
        chosen = dict(cfg)
        if cfg["id"] in (MARGINAL, COMPOSITE):
            chosen["mix"] = (
                a.choose_mix(history, steps, returns)
                if cfg["id"] == MARGINAL
                else choose_composite(history, steps, returns, item)
            )["mix"]
        return j.predict(history, steps, returns, family, False, chosen, draws, seed)

    return predict


def diagnostics(item):
    def report(history, steps, returns, cfg):
        result = dict(
            selected_mix=cfg["mix"],
            inner_cases=0,
            inner_fallback=False,
            inner_latest_target_index=None,
            inner_last_history_index=len(history) - 1,
            inner_losses="",
        )
        if cfg["id"] in (MARGINAL, COMPOSITE):
            choice = (
                a.choose_mix(history, steps, returns)
                if cfg["id"] == MARGINAL
                else choose_composite(history, steps, returns, item)
            )
            result.update(
                selected_mix=choice["mix"],
                inner_cases=choice["inner_cases"],
                inner_fallback=choice["fallback"],
                inner_latest_target_index=choice["inner_latest_target_index"],
                inner_losses=json.dumps(choice["inner_losses"], sort_keys=True),
            )
        result["history_sha256"] = hashlib.sha256(
            np.ascontiguousarray(history, dtype=np.float64).tobytes()
        ).hexdigest()
        return result

    return report


def run_cases(items, origins, bounds, cfgs, seeds, draws):
    frames, cells, skipped = [], [], []
    for index, item in enumerate(items):
        print(f"[{index+1}/{len(items)}] {item['source']}", flush=True)
        frame, cell, omitted = f.run_cases(
            [item],
            origins,
            cfgs,
            seeds,
            draws,
            "holdout",
            predictor=predictor(item),
            baseline_cfg=baseline_config(),
            validation_bounds=bounds,
            prediction_diagnostics=diagnostics(item),
            history_validator=a.validate_history,
        )
        frames.append(frame)
        cells.append(cell)
        skipped.extend(omitted)
    return pd.concat(frames, ignore_index=True), pd.concat(cells, ignore_index=True), skipped


def shortlist(frame):
    summary = batch.summarize(frame)
    # Freeze all comparisons before validation. Controls are mandatory, independent of their rank.
    names = [BASE, CALIBRATED, MARGINAL, COMPOSITE]
    names += summary[summary.config != BASE].config.head(3).tolist()
    return list(dict.fromkeys(names))


def save(frame, cells, omitted, out, phase):
    f.write_json(out / (phase + "_coverage.json"), omitted)
    if frame.empty:
        f.write_json(out / (phase + "_status.json"), dict(status="no eligible cases"))
        return
    frame.to_csv(out / (phase + "_cases.csv"), index=False)
    cells.to_csv(out / (phase + "_cells.csv"), index=False)
    batch.save_diagnostics(frame, out, phase)
    batch.summarize(frame).to_csv(out / (phase + "_summary.csv"), index=False)
    tables = []
    for year, subset in frame.groupby(frame.origin.str[:4]):
        table = batch.summarize(subset)
        table.insert(0, "year", year)
        tables.append(table)
    pd.concat(tables).to_csv(out / (phase + "_by_year.csv"), index=False)
    report = frame[frame.config.isin([MARGINAL, COMPOSITE])].drop_duplicates(
        ["config", "group", "origin"]
    )
    columns = [
        "config",
        "group",
        "cluster",
        "origin",
        "selected_mix",
        "inner_cases",
        "inner_latest_target_index",
        "inner_last_history_index",
        "inner_fallback",
        "inner_losses",
        "history_sha256",
    ]
    report[columns].to_csv(out / (phase + "_weight_audit.csv"), index=False)
    f.write_json(
        out / (phase + "_paired.json"),
        {
            name: batch.paired_interval(frame, name)
            for name in frame.config.unique()
            if name != BASE
        },
    )


def validation_report(frame):
    if frame.empty:
        return dict(status="insufficient data", promotion=False)
    cases = frame[["group", "origin"]].drop_duplicates().shape[0]
    baskets = frame.cluster.nunique()
    summary = batch.summarize(frame)
    winner = f.choose(summary)
    return dict(
        status="limited later-period validation",
        cases=cases,
        asset_baskets=baskets,
        lowest_loss_config=winner,
        promotion=False,
        reason=(
            "One later origin, overlapping baskets and limited coverage "
            "cannot establish robust official improvement."
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 17, 41])
    parser.add_argument(
        "--quick-check",
        action="store_true",
        help="Exercise all configurations on one explored date; no validation or promotion",
    )
    args = parser.parse_args(argv)
    out = args.run_dir.resolve()
    if out == batch.ROOT or batch.ROOT in out.parents:
        parser.error("Results must stay outside the public repository")
    if not 200 <= args.draws <= 20000 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use 200-20000 draws and unique seeds")
    if out.exists() and any(out.iterdir()):
        parser.error("An existing suite run cannot be overwritten")
    items, skipped, inputs = batch.inventory(batch.ROOT)
    items = [i for i in items if not i["monthly"]]
    code = {
        str(p.relative_to(batch.ROOT)): batch.sha(p)
        for p in [
            Path(__file__),
            Path(a.__file__),
            Path(batch.__file__),
            Path(f.__file__),
            Path(j.__file__),
            *sorted((batch.ROOT / "qfbench2_track_forecasting").glob("*.py")),
        ]
    }
    code["shared_crps"] = batch.sha(Path(crps.__file__))
    out.mkdir(parents=True, exist_ok=True)
    origins = a.ORIGINS[:1] if args.quick_check else a.ORIGINS
    seeds, draws = ([0], 200) if args.quick_check else (args.seeds, args.draws)
    plan = dict(
        created_utc=datetime.now(UTC).isoformat(),
        configs=configs(),
        seeds=seeds,
        draws=draws,
        diagnostic_origins=origins,
        diagnostic_bounds=a.BOUNDS,
        validation_bounds=VALIDATION,
        inner_weights=a.WEIGHTS,
        inner_draws=a.INNER_DRAWS,
        inner_seed=a.INNER_SEED,
        inner_spacing=21,
        inner_max_cases=6,
        input_sha256=inputs,
        code_sha256=code,
        environment=batch.audit(batch.ROOT),
        quick_check=args.quick_check,
        rankable=False,
        shortlist_rule=(
            "Baseline + calibrated/marginal/composite controls "
            "+ top three diagnostic configurations"
        ),
        warning=(
            "Diagnostics reuse explored dates. January 2024 is a later, newly used "
            "outer period with limited coverage. No House calls. "
            "Monthly baseline unchanged. No submission."
        ),
    )
    f.write_json(out / "plan.json", plan)
    f.write_json(out / "inventory_coverage.json", skipped)
    frame, cells, omitted = run_cases(items, origins, a.BOUNDS, configs(), seeds, draws)
    save(frame, cells, omitted, out, "diagnostic")
    if frame.empty:
        raise SystemExit("No diagnostic cases")
    locked = dict(
        configs=shortlist(frame),
        plan_sha256=batch.sha(out / "plan.json"),
        summary_sha256=batch.sha(out / "diagnostic_summary.csv"),
        frozen_utc=datetime.now(UTC).isoformat(),
    )
    f.write_json(out / "shortlist.json", locked)
    if not args.quick_check:
        # Check the lock before the first access to outer validation targets.
        if (
            locked["plan_sha256"] != batch.sha(out / "plan.json")
            or locked["summary_sha256"] != batch.sha(out / "diagnostic_summary.csv")
            or json.loads((out / "shortlist.json").read_text(encoding="utf-8")) != locked
            or any(
                batch.sha(batch.ROOT / path) != value
                for path, value in code.items()
                if path != "shared_crps"
            )
            or batch.sha(Path(crps.__file__)) != code["shared_crps"]
            or any(batch.sha(batch.ROOT / path) != value for path, value in inputs.items())
        ):
            raise ValueError("Source, input, plan or shortlist changed before validation")
        f.write_json(out / "validation_started.json", locked)
        cfgs = [c for c in configs() if c["id"] in locked["configs"]]
        valid, cells, omitted = run_cases(items, list(VALIDATION), VALIDATION, cfgs, seeds, draws)
        save(valid, cells, omitted, out, "validation")
        f.write_json(out / "decision.json", validation_report(valid))
        if not valid.empty:
            print("Later validation:", flush=True)
            print(batch.summarize(valid).to_string(index=False), flush=True)
    else:
        f.write_json(out / "decision.json", dict(status="quick check only", promotion=False))
    print("Explored-date diagnostics:", flush=True)
    print(batch.summarize(frame).to_string(index=False), flush=True)
    print("Suite complete. No competition submission or image change.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
