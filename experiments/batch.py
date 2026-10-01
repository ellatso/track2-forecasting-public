"""## Executive summary (read this first)

Compare 20 numerical configurations on fixed historical origins, then evaluate only
one locked winner on later dates. All results stay outside the public checkout.
Shared organizer code computes losses; these are local diagnostics, not board scores.
No House call, private answer, other-card lookup, or packaged fitted model is used.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from qfbench2_track_forecasting import cli, scoring
from qfbench2_track_forecasting.grid import GridSpec
from qfbench2_track_forecasting.normalization import NormalizationMode

ROOT = Path(__file__).resolve().parents[1]
DEV_ORIGINS = ("2011-01-03", "2013-01-02", "2015-01-02", "2017-01-03")
HOLDOUT_ORIGINS = ("2021-01-04", "2023-01-03")
BASE = "w300_d1_s1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configs():
    rows = []
    for w, d, s in itertools.product((60, 120, 300), (0.0, 0.5, 1.0), (0.85, 1.0)):
        rows.append(
            dict(
                id=f"w{w}_d{d:g}_s{s:g}",
                window=w,
                drift=d,
                spread=s,
                policy="fixed",
                volatility="sample",
            )
        )
    rows.extend(
        [
            dict(
                id="calibrated_policy",
                window=300,
                drift=1.0,
                spread=1.0,
                policy="calibrated",
                volatility="sample",
            ),
            dict(
                id="ewma60", window=300, drift=0.5, spread=1.0, policy="fixed", volatility="ewma60"
            ),
        ]
    )
    return rows


def predict(history, steps, returns, family, monthly, cfg, draws, seed):
    """Inputs are already truncated; future outcomes never enter this function."""
    recent = np.asarray(history[-cfg["window"] :], dtype=float)
    increments = np.log1p(recent) if returns else np.diff(recent, axis=0)
    if len(increments) < 30 or not np.isfinite(increments).all():
        raise ValueError("insufficient finite increments")
    drift_factor, spread = cfg["drift"], cfg["spread"]
    if cfg["policy"] == "calibrated" and not monthly:
        drift_factor, spread = {
            "T2-F1": (0.5, 0.85),
            "T2-F2": (0.5, 0.85),
            "T2-F3": (1.0, 0.85),
        }.get(family, (1.0, 1.0))
    drift = increments.mean(axis=0) * drift_factor
    sd = increments.std(axis=0, ddof=1)
    if cfg["volatility"] == "ewma60":
        weights = np.exp2(-np.arange(len(increments) - 1, -1, -1) / 60.0)
        weights /= weights.sum()
        center = (increments * weights[:, None]).sum(axis=0)
        sd = np.sqrt(((increments - center) ** 2 * weights[:, None]).sum(axis=0))
    sd = np.maximum(sd, 1e-10) * spread
    corr = np.atleast_2d(np.corrcoef(increments, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    eig, vec = np.linalg.eigh(corr)
    positive = (vec * np.maximum(eig, 1e-8)) @ vec.T
    norm = np.sqrt(np.diag(positive))
    positive /= np.outer(norm, norm)
    chol = np.linalg.cholesky(positive)
    rng = np.random.default_rng(seed)
    path = np.zeros((draws, history.shape[1]))
    grid = np.asarray(steps, dtype=int)
    if grid.ndim == 1:
        grid = np.tile(grid, (history.shape[1], 1))
    result = np.empty((draws, history.shape[1], grid.shape[1]))
    anchor = np.zeros(history.shape[1]) if returns else history[-1]
    previous = 0
    for horizon in np.unique(grid):
        span = int(horizon) - previous
        path += (rng.normal(size=path.shape) @ chol.T) * sd * np.sqrt(span) + drift * span
        for asset, column in np.argwhere(grid == horizon):
            result[:, asset, column] = anchor[asset] + path[:, asset]
        previous = int(horizon)
    return result.reshape(draws, -1)


def raw_score(samples, outcome, card, assets, horizons):
    return scoring._score(
        dict(
            realized=outcome,
            _samples=samples,
            card=card,
            expected_grid=GridSpec(tuple(assets), tuple(horizons)),
            unit_handle="local-history",
            grid_source="card",
            normalization_mode=NormalizationMode.RAW_UNRANKABLE,
        )
    )


def relative_score(samples, outcome, card, baseline):
    scales = {k: baseline[k] for k in ("marginal", "joint", "tail")}
    weights = tuple(baseline["weights_effective"])
    for k, weight in zip(scales, weights, strict=True):
        if not np.isfinite(scales[k]) or (weight and scales[k] <= 1e-12):
            raise ValueError("degenerate local baseline component")
        if not weight:
            scales[k] = 1.0
    params = card.get("scoring", {}).get("params", {})
    return scoring._composite(
        samples,
        outcome,
        weights=weights,
        tail_levels=tuple(params.get("tail_levels", (0.01, 0.05, 0.95, 0.99))),
        joint=scoring.card_joint_statistic(card),
        tail_metric=params.get("tail_metric", "pinball"),
        ref_scale=scales,
    )


def inventory(root):
    """Choose longest single-unit history per identical grid; never merge cards."""
    selected, skipped, hashes = {}, [], {}
    for cp in sorted((root / "units").glob("*/card.toml")):
        try:
            card = tomllib.loads(cp.read_text())
            target = card["targets"]
            assets, horizons = list(target["asset_ids"]), list(target["horizons"])
            asof = str(card["provenance"]["data_cutoff"])[:10]
            panels = cli._read_panels(cp.parent / "panels")
            series = [cli._series(panels, a, asof).rename(a) for a in assets]
            if any(s.index.has_duplicates for s in series):
                raise ValueError("duplicate observation date")
            frame = pd.concat(series, axis=1, join="inner").dropna().sort_index()
            frame.index = pd.to_datetime(frame.index)
            if not np.isfinite(frame.to_numpy()).all():
                raise ValueError("nonfinite panel")
            monthly_steps = cli._monthly_inputs(panels, card, cp, asof)
            monthly = monthly_steps is not None
            steps = np.array(horizons if monthly_steps is None else monthly_steps, dtype=int)
            if np.any(np.diff(steps, axis=-1) <= 0) or np.any(steps <= 0):
                raise ValueError("unordered steps")
            family = card.get("metadata", {}).get("category", "")
            returns = target.get("target_type") == "log_return"
            key = json.dumps([sorted(assets), horizons, steps.tolist(), monthly, returns, family])
            paths = sorted(
                set(cp.parent.glob("*.parquet")) | set((cp.parent / "panels").glob("*.parquet"))
            )
            paths += [cp]
            spec = cp.parent / "forecast_spec.json"
            if spec.exists():
                paths.append(spec)
            sources = {str(p.relative_to(root)): sha(p) for p in paths}
            item = dict(
                card=card,
                assets=assets,
                horizons=horizons,
                steps=steps,
                monthly=monthly,
                returns=returns,
                family=family,
                frame=frame,
                source=cp.parent.name,
                sources=sources,
                group=hashlib.sha256(key.encode()).hexdigest()[:16],
                cluster=json.dumps(sorted(assets)),
            )
            if key not in selected or (len(frame), frame.index[-1], cp.parent.name) > (
                len(selected[key]["frame"]),
                selected[key]["frame"].index[-1],
                selected[key]["source"],
            ):
                if key in selected:
                    skipped.append(
                        dict(
                            source=selected[key]["source"],
                            reason="duplicate grid: longer history retained",
                        )
                    )
                selected[key] = item
            else:
                skipped.append(
                    dict(source=cp.parent.name, reason="duplicate grid: longer history retained")
                )
        except (Exception, SystemExit) as exc:
            skipped.append(dict(source=cp.parent.name, reason=str(exc)))
    items = sorted(selected.values(), key=lambda x: x["group"])
    for item in items:
        hashes.update(item["sources"])
    return items, skipped, hashes


def case_at(item, origin):
    """Locate an origin on this card's own panel and refuse long/missing intervals."""
    frame, steps = item["frame"], item["steps"]
    t = frame.index.searchsorted(pd.Timestamp(origin), side="right") - 1
    longest = int(np.max(steps))
    minimum = 60 if item["monthly"] else 300
    if t < minimum - 1 or t + longest >= len(frame):
        raise ValueError("insufficient history or future before original cutoff")
    if (pd.Timestamp(origin) - frame.index[t]).days > (45 if item["monthly"] else 7):
        raise ValueError("origin lies in a history gap")
    segment = frame.iloc[max(0, t - 299) : t + longest + 1]
    gaps = np.diff(segment.index.to_numpy()).astype("timedelta64[D]").astype(int)
    if item["monthly"]:
        if np.any(np.diff(segment.index.to_period("M").asi8) != 1):
            raise ValueError("missing monthly observation")
    elif np.max(gaps) > 10:
        raise ValueError("history/future spans a gap")
    values = frame.to_numpy()
    history = values[: t + 1]
    grid = np.asarray(steps, dtype=int)
    if grid.ndim == 1:
        grid = np.tile(grid, (values.shape[1], 1))
    outcome = np.empty(grid.shape)
    if item["returns"] and np.any(values[t + 1 : t + longest + 1] <= -1):
        raise ValueError("invalid return")
    for asset, column in np.ndindex(grid.shape):
        span = int(grid[asset, column])
        outcome[asset, column] = (
            np.log1p(values[t + 1 : t + span + 1, asset]).sum()
            if item["returns"]
            else values[t + span, asset]
        )
    return (
        history,
        outcome.ravel(),
        str(frame.index[t].date()),
        str(frame.index[t + longest].date()),
    )


def run_cases(items, origins, cfgs, seeds, draws):
    rows, excluded = [], []
    base_cfg = next(c for c in configs() if c["id"] == BASE)
    for i, item in enumerate(items):
        print(f"[{i+1}/{len(items)}] {item['source']}", flush=True)
        for origin in origins:
            try:
                history, outcome, actual, future = case_at(item, origin)
                if origin in DEV_ORIGINS and future >= "2020-01-01":
                    raise ValueError("selection outcome enters holdout period")
                # Preflight every configuration: an error cannot improve a method by omission.
                pending = []
                for seed in seeds:
                    base = predict(
                        history,
                        item["steps"],
                        item["returns"],
                        item["family"],
                        item["monthly"],
                        base_cfg,
                        draws,
                        seed,
                    )
                    baseline = raw_score(
                        base, outcome, item["card"], item["assets"], item["horizons"]
                    )
                    for cfg in cfgs:
                        samples = predict(
                            history,
                            item["steps"],
                            item["returns"],
                            item["family"],
                            item["monthly"],
                            cfg,
                            draws,
                            seed,
                        )
                        metric = relative_score(samples, outcome, item["card"], baseline)
                        pending.append(
                            dict(
                                config=cfg["id"],
                                group=item["group"],
                                cluster=item["cluster"],
                                source=item["source"],
                                family=item["family"],
                                monthly=item["monthly"],
                                origin=actual,
                                outcome_end=future,
                                seed=seed,
                                **metric,
                                coverage90=float(
                                    np.mean(
                                        (outcome >= np.quantile(samples, 0.05, axis=0))
                                        & (outcome <= np.quantile(samples, 0.95, axis=0))
                                    )
                                ),
                            )
                        )
                rows.extend(pending)
            except (Exception, SystemExit) as exc:
                excluded.append(dict(source=item["source"], origin=origin, reason=str(exc)))
    return pd.DataFrame(rows), excluded


def summarize(frame):
    # Each asset basket has equal weight, regardless of repeated horizons/cards/seeds.
    baskets = frame.groupby(["config", "cluster"], as_index=False).composite.mean()
    summary = baskets.groupby("config").composite.agg(["mean", "median", "count"]).reset_index()
    summary = summary.rename(columns={"mean": "local_relative_loss", "count": "asset_baskets"})
    return summary.sort_values(["local_relative_loss", "config"])


def save_diagnostics(frame, out, prefix):
    frame.groupby(["config", "family", "monthly", "seed"]).agg(
        local_relative_loss=("composite", "mean"),
        coverage90=("coverage90", "mean"),
        cases=("composite", "size"),
    ).to_csv(out / (prefix + "_by_family_seed.csv"))


def paired_interval(frame, winner):
    baskets = frame.groupby(["config", "cluster"]).composite.mean().unstack(0)
    delta = (baskets[winner] - baskets[BASE]).to_numpy()
    if len(delta) < 2:
        return dict(asset_baskets=len(delta), interval=None, warning="too few baskets")
    rng = np.random.default_rng(71)
    bootstrap = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
    return dict(
        asset_baskets=len(delta),
        mean_difference=float(delta.mean()),
        interval95=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        caveat=(
            "Asset baskets can share instruments; this is descriptive, " "not independent evidence."
        ),
    )


def audit(root):
    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "qfbench2-common"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "source checkout; see code hashes"
    result = {"python": sys.version, "platform": platform.platform(), "packages": packages}
    for name, args in [
        ("git_commit", ["rev-parse", "HEAD"]),
        ("git_status", ["status", "--porcelain"]),
    ]:
        p = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        result[name] = p.stdout.strip()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=["select", "holdout"], default="select")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 17, 41])
    args = parser.parse_args(argv)
    out = args.run_dir.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error("Results must be outside the public repository.")
    if not 200 <= args.draws <= 20000 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use 200-20000 draws and unique seeds.")
    items, skipped, inputs = inventory(ROOT)
    code = {
        str(p.relative_to(ROOT)): sha(p)
        for p in (
            Path(__file__),
            ROOT / "qfbench2_track_forecasting/cli.py",
            ROOT / "qfbench2_track_forecasting/scoring.py",
            ROOT / "qfbench2_track_forecasting/tail.py",
            ROOT / "qfbench2_track_forecasting/horizons.py",
            ROOT / "qfbench2_track_forecasting/grid.py",
            ROOT / "qfbench2_track_forecasting/normalization.py",
        )
    }
    from qfbench2_common.scoring import crps

    code["shared_crps"] = sha(Path(crps.__file__))
    plan_path = out / "plan.json"
    if args.phase == "select":
        if out.exists() and any(out.iterdir()):
            parser.error("Use a fresh run directory; an existing run is immutable.")
        out.mkdir(parents=True, exist_ok=True)
        plan = dict(
            created_utc=datetime.now(UTC).isoformat(),
            configs=configs(),
            seeds=args.seeds,
            draws=args.draws,
            selection_origins=DEV_ORIGINS,
            holdout_origins=HOLDOUT_ORIGINS,
            code_sha256=code,
            input_sha256=inputs,
            rankable=False,
            baseline="local 300-observation drift, NOT official M0",
            environment=audit(ROOT),
        )
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        frame, excluded = run_cases(items, DEV_ORIGINS, configs(), args.seeds, args.draws)
        (out / "coverage.json").write_text(
            json.dumps(
                dict(
                    inventory_skipped=skipped,
                    selection_excluded=excluded,
                    selected_source_units=len(items),
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        if frame.empty:
            raise SystemExit("No eligible historical selection cases; see coverage.json.")
        frame.to_csv(out / "selection_cases.csv", index=False)
        save_diagnostics(frame, out, "selection")
        summary = summarize(frame)
        summary.to_csv(out / "selection_summary.csv", index=False)
        winner = str(summary.iloc[0]["config"])
        locked = dict(
            config=winner,
            plan_sha256=sha(plan_path),
            selection_only=True,
            selection_summary_sha256=sha(out / "selection_summary.csv"),
            warning="Later test is a locked diagnostic, not untouched competition data.",
        )
        (out / "selected.json").write_text(json.dumps(locked, indent=2), encoding="utf-8")
        print(summary.head(10).to_string(index=False))
        print("Locked:", winner, "Run holdout separately using the same --run-dir.")
    else:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        locked = json.loads((out / "selected.json").read_text(encoding="utf-8"))
        if (
            sha(plan_path) != locked["plan_sha256"]
            or code != plan["code_sha256"]
            or inputs != plan["input_sha256"]
            or audit(ROOT)["packages"] != plan["environment"]["packages"]
            or sys.version != plan["environment"]["python"]
            or sha(out / "selection_summary.csv") != locked["selection_summary_sha256"]
            or locked["config"] != pd.read_csv(out / "selection_summary.csv").iloc[0]["config"]
        ):
            parser.error("Plan, source code or input bytes changed; holdout refused.")
        flag = out / "holdout_started.json"
        if flag.exists():
            parser.error(
                "Holdout already started. Read existing results; do not repeatedly select on it."
            )
        cfgs = [c for c in plan["configs"] if c["id"] in (locked["config"], BASE)]
        flag.write_text(json.dumps(locked, indent=2), encoding="utf-8")
        frame, excluded = run_cases(
            items, plan["holdout_origins"], cfgs, plan["seeds"], plan["draws"]
        )
        (out / "holdout_coverage.json").write_text(json.dumps(excluded, indent=2), encoding="utf-8")
        if frame.empty:
            raise SystemExit("No eligible later cases; see holdout_coverage.json.")
        frame.to_csv(out / "holdout_cases.csv", index=False)
        save_diagnostics(frame, out, "holdout")
        (out / "paired_interval.json").write_text(
            json.dumps(paired_interval(frame, locked["config"]), indent=2), encoding="utf-8"
        )
        summary = summarize(frame)
        summary.to_csv(out / "holdout_summary.csv", index=False)
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
