"""## Executive summary (read this first)

Compare current production, larger normal samples, antithetic samples and Sobol
samples. Compare monthly recent trend means separately. Rebuild the documented
M0 distribution on historical pseudo-cases, never practice targets. Keep both
per-case and pooled-scale diagnostics. These explored snapshots are not new OOS.
Monthly snapshots lack historical release vintages. Save all results privately.
"""

from __future__ import annotations

import argparse
import tomllib
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from qfbench2_track_forecasting import cli, scoring

from . import batch, distributions, frequency, means

BASE = "current1000"
SEEDS = [0, 17, 41, 73, 101]


def configs(monthly):
    rows = [dict(id=BASE, sampling="pseudo", draws=1000, drift=0.5, trend=0.0)]
    for sampling in ["pseudo", "antithetic", "sobol"]:
        rows.append(dict(id=sampling + "4096", sampling=sampling, draws=4096, drift=0.5, trend=0.0))
    if monthly:
        for strength in [0.5, 1.0]:
            rows.append(
                dict(
                    id=f"trend12_mix{strength:g}",
                    sampling="sobol",
                    draws=4096,
                    drift=0.5,
                    trend=strength,
                )
            )
        rows.append(
            dict(id="recent12", sampling="sobol", draws=4096, drift=0.5, trend=1.0, recent=True)
        )
    else:
        for drift in [0.0, 0.25]:
            rows.append(
                dict(
                    id=f"sobol_drift{drift:g}", sampling="sobol", draws=4096, drift=drift, trend=0.0
                )
            )
    return rows


def panels_for(history, monthly):
    values = np.ascontiguousarray(history[-300:], dtype=float)
    assets = [f"asset{i}" for i in range(values.shape[1])]
    dates = (
        pd.date_range("1900-01-01", periods=len(values), freq="MS")
        if monthly
        else pd.bdate_range("1900-01-01", periods=len(values))
    )
    panel = (
        pd.DataFrame(values, index=dates, columns=assets)
        .rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="asset_id", value_name="value")
    )
    return {"history": panel}, assets, str(dates[-1].date())


def predict(history, steps, returns, monthly, cfg, seed, prepared=None):
    panels, assets, asof = panels_for(history, monthly) if prepared is None else prepared
    grid = np.asarray(steps, dtype=int)
    keys = list(range(1, grid.shape[-1] + 1)) if monthly else list(grid)
    samples, _ = cli._draw(
        panels,
        assets,
        keys,
        asof,
        cfg["draws"],
        seed,
        target_type="log_return" if returns else "level",
        panel_steps=grid if monthly else None,
        drift_factor=cfg["drift"],
        variance_mix=0.5,
        sampling=cfg["sampling"],
    )
    result = samples.reshape(cfg["draws"], -1)
    if monthly and cfg["trend"]:
        model = dict(model="trend", window=12, half_life=12, strength=cfg["trend"])
        old = means.baseline_mean(history, steps, False, True).ravel()
        if cfg.get("recent"):
            actual_grid = np.tile(grid, (len(assets), 1)) if grid.ndim == 1 else grid
            proposed = (
                history[-1, :, None] + np.diff(history[-13:], axis=0).mean(0)[:, None] * actual_grid
            )
            new = proposed.ravel()
        else:
            new = means.forecast_mean(history, steps, False, True, model)
        result += (new - old)[None, :]
    return result


def m0_samples(history, steps, returns, assets, origin, group):
    """Documented M0: sorted cells, raw return rows, 500 draws, fixed CRC seed."""
    values = np.asarray(history[-300:], dtype=float)
    order = np.argsort(assets)
    sorted_values = values[:, order]
    increments = sorted_values[1:] if returns else np.diff(sorted_values, axis=0)
    drift = increments.mean(0)
    covariance = np.atleast_2d(np.cov(increments, rowvar=False, ddof=1))
    anchor = np.zeros(len(assets)) if returns else sorted_values[-1]
    grid = np.asarray(steps, dtype=int)
    if grid.ndim == 1:
        grid = np.tile(grid, (len(assets), 1))
    cells = sorted((assets[ai], int(grid[ai, hi]), ai, hi) for ai, hi in np.ndindex(grid.shape))
    sorted_lookup = {int(original): index for index, original in enumerate(order)}
    indices = np.array([sorted_lookup[ai] for _, _, ai, _ in cells])
    horizons = np.array([h for _, h, _, _ in cells])
    mean = anchor[indices] + drift[indices] * horizons
    joint_cov = (
        np.minimum(horizons[:, None], horizons[None, :]) * covariance[np.ix_(indices, indices)]
    )
    joint_cov += np.eye(len(cells)) * 1.1e-9
    try:
        root = np.linalg.cholesky(joint_cov)
    except np.linalg.LinAlgError:
        root = np.diag(np.sqrt(np.diag(joint_cov)))
    seed = zlib.crc32(("history:" + group + ":" + origin).encode()) & 0x7FFFFFFF
    samples = mean + np.random.default_rng(seed).standard_normal((500, len(cells))) @ root.T
    result = np.empty((500, *grid.shape))
    for column, (_, _, ai, hi) in enumerate(cells):
        result[:, ai, hi] = samples[:, column]
    return result.reshape(500, -1)


def score(samples, outcome, item, scales, weights):
    params = item["card"].get("scoring", {}).get("params", {})
    return scoring._composite(
        samples,
        outcome,
        weights=weights,
        tail_levels=tuple(params.get("tail_levels", (0.01, 0.05, 0.95, 0.99))),
        joint=scoring.card_joint_statistic(item["card"]),
        tail_metric=params.get("tail_metric", "pinball"),
        ref_scale=scales,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--quick-check", action="store_true")
    args = parser.parse_args(argv)
    out = args.run_dir.resolve()
    if out == batch.ROOT or batch.ROOT in out.parents or (out.exists() and any(out.iterdir())):
        parser.error("Use an empty private directory outside the repository")
    out.mkdir(parents=True, exist_ok=True)
    items, skipped, inputs = batch.inventory(batch.ROOT)
    seeds = SEEDS[:1] if args.quick_check else SEEDS
    origins = distributions.ORIGINS[:1] if args.quick_check else distributions.ORIGINS
    source_paths = [
        Path(__file__),
        *sorted((batch.ROOT / "qfbench2_track_forecasting").glob("*.py")),
    ]
    code = {str(p.relative_to(batch.ROOT)): batch.sha(p) for p in source_paths}
    plan = dict(
        seeds=seeds,
        origins=origins,
        daily=configs(False),
        monthly=configs(True),
        input_sha256=inputs,
        code_sha256=code,
        rankable=False,
        warning=(
            "Explored diagnostics; monthly publication vintages unavailable. "
            "No official score promise."
        ),
    )
    frequency.write_json(out / "plan.json", plan)
    practice = []
    for cp in (batch.ROOT / "units").glob("*/card.toml"):
        card = tomllib.loads(cp.read_text())
        practice.append(
            (
                set(card["targets"]["asset_ids"]),
                pd.Timestamp(str(card["provenance"]["data_cutoff"])[:10]),
            )
        )
    rows, excluded = [], []
    for ix, item in enumerate(items):
        print(f"grid {ix+1}/{len(items)} monthly={item['monthly']} {item['source']}", flush=True)
        for origin in origins:
            try:
                history, outcome, actual, end = batch.case_at(item, origin)
                lower, upper = distributions.BOUNDS[origin]
                if actual < lower or end >= upper:
                    raise ValueError("outcome outside locked diagnostic year")
                if any(
                    set(item["assets"]) & assets and abs((pd.Timestamp(actual) - when).days) <= 7
                    for assets, when in practice
                ):
                    raise ValueError("origin within seven days of published practice cutoff")
            except ValueError as exc:
                excluded.append(dict(group=item["group"], origin=origin, reason=str(exc)))
                continue
            prepared = panels_for(history, item["monthly"])
            m0 = m0_samples(
                history, item["steps"], item["returns"], item["assets"], actual, item["group"]
            )
            m0_raw = batch.raw_score(m0, outcome, item["card"], item["assets"], item["horizons"])
            scales = {k: m0_raw[k] if m0_raw[k] > 0 else 1.0 for k in ["marginal", "joint", "tail"]}
            weights = tuple(m0_raw["weights_effective"])
            for seed in seeds:
                for cfg in configs(item["monthly"]):
                    samples = predict(
                        history,
                        item["steps"],
                        item["returns"],
                        item["monthly"],
                        cfg,
                        seed,
                        prepared,
                    )
                    metric = score(samples, outcome, item, scales, weights)
                    rows.append(
                        dict(
                            config=cfg["id"],
                            group=item["group"],
                            cluster=item["cluster"],
                            source=item["source"],
                            origin=actual,
                            end=end,
                            monthly=item["monthly"],
                            family=item["family"],
                            seed=seed,
                            **metric,
                            clipped_composite=float(np.clip(metric["composite"], 0, 4)),
                            **{k + "_m0": m0_raw[k] for k in ["marginal", "joint", "tail"]},
                        )
                    )
        pd.DataFrame(rows).to_csv(out / "checkpoint.csv", index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "cases.csv", index=False)
    frequency.write_json(out / "coverage.json", dict(excluded=excluded, inventory_skipped=skipped))
    for monthly, name in [(False, "daily"), (True, "monthly")]:
        selected = frame[frame.monthly == monthly].copy()
        print(name, flush=True)
        table = (
            selected.groupby(["config", "cluster"])
            .clipped_composite.mean()
            .groupby("config")
            .mean()
            .sort_values()
        )
        table.to_csv(out / (name + "_m0_summary.csv"))
        print(table.to_string(), flush=True)
        # Pooled scales give a separate diagnostic without a near-zero per-case denominator.
        reference = selected[selected.config == BASE]
        pooled = reference.groupby("group")[
            [k + "_m0" for k in ["marginal", "joint", "tail"]]
        ].mean()
        stable = []
        lookup = {i["group"]: i for i in items}
        for _, row in selected.iterrows():
            item = lookup[row.group]
            history, outcome, _, _ = batch.case_at(item, row.origin)
            cfg = next(c for c in configs(monthly) if c["id"] == row.config)
            samples = predict(history, item["steps"], item["returns"], monthly, cfg, int(row.seed))
            weights = tuple(
                batch.raw_score(samples, outcome, item["card"], item["assets"], item["horizons"])[
                    "weights_effective"
                ]
            )
            scale = {
                k: max(float(pooled.loc[row.group, k + "_m0"]), 1e-12) if w else 1.0
                for k, w in zip(["marginal", "joint", "tail"], weights, strict=True)
            }
            metric = score(samples, outcome, item, scale, weights)
            stable.append(dict(row, **metric))
        stable = pd.DataFrame(stable)
        stable.to_csv(out / (name + "_pooled_cases.csv"), index=False)
        summary = batch.summarize(stable)
        summary.to_csv(out / (name + "_pooled_summary.csv"), index=False)
        print(summary.to_string(index=False), flush=True)
        stable.groupby(["config", "seed"]).composite.mean().to_csv(out / (name + "_seed.csv"))
        stable.groupby(["config", stable.origin.str[:4]]).composite.mean().to_csv(
            out / (name + "_year.csv")
        )
    if any(batch.sha(batch.ROOT / p) != h for p, h in code.items()) or any(
        batch.sha(batch.ROOT / p) != h for p, h in inputs.items()
    ):
        raise ValueError("Source or input changed during run")
    print("Complete; historical diagnostics only.", flush=True)


if __name__ == "__main__":
    main()
