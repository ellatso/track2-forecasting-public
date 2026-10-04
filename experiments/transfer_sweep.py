"""## Executive summary (read this first)

Simulate a target history ending two, five or ten years before its current anchor.
Use one public unit's own FX panel for target, context and historical outcomes.
Never reconstruct practice outcomes. Compare stale absolute volatility with
scale-free early volatility and contemporaneous context. Keep results private.
"""

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from qfbench2_track_forecasting import cli

from . import batch, precision

ORIGINS = [f"{year}-06-01" for year in (2008, 2011, 2014, 2017, 2020, 2023)]
GAPS = [2, 5, 10]
SEEDS = [0, 17, 41]
CONFIGS = ["v9", "zero", "rescale", "logearly", "context25", "context50", "context100"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.run_dir.resolve()
    if batch.ROOT in out.parents or out.exists():
        raise ValueError("Use a new private directory")
    out.mkdir()
    selected = None
    excluded_cutoffs = []
    for cp in sorted((batch.ROOT / "units").glob("*/card.toml")):
        card = tomllib.loads(cp.read_text())
        cutoff = str(card["provenance"]["data_cutoff"])[:10]
        excluded_cutoffs.append((set(card["targets"]["asset_ids"]), pd.Timestamp(cutoff)))
        for panel in cli._read_panels(cp.parent / "panels").values():
            col = "asset_id" if "asset_id" in panel else "asset"
            if not {"EUR", "JPY", "GBP", "AUD", "CAD", "CHF"} <= set(panel[col]):
                continue
            frame = panel.pivot(index="date", columns=col, values="value").sort_index().dropna()
            frame.index = pd.to_datetime(frame.index)
            if selected is None or len(frame) > len(selected[0]):
                selected = (frame, cp.parent.name)
    frame, source = selected
    baskets = [[a] for a in frame.columns] + [["EUR", "GBP", "JPY"], ["AUD", "CAD", "NZD"]]
    plan = dict(
        origins=ORIGINS,
        gaps_years=GAPS,
        seeds=SEEDS,
        configs=CONFIGS,
        source=source,
        inputs_sha256=batch.sha(batch.ROOT / "units" / source / "g10_fx_daily.parquet"),
        warning="Synthetic G10 transfer diagnostic; not observed EM transfer performance.",
    )
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    rows = []
    for assets in baskets:
        for origin in ORIGINS:
            t = frame.index.searchsorted(pd.Timestamp(origin), side="right") - 1
            if t < 300 or t + 64 >= len(frame):
                continue
            actual = frame.index[t]
            if any(set(assets) & a and abs((actual - d).days) <= 7 for a, d in excluded_cutoffs):
                continue
            for years in GAPS:
                end = actual - pd.DateOffset(years=years)
                early = frame.loc[:end, assets].iloc[-299:]
                if len(early) < 299:
                    continue
                target = early.copy()
                target.loc[actual] = frame.iloc[t][assets]
                panel = (
                    target.rename_axis("date")
                    .reset_index()
                    .melt(id_vars="date", var_name="asset_id", value_name="value")
                )
                horizons = [21, 64] if len(assets) > 1 else [64]
                outcome = np.array(
                    [[frame.iloc[t + h][a] for h in horizons] for a in assets]
                ).ravel()
                last = frame.iloc[t][assets].to_numpy()
                m0 = precision.m0_samples(
                    early.to_numpy(),
                    horizons,
                    False,
                    assets,
                    str(actual.date()),
                    str(assets) + str(years),
                )
                m0 += np.repeat(last - early.iloc[-1].to_numpy(), len(horizons))
                card = dict(
                    scoring=dict(
                        params=dict(
                            joint="variogram",
                            tail_levels=[0.01, 0.05, 0.95, 0.99],
                            weights=dict(marginal=0.5, joint=0.3, tail=0.2),
                        )
                    )
                )
                raw = batch.raw_score(m0, outcome, card, assets, horizons)
                scale = {k: max(raw[k], 1e-12) for k in ["marginal", "joint", "tail"]}
                weights = tuple(raw["weights_effective"])
                recent = frame.iloc[max(0, t - 299) : t + 1].drop(columns=assets)
                context = np.median(np.log(recent).diff().std(ddof=1))
                early_log = np.log(early).diff().std(ddof=1).to_numpy()
                for seed in SEEDS:
                    samples, stats = cli._draw(
                        {"target": panel},
                        assets,
                        horizons,
                        str(actual.date()),
                        4096,
                        seed,
                        drift_factor=0.5,
                        variance_mix=0.5,
                        sampling="sobol",
                    )
                    sd = np.array([stats["daily_sd"][a] for a in assets])
                    drift = np.array([stats["daily_drift"][a] for a in assets])
                    base_mean = last[:, None] + drift[:, None] * np.array(horizons)
                    noise = samples - base_mean
                    for cfg in CONFIGS:
                        if cfg == "v9":
                            result = samples
                        else:
                            multiplier = np.ones(len(assets))
                            if cfg == "rescale":
                                multiplier = last / early.iloc[-1].to_numpy()
                            elif cfg == "logearly":
                                multiplier = last * early_log / sd
                            elif cfg.startswith("context"):
                                w = int(cfg.replace("context", "")) / 100
                                target_log = np.exp(
                                    (1 - w) * np.log(np.maximum(early_log, 1e-12))
                                    + w * np.log(context)
                                )
                                multiplier = last * target_log / sd
                            result = last[None, :, None] + noise * multiplier[None, :, None]
                        metric = precision.score(
                            result.reshape(4096, -1), outcome, dict(card=card), scale, weights
                        )
                        rows.append(
                            dict(
                                config=cfg,
                                cluster=json.dumps(assets),
                                origin=str(actual.date()),
                                gap=years,
                                seed=seed,
                                **metric,
                                clipped=float(np.clip(metric["composite"], 0, 4)),
                            )
                        )
        print("complete basket", assets, flush=True)
        pd.DataFrame(rows).to_csv(out / "cases.csv", index=False)
    result = pd.DataFrame(rows)
    table = (
        result.groupby(["config", "cluster"]).clipped.mean().groupby("config").mean().sort_values()
    )
    table.to_csv(out / "summary.csv")
    print(table.to_string(), flush=True)


if __name__ == "__main__":
    main()
