"""## Executive summary (read this first)

Compare short-horizon forecasting rules under the October 6 expected-error scale.
Use one published panel per domain, sliced before each historical origin; no
practice target is looked up. Fit only on the prefix, then score known historical
outcomes privately through the organizer's scoring functions. Selection and later
confirmation periods are disjoint. These series have been explored previously.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from qfbench2_track_forecasting import scoring
from qfbench2_track_forecasting.short_horizon import predict

from .m0_scale import distribution, normal_scales

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [2, 5, 10, 12]


def configs():
    base = dict(id="v9", vol="v9", drift=0.5)
    rows = [base, dict(id="m0", vol="sample", drift=1.0)]
    for vol in [
        "sample",
        "v9",
        "ew5",
        "ew10",
        "ew20",
        "ew40",
        "ew60",
        "ew120",
        "blend10",
        "blend20",
        "blend60",
        "har",
        "garch",
    ]:
        rows.append(dict(id=vol + "_zero", vol=vol, drift=0.0))
    for vol in ["ew10", "ew20", "ew60", "blend20", "har", "garch"]:
        for df in [5, 8]:
            rows.append(dict(id=vol + f"_t{df}", vol=vol, df=df, drift=0.0))
        rows.append(dict(id=vol + "_log", vol=vol, drift=0.0, log=True))
    for vol in ["ew20", "ew60", "blend20", "garch"]:
        rows.extend(
            [
                dict(id=vol + "_drift25", vol=vol, drift=0.25),
                dict(id=vol + "_ar", vol=vol, mean="ar", strength=0.5, drift=0.0),
                dict(id=vol + "_trend", vol=vol, mean="trend", strength=0.1, drift=0.0),
                dict(id=vol + "_corr20", vol=vol, corr=20, drift=0.0),
                dict(id=vol + "_corr60", vol=vol, corr=60, drift=0.0),
                dict(id=vol + "_tailmix", vol=vol, df=5, tail_weight=0.25, drift=0.0),
                dict(id=vol + "_volmix", vol=vol, vol_mixture=True, drift=0.0),
            ]
        )
    return rows


def frames(official):
    paths = {
        "rates": official / "units/t2-F1-hawkish-cut-2024/rates_daily.parquet",
        "fx": official / "units/t2-F3-election-2024-joint/g10_fx_daily.parquet",
    }
    groups = {
        "rates": ["UST_2Y", "UST_5Y", "UST_10Y", "UST_30Y"],
        "fx_major": ["EUR", "GBP", "JPY", "CHF"],
        "fx_other": ["AUD", "CAD", "NOK", "NZD"],
        "fx_nordic": ["SEK", "DKK"],
    }
    result = {}
    for name, assets in groups.items():
        p = paths["rates" if name == "rates" else "fx"]
        d = pd.read_parquet(p)
        col = "asset" if "asset" in d else "asset_id"
        d.date = pd.to_datetime(d.date)
        frame = d.pivot(index="date", columns=col, values="value")[assets].dropna().sort_index()
        result[name] = (frame, p)
    return result


def one_group(task):
    name, frame, origins, cfgs, draws, seeds, horizons, practice = task
    values = frame.to_numpy()
    dates = frame.index
    rows = []
    for counter, origin in enumerate(origins):
        pos = dates.searchsorted(origin, side="right") - 1
        if pos < 350:
            continue
        actual = dates[pos]
        ends = [actual + pd.offsets.BDay(h) for h in horizons]
        index = dates.get_indexer(ends)
        if np.any(index < 0):
            continue
        # Exclude overlapping official practice windows for this basket entirely.
        if any(actual <= hi and ends[-1] >= lo for lo, hi in practice):
            continue
        hist = values[max(0, pos - 998) : pos + 1]
        when = dates[max(0, pos - 998) : pos + 1]
        diffs = np.diff(hist, axis=0)
        gap = np.diff(when.to_numpy()).astype("timedelta64[D]").astype(float)
        valid = gap <= max(10 * np.median(gap[-299:]), 5)
        diffs = diffs[valid]
        logs = np.diff(np.log(hist), axis=0)[valid] if name.startswith("fx") else diffs
        anchor = hist[-1]
        outcome = values[index].T.ravel()
        mean, cov = distribution(diffs, anchor, horizons)
        scale = normal_scales(mean, cov)
        # Balance single-cell and joint shapes; freeze the cell rotation before results.
        cell = counter % len(outcome)
        sscale = normal_scales(mean[cell : cell + 1], cov[cell : cell + 1, cell : cell + 1])
        for seed in seeds:
            for cfg in cfgs:
                use_log = cfg.get("log", False) and name.startswith("fx")
                if cfg.get("production"):
                    from .production_audit import forecast

                    samples = forecast(
                        frame.iloc[max(0, pos - 998) : pos + 1],
                        horizons,
                        draws,
                        seed,
                        coupling=cfg.get("coupling"),
                        width=cfg.get("width", 1.0),
                        online_width=cfg.get("online_width", False),
                    ).reshape(draws, -1)
                    if cfg.get("mixture") is not None:
                        from qfbench2_track_forecasting.short_horizon import innovations

                        other = predict(
                            diffs, anchor, horizons, dict(vol="sample", drift=0.0), draws, seed
                        ).reshape(draws, -1)
                        _, _, selector = innovations(
                            draws, len(anchor), len(horizons), seed + 104729, 0
                        )
                        samples[selector < cfg["mixture"]] = other[selector < cfg["mixture"]]
                else:
                    samples = predict(
                        logs if use_log else diffs,
                        np.log(anchor) if use_log else anchor,
                        horizons,
                        cfg,
                        draws,
                        seed,
                        np.ones(len(anchor), bool) if use_log else None,
                    ).reshape(draws, -1)
                joint = scoring._composite(
                    samples,
                    outcome,
                    weights=(0.5, 0.3, 0.2),
                    tail_levels=(0.01, 0.05, 0.95, 0.99),
                    joint="variogram",
                    tail_metric="pinball",
                    ref_scale=scale,
                )
                single = scoring._composite(
                    samples[:, cell : cell + 1],
                    outcome[cell : cell + 1],
                    weights=(5 / 7, 0.0, 2 / 7),
                    tail_levels=(0.01, 0.05, 0.95, 0.99),
                    joint="variogram",
                    tail_metric="pinball",
                    ref_scale=sscale,
                )
                rows.append(
                    dict(
                        group=name,
                        origin=str(actual.date()),
                        end=str(ends[-1].date()),
                        config=cfg["id"],
                        seed=seed,
                        loss=0.75 * min(joint["composite"], 8) + 0.25 * min(single["composite"], 8),
                        joint_loss=min(joint["composite"], 8),
                        single_loss=min(single["composite"], 8),
                        marginal=joint["marginal"] / scale["marginal"],
                        joint=joint["joint"] / scale["joint"],
                        tail=joint["tail"] / scale["tail"],
                    )
                )
                if cfg.get("production") and not cfg.get("coupling"):
                    row = rows[-1]
                    pit = np.mean(samples <= outcome, axis=0)
                    row["pit_mean"] = float(pit.mean())
                    row["bias_z"] = float(
                        np.mean((outcome - samples.mean(0)) / np.maximum(samples.std(0), 1e-10))
                    )
                    for coverage in (50, 80, 95):
                        alpha = (1 - coverage / 100) / 2
                        lo, hi = np.quantile(samples, [alpha, 1 - alpha], axis=0)
                        inside = ((lo <= outcome) & (outcome <= hi)).reshape(
                            len(anchor), len(horizons)
                        )
                        row[f"coverage{coverage}"] = float(inside.mean())
                        for j, horizon in enumerate(horizons):
                            row[f"coverage{coverage}_h{horizon}"] = float(inside[:, j].mean())
        if counter % 25 == 0:
            print(name, counter, len(origins), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", default="2003-01-06")
    p.add_argument("--end", default="2014-12-01")
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--draws", type=int, default=512)
    p.add_argument("--seeds", default="0")
    p.add_argument("--configs", type=Path)
    p.add_argument("--horizons", default="2,5,10,12")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError("Save private results outside checkout")
    out.mkdir(parents=True, exist_ok=True)
    cfgs = json.loads(args.configs.read_text()) if args.configs else configs()
    horizons = list(map(int, args.horizons.split(",")))
    seeds = list(map(int, args.seeds.split(",")))
    origins = pd.bdate_range(args.start, args.end)[:: args.stride]
    data = frames(args.official)
    tasks = []
    sources = {}
    for group, (frame, path) in data.items():
        excluded = []
        for cp in (args.official / "units").glob("*/card.toml"):
            c = tomllib.loads(cp.read_text())
            assets = set(c["targets"]["asset_ids"])
            if not assets.intersection(frame.columns):
                continue
            date = pd.Timestamp(str(c["provenance"]["data_cutoff"])[:10])
            excluded.append(
                (date - pd.Timedelta(days=7), date + pd.offsets.BDay(max(c["targets"]["horizons"])))
            )
        tasks.append((group, frame, origins, cfgs, args.draws, seeds, horizons, excluded))
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = dict(
        start=args.start,
        end=args.end,
        stride=args.stride,
        horizons=horizons,
        seeds=seeds,
        draws=args.draws,
        configs=cfgs,
        sources=sources,
        scorer="organizer _composite, common 2.6.0",
        scale="M0 expected error (2026-10-06)",
        weighting="75% joint grid and 25% rotating one-cell grid, equal domain",
        rankable=False,
    )
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for block in pool.map(one_group, tasks):
            rows.extend(block)
            pd.DataFrame(rows).to_csv(out / "checkpoint.csv", index=False)
    f = pd.DataFrame(rows)
    f.to_csv(out / "cases.csv", index=False)
    s = f.groupby(["config", "group"]).loss.mean().unstack()
    s["fx"] = s[[c for c in s if c.startswith("fx")]].mean(axis=1)
    s["balanced"] = (s.rates + s.fx) / 2
    s.sort_values("balanced").to_csv(out / "summary.csv")
    print(s.sort_values("balanced").to_string())
    f.groupby(["config", "group", f.origin.str[:4]]).loss.mean().to_csv(out / "years.csv")
    print(
        "COMPLETE",
        len(f),
        "scores",
        f[["group", "origin"]].drop_duplicates().shape[0],
        "cases",
        flush=True,
    )


if __name__ == "__main__":
    main()
