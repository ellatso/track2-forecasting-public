"""## Executive summary (read this first)

Compare factor-return distributions using prefix-only historical forecasts.
Exclude all windows overlapping a published practice target. Fix half drift,
seeds and draws when comparing volatility. Save scores only outside this repo.
"""

import argparse
import json
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from qfbench2_track_forecasting import cli, scoring
from qfbench2_track_forecasting.factor_candidate import estimate as factor_estimate
from qfbench2_track_forecasting.short_horizon import predict

from .m0_scale import distribution, normal_scales


def configurations():
    rows = []
    for vol in ["v9", "garch", "ew20", "ew60"]:
        for df in [0, 8]:
            rows.append(dict(id=f"{vol}_t{df}", vol=vol, df=df, drift=0.5, window=300))
    for width in [1.25, 1.5]:
        rows.append(
            dict(id=f"garch_t8_w{width}", vol="garch", df=8, drift=0.5, width=width, window=300)
        )
    for half in [20, 60]:
        rows.append(
            dict(id=f"ew20_t8_revert{half}", vol="ew20", df=8, drift=0.5, revert=half, window=300)
        )
    return rows


def score(samples, y, scales):
    n = len(y)
    m = scoring._composite(
        samples,
        y,
        weights=(0.5, 0.3, 0.2) if n > 1 else (5 / 7, 0, 2 / 7),
        tail_levels=(0.01, 0.05, 0.95, 0.99),
        joint="variogram",
        tail_metric="pinball",
        ref_scale=scales,
    )
    return dict(
        loss=min(m["composite"], 8), **{k: m[k] / scales[k] for k in ["marginal", "joint", "tail"]}
    )


def worker(task):
    name, frame, exclusions, origins, cfgs, draws, seeds, production = task
    data, dates = frame.to_numpy(), frame.index
    rows = []
    for counter, origin in enumerate(origins):
        pos = dates.searchsorted(origin, side="right") - 1
        if pos < 350:
            continue
        asof = dates[pos]
        raw = data[pos - 299 : pos + 1]
        steps = np.log1p(raw)
        if production:
            panel = (
                frame.iloc[pos - 299 : pos + 1]
                .rename_axis("date")
                .reset_index()
                .melt(id_vars="date", var_name="asset", value_name="value")
            )
        for h in [7, 21, 63, 126]:
            end = asof + pd.offsets.BDay(h)
            endpoint = dates.get_indexer([end])[0]
            if endpoint < 0 or any(asof <= hi and end >= lo for lo, hi in exclusions):
                continue
            y = np.log1p(data[pos + 1 : endpoint + 1]).sum(0)
            # M0 models panel simple returns, while the target sums log1p returns.
            mu, cov = distribution(raw[1:], np.zeros(len(y)), [h])
            scale = normal_scales(mu, cov)
            for seed in seeds:
                for cfg in cfgs:
                    if production:
                        args = (
                            {"factors": panel},
                            list(frame.columns),
                            [h],
                            str(asof.date()),
                            draws,
                            seed,
                        )
                        if cfg["id"] == "v9_t0_d0.5":
                            s, _ = cli._draw(
                                *args,
                                target_type="log_return",
                                drift_factor=0.5,
                                variance_mix=0.5,
                                sampling="sobol",
                            )
                        elif cfg["id"] == "garch_t8_d0.5":
                            s, _ = factor_estimate(*args)
                        else:
                            raise ValueError("Production comparison supports only selected models")
                        s = s.reshape(draws, -1)
                    else:
                        s = predict(steps, np.zeros(len(y)), [h], cfg, draws, seed).reshape(
                            draws, -1
                        )
                    rows.append(
                        dict(
                            group=name,
                            origin=str(asof.date()),
                            end=str(end.date()),
                            horizon=h,
                            config=cfg["id"],
                            seed=seed,
                            **score(s, y, scale),
                        )
                    )
        if counter % 25 == 0:
            print(name, counter, len(origins), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", default="2003-01-06")
    p.add_argument("--end", default="2014-12-01")
    p.add_argument("--stride", type=int, default=42)
    p.add_argument("--draws", type=int, default=1024)
    p.add_argument("--seeds", default="0")
    p.add_argument("--configs", type=Path)
    p.add_argument("--production", action="store_true")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if a.out.resolve() == root or root in a.out.resolve().parents:
        raise ValueError("Write research outputs outside repository")
    a.out.mkdir(parents=True, exist_ok=True)
    cfgs = json.loads(a.configs.read_text()) if a.configs else configurations()
    source = a.official / "units/t2-F1-ai-mom-2024/factors_daily.parquet"
    d = pd.read_parquet(source)
    d.date = pd.to_datetime(d.date)
    frame = d.pivot(index="date", columns="asset", values="value").sort_index()
    groups = {x: [x] for x in ["MKT", "SMB", "HML", "MOM", "BAB", "QMJ"]}
    groups["joint"] = ["MKT", "SMB", "HML", "MOM", "BAB", "QMJ"]
    tasks = []
    origins = pd.bdate_range(a.start, a.end)[:: a.stride]
    for name, assets in groups.items():
        exclude = []
        for path in (a.official / "units").glob("*/card.toml"):
            card = tomllib.loads(path.read_text())
            if not set(assets).intersection(card["targets"]["asset_ids"]):
                continue
            cutoff = pd.Timestamp(str(card["provenance"]["data_cutoff"])[:10])
            exclude.append(
                (
                    cutoff - pd.Timedelta(days=7),
                    cutoff + pd.offsets.BDay(max(card["targets"]["horizons"])),
                )
            )
        tasks.append(
            (
                name,
                frame[assets].dropna(),
                exclude,
                origins,
                cfgs,
                a.draws,
                list(map(int, a.seeds.split(","))),
                a.production,
            )
        )
    (a.out / "plan.json").write_text(
        json.dumps(
            dict(
                configs=cfgs,
                start=a.start,
                end=a.end,
                stride=a.stride,
                draws=a.draws,
                seeds=a.seeds,
                source=str(source),
                rankable=False,
                production=a.production,
                excludes_practice_target_windows=True,
            ),
            indent=2,
        )
    )
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for block in pool.map(worker, tasks):
            rows.extend(block)
            pd.DataFrame(rows).to_csv(a.out / "checkpoint.csv", index=False)
    f = pd.DataFrame(rows)
    f.to_csv(a.out / "cases.csv", index=False)
    table = f.groupby(["config", "group"]).loss.mean().unstack()
    table["balanced"] = 0.5 * table.drop(columns="joint").mean(axis=1) + 0.5 * table.joint
    table.sort_values("balanced").to_csv(a.out / "summary.csv")
    print(table.sort_values("balanced").to_string(), flush=True)


if __name__ == "__main__":
    main()
