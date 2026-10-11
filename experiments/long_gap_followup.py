"""## Executive summary (read this first)

Test long-horizon forecasts when the target has an old history and today's anchor.
Use other FX series only before each origin. Omit every overlapping practice
target window. Outputs and historical evaluation observations remain private.
"""

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from qfbench2_track_forecasting.short_horizon import ew_cov, predict

from .factor_followup import score
from .final_study import frames
from .m0_scale import distribution, normal_scales


def configurations():
    return [
        dict(id="v9", drift=0.5),
        dict(id="zero", drift=0.0),
        dict(id="logscale", logscale=True),
        dict(id="context50", logscale=True, context=0.5),
        dict(id="context50_t8", logscale=True, context=0.5, df=8),
        dict(id="context50_t8_floor25", logscale=True, context=0.5, df=8, floor=0.25),
        dict(id="context50_t8_floor50", logscale=True, context=0.5, df=8, floor=0.5),
        dict(id="context50_t8_floor100", logscale=True, context=0.5, df=8, floor=1.0),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", default="2006-01-05")
    p.add_argument("--end", default="2014-12-02")
    p.add_argument(
        "--target-end", help="Latest outcome date; defaults to the origin end-year close"
    )
    p.add_argument("--stride", type=int, default=35)
    p.add_argument("--draws", type=int, default=1024)
    p.add_argument("--seeds", default="0")
    p.add_argument("--configs", type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if a.out.resolve() == root or root in a.out.resolve().parents:
        raise ValueError("Save research outputs outside repository")
    a.out.mkdir(parents=True, exist_ok=True)
    cfgs = json.loads(a.configs.read_text()) if a.configs else configurations()
    data = frames(a.official)
    frame = pd.concat([data[g][0] for g in ["fx_major", "fx_other", "fx_nordic"]], axis=1).dropna()
    vals, dates = frame.to_numpy(), frame.index
    exclusions = []
    for cp in (a.official / "units").glob("*/card.toml"):
        c = tomllib.loads(cp.read_text())
        d = pd.Timestamp(str(c["provenance"]["data_cutoff"])[:10])
        exclusions.append(
            (
                set(c["targets"]["asset_ids"]),
                d - pd.Timedelta(days=7),
                d + pd.offsets.BDay(max(c["targets"]["horizons"])),
            )
        )
    rows = []
    target_end = pd.Timestamp(a.target_end or f"{pd.Timestamp(a.end).year}-12-31")
    (a.out / "plan.json").write_text(
        json.dumps(
            dict(
                configs=cfgs,
                start=a.start,
                end=a.end,
                stride=a.stride,
                draws=a.draws,
                seeds=a.seeds,
                horizons=[21, 63, 126],
                gaps=[2, 5, 10],
                rankable=False,
                target_end=str(target_end.date()),
            ),
            indent=2,
        )
    )
    for counter, origin in enumerate(pd.bdate_range(a.start, a.end)[:: a.stride]):
        i = dates.searchsorted(origin, side="right") - 1
        for gap in [2, 5, 10]:
            old = dates.searchsorted(dates[i] - pd.DateOffset(years=gap), side="right") - 1
            if old < 300:
                continue
            for ai in [counter % 10, (counter + 5) % 10]:
                asset = frame.columns[ai]
                # Last 300 rows consist of 299 old observations plus today's anchor.
                past = vals[old - 298 : old + 1, ai : ai + 1]
                x = np.diff(past, axis=0)
                anchor = vals[i, ai : ai + 1]
                logx = np.diff(np.log(past), axis=0) * anchor
                others = np.arange(10) != ai
                early = np.diff(np.log(vals[old - 298 : old + 1, others]), axis=0)
                recent = np.diff(np.log(vals[i - 299 : i + 1, others]), axis=0)
                ratio = float(
                    np.clip(
                        np.median(
                            np.diag(ew_cov(recent, 20))
                            / np.maximum(np.diag(ew_cov(early, 20)), 1e-12)
                        ),
                        0.25,
                        4,
                    )
                )
                context_sd = float(np.median(np.sqrt(np.diag(ew_cov(recent, 20)))))
                for h in [21, 63, 126]:
                    end = dates[i] + pd.offsets.BDay(h)
                    if end > target_end:
                        continue
                    target = dates.get_indexer([end])[0]
                    if target < 0 or any(
                        asset in assets and dates[i] <= hi and end >= lo
                        for assets, lo, hi in exclusions
                    ):
                        continue
                    mu, cov = distribution(x, anchor, [h])
                    scales = normal_scales(mu, cov)
                    y = vals[target, ai : ai + 1]
                    for seed in map(int, a.seeds.split(",")):
                        for cfg in cfgs:
                            changes = logx if cfg.get("logscale") else x
                            c = dict(vol="v9", drift=cfg.get("drift", 0), df=cfg.get("df", 0))
                            c["width"] = float(
                                np.sqrt(1 - cfg.get("context", 0) + cfg.get("context", 0) * ratio)
                            )
                            if cfg.get("floor"):
                                w = np.exp2(-np.arange(len(changes) - 1, -1, -1) / 60)
                                w /= w.sum()
                                sd = np.sqrt(
                                    0.5 * changes.var(0, ddof=1)
                                    + 0.5 * (w @ (changes - w @ changes) ** 2)
                                )[0]
                                c["width"] = max(
                                    c["width"], cfg["floor"] * context_sd * anchor[0] / sd
                                )
                            s = predict(changes, anchor, [h], c, a.draws, seed).reshape(a.draws, -1)
                            rows.append(
                                dict(
                                    origin=str(dates[i].date()),
                                    end=str(end.date()),
                                    asset=asset,
                                    gap=gap,
                                    horizon=h,
                                    seed=seed,
                                    config=cfg["id"],
                                    **score(s, y, scales),
                                )
                            )
        if counter % 10 == 0:
            pd.DataFrame(rows).to_csv(a.out / "checkpoint.csv", index=False)
            print(counter, len(rows), flush=True)
    f = pd.DataFrame(rows)
    f.to_csv(a.out / "cases.csv", index=False)
    f.groupby(["config", "asset"]).loss.mean().unstack().to_csv(a.out / "assets.csv")
    s = f.groupby(["config", "asset"]).loss.mean().groupby("config").mean().sort_values()
    s.to_csv(a.out / "summary.csv")
    print(s.to_string(), flush=True)


if __name__ == "__main__":
    main()
