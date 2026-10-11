"""## Executive summary (read this first)

Stress-test missing-middle FX histories at short horizons. The target supplies
only 300 old observations and the current anchor. Other series supply only their
own current and old prefixes. The target future is passed only to the scorer.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qfbench2_track_forecasting import scoring
from qfbench2_track_forecasting.short_horizon import ew_cov, predict

from .final_study import frames
from .m0_scale import distribution, normal_scales


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", default="2007-01-05")
    p.add_argument("--end", default="2014-12-02")
    p.add_argument("--official", type=Path, required=True)
    p.add_argument("--gap", type=int, default=5)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = frames(args.official)
    frame = pd.concat([data[g][0] for g in ["fx_major", "fx_other", "fx_nordic"]], axis=1).dropna()
    vals = frame.to_numpy()
    dates = frame.index
    h = [2, 5, 10, 12]
    rows = []
    for counter, o in enumerate(pd.bdate_range(args.start, args.end)[::35]):
        i = dates.searchsorted(o, side="right") - 1
        old = dates.searchsorted(dates[i] - pd.DateOffset(years=args.gap), side="right") - 1
        end = dates.get_indexer([dates[i] + pd.offsets.BDay(k) for k in h])
        if old < 300 or np.any(end < 0):
            continue
        for a in [counter % 10, (counter + 5) % 10]:
            past = vals[old - 299 : old + 1, a : a + 1]
            x = np.diff(past, axis=0)
            anchor = vals[i, a : a + 1]
            y = vals[end, a]
            mean, cov = distribution(x, anchor, h)
            scale = normal_scales(mean, cov)
            c = np.arange(10) != a
            early = np.diff(np.log(vals[old - 299 : old + 1, c]), axis=0)
            recent = np.diff(np.log(vals[i - 299 : i + 1, c]), axis=0)
            ratio = np.median(
                np.diag(ew_cov(recent, 20)) / np.maximum(np.diag(ew_cov(early, 20)), 1e-12)
            )
            ratio = float(np.clip(ratio, 0.25, 4.0))
            for name in ["v9", "zero", "logscale", "context25", "context50", "context75"]:
                cfg = dict(vol="v9", drift=0.5 if name == "v9" else 0.0)
                changes = x
                if name not in ["v9", "zero"]:
                    changes = np.diff(np.log(past), axis=0) * anchor
                if name.startswith("context"):
                    weight = int(name[7:]) / 100
                    cfg["width"] = np.sqrt((1 - weight) + weight * ratio)
                s = predict(changes, anchor, h, cfg, 4096, 0).reshape(4096, -1)
                m = scoring._composite(
                    s,
                    y,
                    weights=(0.5, 0.3, 0.2),
                    tail_levels=(0.01, 0.05, 0.95, 0.99),
                    joint="variogram",
                    tail_metric="pinball",
                    ref_scale=scale,
                )
                rows.append(
                    dict(
                        origin=str(dates[i].date()),
                        asset=frame.columns[a],
                        config=name,
                        loss=min(m["composite"], 8.0),
                    )
                )
    f = pd.DataFrame(rows)
    f.to_csv(args.out / "cases.csv", index=False)
    print(f.groupby("config").loss.mean().sort_values().to_string())
    print("cases", len(f) // 6)


if __name__ == "__main__":
    main()
