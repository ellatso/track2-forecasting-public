"""## Executive summary (read this first)

Validate the frozen missing-middle policy through the actual production entry
functions, with the exact trailing-window anchor treatment. Save historical
scores privately; this script never writes a practice answer into the image.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qfbench2_track_forecasting import cli, scoring
from qfbench2_track_forecasting.daily_candidate import estimate

from .final_study import frames
from .m0_scale import distribution, normal_scales


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = frames(args.official)
    frame = pd.concat([data[g][0] for g in ["fx_major", "fx_other", "fx_nordic"]], axis=1).dropna()
    dates = frame.index
    vals = frame.to_numpy()
    rows = []
    h = [2, 5, 10, 12]
    for stage, start, end in [
        ("train", "2007-01-05", "2014-12-02"),
        ("confirm", "2015-01-05", "2019-12-02"),
        ("test", "2020-01-06", "2024-10-01"),
    ]:
        for counter, origin in enumerate(pd.bdate_range(start, end)[::70]):
            i = dates.searchsorted(origin, side="right") - 1
            target = dates.get_indexer([dates[i] + pd.offsets.BDay(k) for k in h])
            if np.any(target < 0):
                continue
            for gap in [2, 5, 10]:
                old = dates.searchsorted(dates[i] - pd.DateOffset(years=gap), side="right") - 1
                if old < 300:
                    continue
                a = counter % 10
                asset = frame.columns[a]
                selected = frame.iloc[: i + 1].copy()
                selected.loc[(selected.index > dates[old]) & (selected.index < dates[i]), asset] = (
                    np.nan
                )
                panel = (
                    selected.rename_axis("date")
                    .reset_index()
                    .melt(id_vars="date", var_name="asset", value_name="value")
                    .dropna()
                )
                panels = {"fx": panel}
                asof = str(dates[i].date())
                hist = cli._series(panels, asset, asof).iloc[-300:]
                changes = cli._diff_without_gaps(hist).dropna().to_numpy()[:, None]
                mean, cov = distribution(changes, np.array([hist.iloc[-1]]), h)
                scale = normal_scales(mean, cov)
                y = vals[target, a]
                for seed in [0, 17]:
                    base, _ = cli._draw(
                        panels,
                        [asset],
                        h,
                        asof,
                        4096,
                        seed,
                        drift_factor=0.5,
                        variance_mix=0.5,
                        sampling="sobol",
                    )
                    new, ledger = estimate(panels, [asset], h, asof, 4096, seed)
                    for name, s in [("v9", base), ("context50", new)]:
                        metric = scoring._composite(
                            s.reshape(4096, -1),
                            y,
                            weights=(0.5, 0.3, 0.2),
                            tail_levels=(0.01, 0.05, 0.95, 0.99),
                            joint="variogram",
                            tail_metric="pinball",
                            ref_scale=scale,
                        )
                        rows.append(
                            dict(
                                stage=stage,
                                origin=asof,
                                gap=gap,
                                asset=asset,
                                seed=seed,
                                config=name,
                                loss=min(metric["composite"], 8.0),
                                context_ratio=ledger["context_variance_ratio"],
                            )
                        )
        pd.DataFrame(rows).to_csv(args.out / "cases.csv", index=False)
        print(
            stage,
            pd.DataFrame(rows).groupby(["stage", "config"]).loss.mean().to_string(),
            flush=True,
        )
    print("cases", len(rows) // 4)


if __name__ == "__main__":
    main()
