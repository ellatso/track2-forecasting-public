"""## Executive summary (read this first)

Check fixed candidates on Monday historical origins, excluding published ps1 weeks.
These periods have previously been explored and are robustness diagnostics.
All forecasts use only the available prefix; no future outcome selects a policy.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from .final_daily import panels
from .final_study import one_group


def group_run(task):
    name, frame, origins, cfgs, practice = task
    return one_group((name, frame, origins, cfgs, 4096, [0, 17, 41], [1, 2, 3, 4], practice))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--official", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    frames, hashes = panels(a.raw, pd.Timestamp("2026-10-09"), extended=True)
    # Freeze candidate definitions before reading the historical target weeks.
    cfgs = [
        dict(id="production", production=True),
        dict(id="sample_zero", vol="sample", drift=0),
        dict(id="mix25", production=True, mixture=0.25),
        dict(id="mix50", production=True, mixture=0.5),
    ]
    origins = pd.date_range("2017-01-02", "2026-10-05", freq="4W-MON")
    import tomllib

    excluded = []
    for cp in (a.official / "units").glob("*-ps1/card.toml"):
        card = tomllib.loads(cp.read_text())
        cutoff = pd.Timestamp(card["provenance"]["data_cutoff"])
        excluded.append((cutoff, cutoff + pd.offsets.BDay(4)))
    groups = ["rates_bills", "rates_curve", "fx_major", "fx_commodity", "fx_em", "fx_em_asia"]
    plan = dict(
        configs=cfgs,
        seeds=[0, 17, 41],
        draws=4096,
        sources=hashes,
        start="2017-01-02",
        end="2026-10-05",
        weekday="Monday",
        exclude_ps1=True,
        selection_source="ps1 initial comparison",
        limitation="Explored historical periods; not an untouched holdout.",
    )
    (a.out / "plan.json").write_text(json.dumps(plan, indent=2))
    tasks = [(g, frames[g], origins[origins.isin(frames[g].index)], cfgs, excluded) for g in groups]
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = [r for result in pool.map(group_run, tasks) for r in result]
    data = pd.DataFrame(rows)
    data["period"] = (
        data.origin.str[:4].astype(int).map(lambda y: "2017-2021" if y < 2022 else "2022-2026")
    )
    data.to_csv(a.out / "cases.csv", index=False)
    print(data.groupby(["period", "config"]).loss.mean().to_string(), flush=True)
