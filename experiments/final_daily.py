"""## Executive summary (read this first)

Research the announced 1-4 business-day Final setting using one source per domain.
Read local official ECB CSV and Treasury par-curve XML downloads. Convert ECB
quotes to the organizer's USD directions, never combine sources or forward-fill
missing outcomes. Keep results outside this public checkout. The organizer's
scorer, not a replacement scoring formula, measures every historical forecast.
This diagnostic does not predict a Final rank or handle official release cards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .final_study import one_group

ROOT = Path(__file__).resolve().parents[1]
USD_PER_UNIT = frozenset(["AUD", "EUR", "GBP", "NZD"])
GROUPS = {
    "rates_bills": ["UST_3M", "UST_6M", "UST_1Y"],
    "rates_curve": ["UST_2Y", "UST_5Y", "UST_10Y", "UST_30Y"],
    "fx_major": ["EUR", "GBP", "JPY", "CHF"],
    "fx_commodity": ["AUD", "CAD", "NOK", "NZD"],
    "fx_nordic": ["SEK", "DKK"],
    "fx_em": ["MXN", "BRL", "ZAR"],
    "fx_em_asia": ["KRW", "INR"],
}


def ecb_to_usd(frame):
    """Input is foreign units per EUR; output preserves the published USD directions."""
    d = frame.apply(pd.to_numeric, errors="coerce")
    assets = sorted({a for g, aa in GROUPS.items() if g.startswith("fx") for a in aa})
    out = pd.DataFrame(index=d.index)
    for a in assets:
        if a == "EUR":
            out[a] = d.USD
        elif a in USD_PER_UNIT:
            out[a] = d.USD / d[a]
        else:
            out[a] = d[a] / d.USD
    return out.where(out > 0).sort_index()


def panels(raw, cutoff):
    """No source download takes place in a submitted container or this research runner."""
    with zipfile.ZipFile(raw / "ecb-history.zip") as z, z.open("eurofxref-hist.csv") as f:
        frame = pd.read_csv(f, index_col="Date", parse_dates=True)
    fx = ecb_to_usd(frame)
    namespaces = {
        "a": "http://www.w3.org/2005/Atom",
        "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
        "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
    }
    fields = {
        "BC_3MONTH": "UST_3M", "BC_6MONTH": "UST_6M", "BC_1YEAR": "UST_1Y",
        "BC_2YEAR": "UST_2Y", "BC_5YEAR": "UST_5Y", "BC_10YEAR": "UST_10Y",
        "BC_30YEAR": "UST_30Y",
    }
    rows = []
    files = [raw / "ecb-history.zip"] + sorted(raw.glob("treasury-*.xml"))
    for file in files[1:]:
        tree = ET.parse(file)
        for item in tree.findall("a:entry/a:content/m:properties", namespaces):
            row = {"date": item.find("d:NEW_DATE", namespaces).text[:10]}
            for source, asset in fields.items():
                cell = item.find("d:" + source, namespaces)
                row[asset] = np.nan if cell is None or cell.text is None else float(cell.text)
            rows.append(row)
    rates = pd.DataFrame(rows).set_index("date")
    rates.index = pd.to_datetime(rates.index)
    rates = rates.sort_index()
    if rates.index.has_duplicates or fx.index.has_duplicates:
        raise ValueError("Ambiguous dates in official source downloads")
    frames = {}
    for group, assets in GROUPS.items():
        data = fx if group.startswith("fx") else rates
        frames[group] = data.loc[data.index <= cutoff, assets].dropna()
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    return frames, hashes


def configs(group):
    legacy = dict(id="legacy", vol="v9", drift=0.5)
    if group.startswith("fx"):
        legacy = dict(id="legacy", components=[
            dict(vol="v9", drift=0.5, weight=0.25),
            dict(vol="ew20", drift=0.0, df=8, weight=0.75),
        ])
    rows = [legacy, dict(id="m0", vol="sample", drift=1.0)]
    rows.extend(dict(id=vol + "_zero", vol=vol, drift=0.0) for vol in ["v9", "ew20"])
    rows.extend(
        dict(id=vol + "_t8", vol=vol, drift=0.0, df=8)
        for vol in ["ew20", "ew60", "blend20", "garch"]
    )
    rows.append(dict(id="ew20_ar", vol="ew20", drift=0.0, mean="ar", strength=0.5))
    if group.startswith("fx"):
        rows.append(dict(id="ew20_log_t8", vol="ew20", drift=0.0, df=8, log=True))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--official", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--cutoff", default="2026-10-09")
    p.add_argument("--draws", type=int, default=512)
    p.add_argument("--seeds", default="0")
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--locked", type=Path, help="Frozen per-group candidate ids; legacy also runs")
    args = p.parse_args()
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError("Save diagnostic outcomes and results outside the public checkout")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new output directory; do not overwrite a completed evaluation")
    data, hashes = panels(args.raw, pd.Timestamp(args.cutoff))
    locked = json.loads(args.locked.read_text(encoding="utf-8")) if args.locked else None
    seeds = list(map(int, args.seeds.split(",")))
    origins = pd.bdate_range(args.start, args.end)[::args.stride]
    tasks = []
    for group, frame in data.items():
        rows = configs(group)
        if locked is not None:
            ids = {"legacy", locked[group]}
            rows = [r for r in rows if r["id"] in ids]
            if {r["id"] for r in rows} != ids:
                raise ValueError("Unknown locked candidate")
        excluded = []
        for file in (args.official / "units").glob("*/card.toml"):
            card = tomllib.loads(file.read_text(encoding="utf-8"))
            if not set(card["targets"]["asset_ids"]).intersection(frame.columns):
                continue
            date = pd.Timestamp(str(card["provenance"]["data_cutoff"])[:10])
            excluded.append((
                date - pd.Timedelta(days=7),
                date + pd.offsets.BDay(max(card["targets"]["horizons"])),
            ))
        tasks.append((group, frame, origins, rows, args.draws, seeds, [1, 2, 3, 4], excluded))
    plan = dict(
        start=args.start, end=args.end, data_cutoff=args.cutoff,
        horizons=[1, 2, 3, 4], draws=args.draws, seeds=seeds, stride=args.stride,
        sources=hashes, groups=GROUPS,
        configurations={t[0]: t[3] for t in tasks},
        split_boundary_rule="origin and every target date must belong to the requested period",
        source_rule="ECB cross rates only for FX; Treasury daily par curve only for yields",
        missing_dates="Drop absent holiday target dates; no forward filling",
        rankable=False, macro_release_cards="NOT covered",
    )
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for block in pool.map(one_group, tasks):
            rows.extend(block)
    f = pd.DataFrame(rows)
    # No target from the next split may be counted in the preceding split.
    f = f[(f.origin >= args.start) & (f.end <= args.end)]
    if f.empty:
        raise ValueError("No evaluable cases")
    f.to_csv(out / "cases.csv", index=False)
    f.groupby(["config", "group"])[
        ["loss", "joint_loss", "single_loss", "marginal", "joint", "tail"]
    ].mean().to_csv(out / "summary.csv")
    f.groupby(["config", "group", f.origin.str[:4]]).loss.mean().to_csv(out / "years.csv")
    print("COMPLETE", len(f), "scores", f[["group", "origin"]].drop_duplicates().shape[0],
          "cases", flush=True)


if __name__ == "__main__":
    main()
