"""## Executive summary (read this first)

Read dated BLS releases in this unit's corpus. Add a consecutive missing monthly
observation for unemployment or seasonally adjusted CPI. Never read another unit,
look up an outcome, call a model or use a release after the requested cutoff.
Ambiguous documents leave the panel unchanged. Quotes and dates form an audit log.
"""

import calendar
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import cli

MONTHS = {name.upper(): number for number, name in enumerate(calendar.month_name) if name}
TITLE = re.compile(r"(CONSUMER PRICE INDEX|THE EMPLOYMENT SITUATION)\s*[-–—]+\s*([A-Z]+)\s+(\d{4})")
VERB = r"(rose|increased|fell|decreased|declined|edged up|edged down)"


def parse(text):
    """Return only unambiguous first-release figures with a reference month."""
    flat = re.sub(r"\s+", " ", text)
    header = TITLE.search(flat[:4000])
    if header is None or header[2] not in MONTHS:
        return None
    month = pd.Period(year=int(header[3]), month=MONTHS[header[2]], freq="M")
    body = flat[header.end() : header.end() + 2600]
    values = {}
    if header[1] == "THE EMPLOYMENT SITUATION":
        match = re.search(
            r"(?:The |the )unemployment rate[^.;]{0,60}?(?:to|at) (\d{1,2}\.\d) percent", body[:700]
        )
        if match and 1 <= float(match[1]) <= 30:
            values["UNRATE"] = dict(value=float(match[1]), kind="level", quote=match[0])
    else:
        patterns = {
            "CPI_ALL": r"\(CPI-U\)\s+"
            + VERB
            + r"\s+(?:by\s+)?(\d+(?:\.\d+)?) percent in "
            + calendar.month_name[month.month]
            + r" on a seasonally adjusted basis",
            "CPI_CORE": r"[Tt]he index for all items less food and energy\s+"
            + VERB
            + r"\s+(?:by\s+)?(\d+(?:\.\d+)?) percent in "
            + calendar.month_name[month.month],
        }
        for asset, pattern in patterns.items():
            match = re.search(pattern, body)
            if match:
                change = float(match[2]) * (
                    -1 if match[1] in {"fell", "decreased", "declined", "edged down"} else 1
                )
                if abs(change) <= 3:
                    values[asset] = dict(value=change, kind="monthly_percent", quote=match[0])
    return month, values


def augment(panels, assets, steps, text_dir, asof):
    """Use only this unit's timestamped corpus; never overwrite existing history."""
    index = Path(text_dir) / "corpus_index.json"
    if steps is None or not index.is_file():
        return panels, []
    cutoff = pd.Timestamp(asof).date()
    records = []
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return panels, []
        documents = payload.get("documents", [])
        if not isinstance(documents, list):
            return panels, []
    except (OSError, ValueError):
        return panels, []
    for doc in documents:
        if not isinstance(doc, dict) or doc.get("doc_type") != "macro_release":
            continue
        if str(doc.get("source", "")).strip().upper() != "BLS":
            continue
        try:
            stamp = pd.Timestamp(doc.get("timestamp"))
            if pd.isna(stamp) or stamp.date() > cutoff:
                continue
            name = doc.get("file", "")
            root = Path(text_dir).resolve()
            path = (root / name).resolve()
            if root not in path.parents or not path.is_file() or path.stat().st_size > 2_000_000:
                continue
            parsed = parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, TypeError, ValueError):
            continue
        if parsed is not None:
            month, values = parsed
            if month.end_time.date() <= stamp.date():
                records.append((month, stamp.date(), name, values))
    result = dict(panels)
    ledger = []
    for ai, asset in enumerate(assets):
        if asset not in {"UNRATE", "CPI_ALL", "CPI_CORE"}:
            continue
        history = cli._monthly_series(cli._series(panels, asset, asof))
        wanted = history.index[-1] + 1
        # Keep all requested forecast endpoints strictly beyond the new observation.
        if np.min(steps[ai]) <= 1:
            continue
        matches = [r for r in records if r[0] == wanted and asset in r[3]]
        if len(matches) != 1:
            continue
        month, published, name, values = matches[0]
        item = values[asset]
        old = float(history.iloc[-1])
        new = item["value"] if item["kind"] == "level" else old * (1 + item["value"] / 100)
        if not np.isfinite(new) or new <= 0:
            continue
        for key, panel in result.items():
            col = cli._asset_col(panel)
            if col and asset in set(panel[col].astype(str)):
                row = panel[panel[col].astype(str) == asset].iloc[-1].copy()
                row["date"] = month.start_time.date()
                row["value"] = new
                result[key] = pd.concat([panel, pd.DataFrame([row])], ignore_index=True)
                ledger.append(
                    dict(
                        asset=asset,
                        observation_month=str(month),
                        publication_date=str(published),
                        source_file=name,
                        quote=item["quote"],
                        interpretation=item["kind"],
                        previous_anchor=old,
                        new_anchor=new,
                    )
                )
                break
    return result, ledger
