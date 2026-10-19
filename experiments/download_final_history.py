"""## Executive summary (read this first)

Download official Treasury par-curve XML for private retrospective research.
These data never enter the submitted image. ECB history is downloaded separately.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def fetch(task):
    root, year = task
    url = (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
        f"pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value={year}"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    if b"NEW_DATE" not in response.content:
        raise ValueError(f"No Treasury observations for {year}")
    (root / f"treasury-{year}.xml").write_bytes(response.content)
    return year, len(response.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(fetch, [(args.out, y) for y in range(2015, 2027)]):
            print(result, flush=True)
