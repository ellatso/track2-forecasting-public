"""## Executive summary (read this first)

Emit the fixed half-drift, 50% variance-mixture daily research candidate through
the competition forecast interface. Preserve monthly baseline sampling and keys.
Use no House calls or learned model artifacts. Write all outputs beside --out.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from . import cli
from .horizons import HorizonMetadataError
from .limits import ParseLimits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forecast", description=__doc__)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--card", type=Path)
    parser.add_argument("--n-draws", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    card_path = args.card or next(
        (p for p in [args.panels / "card.toml", args.panels.parent / "card.toml"] if p.is_file()),
        None,
    )
    if card_path is None:
        raise SystemExit("card.toml is missing")
    card = tomllib.loads(card_path.read_text(encoding="utf-8"))
    targets = card["targets"]
    assets, horizons = list(targets["asset_ids"]), list(targets["horizons"])
    draws = max(
        args.n_draws or 1000,
        1000,
        int(card.get("scoring", {}).get("params", {}).get("n_draws_min", 200)),
    )
    if draws > ParseLimits().max_draws:
        raise SystemExit("draw count exceeds contract ceiling")
    panels = cli._read_panels(args.panels)
    try:
        steps = cli._monthly_inputs(panels, card, card_path, args.asof)
        samples, stats = cli._draw(
            panels,
            assets,
            horizons,
            args.asof,
            draws,
            args.seed,
            target_type=targets.get("target_type", "level"),
            panel_steps=steps,
            drift_factor=0.5,
            variance_mix=0.5,
        )
    except HorizonMetadataError as exc:
        raise SystemExit(str(exc)) from None
    monthly = steps is not None
    if not np.isfinite(samples).all():
        raise SystemExit("nonfinite forecast")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        dict(
            draw=np.repeat(np.arange(draws), len(assets) * len(horizons)),
            asset=np.tile(np.repeat(assets, len(horizons)), draws),
            horizon=np.tile(horizons, draws * len(assets)),
            value=samples.ravel(),
        )
    ).to_parquet(args.out, index=False)
    method = (
        "unchanged monthly baseline" if monthly else "half drift, sample/EWMA60 variance mix 0.5"
    )
    meta = dict(
        unit_id=card["task"]["id"],
        asof=args.asof,
        representation="samples",
        asset_ids=assets,
        horizons=horizons,
        n_draws=draws,
        target=targets.get("target_type", "level"),
        reasoning_applied=False,
        house_requests_attempted=0,
        rationale=dict(file="forecast_rationale.md", method=method),
    )
    (args.out.parent / "forecast_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    text = (
        "# Forecast rationale\n\n## Executive summary (read this first)\n\n"
        f"As-of {args.asof}. {draws} joint samples, seed {args.seed}. Method: {method}.\n\n"
        "Daily forecasts use at most 300 observations per asset, half the mean increment, "
        "and the square root of an equal mixture of sample variance and exponentially "
        "weighted variance with a 60-observation half-life. Empirical correlation is "
        "preserved with a numerical positive-definite repair. All horizons share a "
        "coherent cumulative path. Factor targets sum log(1 + simple return).\n\n"
        "Monthly forecasts retain the original sample variance and full drift, with "
        "explicit observation counts and publication lag. Authored horizon keys are "
        "preserved. Rows after the requested as-of date never enter estimation.\n\n"
        "The text corpus is not used. No model API call or learned model artifact is "
        "used. This is a numeric-only official trial of a fixed research candidate, "
        "not a claim of accuracy from local admissibility gates.\n\n"
        "## Numerical statistics\n\n```json\n" + json.dumps(stats, indent=2) + "\n```\n"
    )
    (args.out.parent / "forecast_rationale.md").write_text(text, encoding="utf-8")
    print(json.dumps(dict(forecast_written=True, draws=draws, seed=args.seed, monthly=monthly)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
