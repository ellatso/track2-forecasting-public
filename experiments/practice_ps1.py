"""## Executive summary (read this first)

Freeze a small comparison before loading the 22 published ps1 outcomes.
Forecasts read each card's own cutoff-filtered inputs; scoring alone reads answers.
Use the organizer's gates and composite, plus input-only documented M0 divisors.
Keep all predictions, outcomes and measured results outside this public checkout.
"""

import argparse
import hashlib
import json
import platform
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from qfbench2_track_forecasting import candidate, cli, scoring
from qfbench2_track_forecasting.grid import grid_from_card
from qfbench2_track_forecasting.normalization import NormalizationMode
from qfbench2_track_forecasting.short_horizon import innovations, predict

from .m0_scale import distribution, normal_scales
from .production_audit import couple, prequential_width

ROOT = Path(__file__).resolve().parents[1]
POLICIES = (
    "production",
    "online_bills",
    "online_all",
    "corr_ew60",
    "corr_shrink50",
    "sample_zero",
    "ew20_t8_zero",
)
SEEDS = (0, 17, 41)
DRAWS = 4096


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transformed(base, hist, changes, horizons, policy, seed):
    """Fixed input-only policies, independent of card identifiers and answers."""
    if policy not in POLICIES + ("mix25", "mix50"):
        raise ValueError("Unknown comparison policy")
    if policy == "production":
        return base
    if policy in ("mix25", "mix50"):
        other = predict(
            changes, hist.iloc[-1].to_numpy(), horizons, dict(vol="sample", drift=0.0), DRAWS, seed
        )
        _, _, selector = innovations(DRAWS, len(hist.columns), len(horizons), seed + 104729, 0)
        output = base.copy()
        selected = selector < (0.25 if policy == "mix25" else 0.5)
        output[selected] = other[selected]
        return output
    if policy.startswith("online"):
        bills = all(a in {"UST_3M", "UST_6M", "UST_1Y", "UST_2Y"} for a in hist)
        if policy == "online_bills" and not bills:
            return base
        width = prequential_width(hist)[None, :, None]
        centre = base.mean(0, keepdims=True)
        return centre + width * (base - centre)
    cfg = dict(vol="sample", drift=0.0)
    if policy.startswith("corr_"):
        cfg.update({"corr": 60} if policy == "corr_ew60" else {"shrink": 0.5})
        template = predict(changes, np.zeros(len(hist.columns)), horizons, cfg, DRAWS, seed)
        return couple(base, template)
    if policy == "ew20_t8_zero":
        cfg.update(vol="ew20", df=8)
    return predict(changes, hist.iloc[-1].to_numpy(), horizons, cfg, DRAWS, seed)


def write_output(out, samples, card, asof, policy, seed):
    out.mkdir(parents=True, exist_ok=False)
    assets, horizons = card["targets"]["asset_ids"], card["targets"]["horizons"]
    pd.DataFrame(
        dict(
            draw=np.repeat(np.arange(DRAWS), len(assets) * len(horizons)),
            asset=np.tile(np.repeat(assets, len(horizons)), DRAWS),
            horizon=np.tile(horizons, DRAWS * len(assets)),
            value=samples.ravel(),
        )
    ).to_parquet(out / "forecast.parquet", index=False)
    (out / "forecast_meta.json").write_text(
        json.dumps(
            dict(
                unit_id=card["task"]["id"],
                asof=asof,
                representation="samples",
                asset_ids=assets,
                horizons=horizons,
                n_draws=DRAWS,
                target="level",
                reasoning_applied=False,
                house_requests_attempted=0,
                rationale=dict(file="forecast_rationale.md", method=policy),
            )
        ),
        encoding="utf-8",
    )
    (out / "forecast_rationale.md").write_text(
        "# Executive summary (read this first)\n\n"
        f"Input-only numeric research policy {policy}; seed {seed}; as-of {asof}. "
        "Only this unit's history through the cutoff enters estimation. "
        "No published answers or cross-card target lookup enters prediction. "
        "Text predictive quality is not tested by this comparison.\n",
        encoding="utf-8",
    )


def run(official, out):
    if out == ROOT or ROOT in out.parents:
        raise ValueError("Results must remain outside the public checkout")
    out.mkdir(parents=True, exist_ok=False)
    units = sorted((official / "units").glob("*-ps1"))
    if len(units) != 22:
        raise ValueError(f"Expected 22 ps1 units, got {len(units)}")
    code = {
        str(p.relative_to(ROOT)): sha(p)
        for p in sorted(
            list((ROOT / "qfbench2_track_forecasting").glob("*.py"))
            + [
                Path(__file__),
                ROOT / "experiments/m0_scale.py",
                ROOT / "experiments/production_audit.py",
            ]
        )
    }
    plan = dict(
        policies=POLICIES,
        seeds=SEEDS,
        draws=DRAWS,
        python=platform.python_version(),
        code=code,
        inputs={
            str(p.relative_to(official)): sha(p) for u in units for p in u.rglob("*") if p.is_file()
        },
        design="All settings frozen before forecast generation and outcome reading. "
        "22 retrospective cards; correlated cards are not independent trials. "
        "No subsequent tuning against this set establishes new holdout evidence.",
    )
    (out / "frozen-plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    records = []
    # Complete and seal every prediction before opening the published outcomes.
    for unit in units:
        card = tomllib.loads((unit / "card.toml").read_text())
        assets, horizons = card["targets"]["asset_ids"], card["targets"]["horizons"]
        asof = str(card["provenance"]["data_cutoff"])[:10]
        panels = cli._read_panels(unit / "panels")
        series = {a: cli._series(panels, a, asof) for a in assets}
        hist = pd.DataFrame(series).dropna()
        hist.index = pd.to_datetime(hist.index)
        changes = (
            pd.DataFrame({a: cli._diff_without_gaps(s.iloc[-300:]) for a, s in series.items()})
            .dropna()
            .to_numpy()
        )
        anchor = np.array([series[a].iloc[-1] for a in assets])
        m, cov = distribution(changes, anchor, horizons)
        scales = normal_scales(m, cov)
        for seed in SEEDS:
            started = time.monotonic()
            base, *_ = candidate.numeric_forecast(
                panels, card, unit / "card.toml", asof, DRAWS, seed
            )
            for policy in POLICIES:
                samples = transformed(base, hist, changes, horizons, policy, seed)
                folder = out / "forecasts" / unit.name / f"{policy}-s{seed}"
                write_output(folder, samples, card, asof, policy, seed)
                records.append(
                    (unit, card, asof, policy, seed, folder, scales, time.monotonic() - started)
                )
        print("forecast sealed", unit.name, flush=True)
    sealed = {
        str(p.relative_to(out)): sha(p) for p in (out / "forecasts").rglob("*") if p.is_file()
    }
    (out / "sealed-forecasts.json").write_text(json.dumps(sealed, indent=2))
    rows, details, hrows = [], [], []
    for unit, card, asof, policy, seed, folder, scales, elapsed in records:
        spec = grid_from_card(card)
        realized = scoring._load_realized_vector(
            official / "practice-answers" / (unit.name + ".parquet"), spec
        )
        ctx = dict(
            card=card,
            output_dir=folder,
            unit_dir=unit,
            unit_handle=unit.name,
            expected_grid=spec,
            grid_source="card",
            realized=realized,
            normalization_mode=NormalizationMode.RAW_UNRANKABLE,
            ref_scale=None,
        )
        verdict = scoring.build_verifier(ctx).run(ctx)
        details.append(
            dict(
                unit=unit.name,
                policy=policy,
                seed=seed,
                passed=verdict.admissible,
                detail=verdict.detail,
                gates={n: v.passed for n, v in verdict.gate_results.items()},
            )
        )
        samples = pd.read_parquet(folder / "forecast.parquet").value.to_numpy().reshape(DRAWS, -1)
        weights = (5 / 7, 0.0, 2 / 7) if spec.cell_count == 1 else (0.5, 0.3, 0.2)
        params = dict(
            weights=weights,
            tail_levels=(0.01, 0.05, 0.95, 0.99),
            joint="variogram",
            tail_metric="pinball",
            ref_scale=scales,
        )
        result = scoring._composite(samples, realized, **params)
        row = dict(
            unit=unit.name,
            policy=policy,
            seed=seed,
            asof=asof,
            group="bills"
            if all(a.startswith("UST_") for a in spec.assets)
            else "em"
            if any(a in {"MXN", "BRL", "ZAR", "KRW", "INR"} for a in spec.assets)
            else "g10",
            family=card["metadata"]["category"],
            passed=verdict.admissible,
            loss=min(result["composite"], 8) if verdict.admissible else 8,
            marginal=result["marginal"] / scales["marginal"],
            joint=result["joint"] / scales["joint"],
            tail=result["tail"] / scales["tail"],
            forecast_seconds=elapsed,
        )
        for coverage in (50, 80, 95):
            alpha = (1 - coverage / 100) / 2
            lo, hi = np.quantile(samples, [alpha, 1 - alpha], axis=0)
            row[f"coverage{coverage}"] = float(((lo <= realized) & (realized <= hi)).mean())
        rows.append(row)
        for j, (asset, horizon) in enumerate(spec.cells()):
            single = scoring._composite(
                samples[:, j : j + 1],
                realized[j : j + 1],
                weights=(5 / 7, 0, 2 / 7),
                tail_levels=(0.01, 0.05, 0.95, 0.99),
                joint="variogram",
                tail_metric="pinball",
                ref_scale=dict(marginal=scales["marginal"], joint=1, tail=scales["tail"]),
            )
            hrows.append(
                dict(
                    unit=unit.name,
                    policy=policy,
                    seed=seed,
                    asset=asset,
                    horizon=horizon,
                    marginal=single["marginal"],
                    tail=single["tail"],
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "card-results.csv", index=False)
    pd.DataFrame(hrows).to_csv(out / "cell-results.csv", index=False)
    (out / "gate-details.json").write_text(json.dumps(details, indent=2, default=str))
    frame.groupby("policy").mean(numeric_only=True).to_csv(out / "summary.csv")
    frame.groupby(["group", "policy"]).mean(numeric_only=True).to_csv(out / "groups.csv")
    print(frame.groupby("policy").loss.mean().sort_values().to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mixtures",
        action="store_true",
        help="Include exploratory mixtures motivated by the initial comparison",
    )
    args = parser.parse_args()
    if args.mixtures:
        POLICIES = POLICIES + ("mix25", "mix50")
    run(args.official.resolve(), args.out.resolve())
