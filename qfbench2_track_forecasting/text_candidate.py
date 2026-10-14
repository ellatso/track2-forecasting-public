"""## Executive summary (read this first)

Run V12, then optionally mix dated, cited House scenarios into half its draws.
Only the organizer's injected model route is called, at most three times per
task. Invalid evidence, replies, unavailable model access or monthly inputs keep
the exact numeric forecast. Local checks validate execution, not model quality.
"""

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from . import agent, candidate, cli, text_scenarios
from .horizons import HorizonMetadataError
from .house import House
from .limits import ParseLimits


def main(argv=None):
    p = argparse.ArgumentParser(prog="forecast", description=__doc__)
    p.add_argument("--panels", type=Path, required=True)
    p.add_argument("--text", type=Path, required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--card", type=Path)
    p.add_argument("--n-draws", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--event-timing", action="store_true", help="Research cited event-date activation"
    )
    p.add_argument("--event-clock", choices=["date", "ecb-ust"], default="date")
    a = p.parse_args(argv)
    if a.event_clock != "date" and not a.event_timing:
        p.error("--event-clock requires --event-timing")
    card_path = a.card or next(
        (x for x in [a.panels / "card.toml", a.panels.parent / "card.toml"] if x.is_file()), None
    )
    if card_path is None:
        raise SystemExit("card.toml is missing")
    card = tomllib.loads(card_path.read_text(encoding="utf-8"))
    targets = card["targets"]
    assets, horizons = list(targets["asset_ids"]), list(targets["horizons"])
    draws = max(
        a.n_draws or 4096,
        4096,
        int(card.get("scoring", {}).get("params", {}).get("n_draws_min", 200)),
    )
    draws = 1 << (draws - 1).bit_length()
    if draws > ParseLimits().max_draws:
        raise SystemExit("draw count exceeds contract ceiling")
    panels = cli._read_panels(a.panels)
    try:
        base, stats, monthly, ledger = candidate.numeric_forecast(
            panels, card, card_path, a.asof, draws, a.seed
        )
    except HorizonMetadataError as exc:
        raise SystemExit(str(exc)) from None
    docs, excluded = agent.read_documents(a.text, a.asof)
    selected, chars = [], 0
    for doc in docs:
        if len(selected) >= 12 or chars + len(doc["text"]) > 24000:
            continue
        selected.append(doc)
        chars += len(doc["text"])
    excluded += len(docs) - len(selected)
    evidence, scenarios = [], []
    house = House()
    summary = {}
    samples = base.copy()
    if selected and not monthly:
        summary, mean, sd, risk = text_scenarios.summarize(
            panels, assets, horizons, a.asof, base, targets
        )
        extract, quantify = text_scenarios.prompts(
            assets,
            horizons,
            a.asof,
            targets.get("target_type", "level"),
            summary,
            selected,
            event_timing=a.event_timing,
            event_clock=a.event_clock,
        )
        evidence = agent.validate_evidence(house.ask(extract), selected, assets)
        if not evidence and house.requests == 1 and not house.failures:
            evidence = agent.validate_evidence(
                house.ask(extract + "\nReturn the exact schema and copy quotations verbatim."),
                selected,
                assets,
            )
        if evidence:
            reply = house.ask(quantify + json.dumps(evidence, ensure_ascii=False))
            scenarios = text_scenarios.validate(
                reply,
                evidence,
                assets,
                horizons,
                asof=a.asof if a.event_timing else None,
                event_clock=a.event_clock,
            )
            if not scenarios and house.requests < 3 and not house.failures:
                reply = house.ask(
                    quantify
                    + json.dumps(evidence, ensure_ascii=False)
                    + "\nUse finite JSON numbers and verified evidence IDs only."
                )
                scenarios = text_scenarios.validate(
                    reply,
                    evidence,
                    assets,
                    horizons,
                    asof=a.asof if a.event_timing else None,
                    event_clock=a.event_clock,
                )
        if scenarios:
            try:
                samples = text_scenarios.apply(
                    base, assets, horizons, summary, mean, sd, risk, scenarios, a.seed
                )
            except ValueError:
                samples, scenarios = base.copy(), []
                house.failures.append("nonfinite scenario; numeric forecast retained")
    if not np.isfinite(samples).all():
        raise SystemExit("nonfinite numeric forecast")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        dict(
            draw=np.repeat(np.arange(draws), len(assets) * len(horizons)),
            asset=np.tile(np.repeat(assets, len(horizons)), draws),
            horizon=np.tile(horizons, draws * len(assets)),
            value=samples.ravel(),
        )
    ).to_parquet(a.out, index=False)
    applied = bool(scenarios) and not np.array_equal(samples, base)
    meta = dict(
        unit_id=card["task"]["id"],
        asof=a.asof,
        representation="samples",
        asset_ids=assets,
        horizons=horizons,
        n_draws=draws,
        target=targets.get("target_type", "level"),
        reasoning_applied=applied,
        house_requests_attempted=house.requests,
        reasoning_skipped_reason=""
        if applied
        else (
            "monthly numeric policy"
            if monthly
            else house.failures[-1]
            if house.failures
            else "no supported nonzero evidence adjustment"
        ),
        rationale=dict(
            file="forecast_rationale.md",
            method="V12 plus evidence-cited House scenarios"
            if applied
            else "exact V12 numeric fallback",
            documents_read=len(selected),
            documents_excluded=excluded,
            verified_evidence_count=len(evidence),
            event_timing_enabled=a.event_timing,
            event_clock=a.event_clock,
        ),
    )
    (a.out.parent / "forecast_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    rationale = (
        "# Forecast rationale\n\n## Executive summary (read this first)\n\n"
        f"As-of {a.asof}; {draws} joint samples; seed {a.seed}. "
        "V12 supplies the numeric distribution. Exactly half of the draw distribution remains "
        "unchanged when cited scenarios are applied. Missing, invalid or unavailable model "
        "evidence gives the exact numerical forecast. Only this task's dated documents and "
        "panels are used. No unit title, ID or hindsight category is sent to the model.\n\n"
        "The conditional peer risk bound applies only to stale FX and requires an evidence-"
        "supported current-regime scenario. Its cap is six baseline standard deviations; "
        "it is not an unconditional volatility floor. Shifts are confidence-shrunk and bounded "
        "to 0.75 risk standard deviations for short horizons and 3 for longer horizons. "
        "Scenario scaling never narrows the numeric uncertainty. Model quotes are checked "
        "against source occurrence; causal interpretations and probabilities are uncertain. "
        "This is an official-trial candidate, without an off-platform live House quality test.\n"
    )
    sections = {
        "Numeric policy": stats,
        "Daily adaptation": ledger,
        "Input-only forecast summary": summary,
        "Verified evidence": evidence,
        "Scenario ledger": scenarios,
        "House status": dict(attempts=house.requests, failures=house.failures),
    }
    for name, value in sections.items():
        rationale += (
            f"\n## {name}\n\n```json\n"
            + json.dumps(value, indent=2, ensure_ascii=False)
            + "\n```\n"
        )
    (a.out.parent / "forecast_rationale.md").write_text(rationale, encoding="utf-8")
    print(
        json.dumps(
            dict(
                forecast_written=True,
                draws=draws,
                reasoning_applied=applied,
                house_requests_attempted=house.requests,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
