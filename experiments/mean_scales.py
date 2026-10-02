"""## Executive summary (read this first)

Audit mean-model rankings with one pooled reference scale per identical target grid.
The primary per-case scale can be nearly zero; preserve it and add sensitivity only.
Pool current-baseline losses across already explored origins/seeds, then delegate
every composite calculation to the existing scorer. This is post-hoc diagnostics,
not organizer M0 or pre-origin calibration. Keep all frozen methods and cases.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qfbench2_track_forecasting import scoring

from . import batch
from . import frequency as f
from . import means as m


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.run_dir.resolve()
    if root == batch.ROOT or batch.ROOT in root.parents:
        parser.error("Keep results outside GitHub")
    if (root / "scale_sensitivity_plan.json").exists():
        parser.error("Sensitivity already started; inspect existing results")
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if any(
        batch.sha(batch.ROOT / p) != h for p, h in plan["code_sha256"].items() if p != "shared_crps"
    ) or any(batch.sha(batch.ROOT / p) != h for p, h in plan["input_sha256"].items()):
        raise ValueError("Primary inputs or source changed")
    items, _, _ = batch.inventory(batch.ROOT)
    by_group = {i["group"]: i for i in items}
    f.write_json(
        root / "scale_sensitivity_plan.json",
        dict(
            created_utc=datetime.now(UTC).isoformat(),
            primary_plan_sha256=batch.sha(root / "plan.json"),
            source_sha256=batch.sha(Path(__file__)),
            case_sha256={
                name: batch.sha(root / (name + "_cases.csv")) for name in ["daily", "monthly"]
            },
            warning=(
                "Post-hoc fixed-scale diagnostic only; full explored data calibrate "
                "scoring scales, never forecast inputs. Not official M0."
            ),
        ),
    )
    decisions = {}
    for frequency, monthly in [("daily", False), ("monthly", True)]:
        frame = pd.read_csv(root / (frequency + "_cases.csv"))
        reference = frame[frame.config == m.BASE]
        scales = reference.groupby("group")[["marginal", "joint", "tail"]].mean()
        scales.to_csv(root / (frequency + "_pooled_scales.csv"))
        cfgs = {c["id"]: c for c in m.configs(monthly)}
        records = []
        for index, ((group, origin, seed), subset) in enumerate(
            frame.groupby(["group", "origin", "seed"], sort=False)
        ):
            item = by_group[group]
            history, outcome, actual, end = batch.case_at(item, origin)
            history = np.ascontiguousarray(history, dtype=float)
            if actual != origin or not (subset.outcome_end == end).all():
                raise ValueError("Replay case changed")
            base = m.predict(
                history,
                item["steps"],
                item["returns"],
                item["family"],
                monthly,
                cfgs[m.BASE],
                plan["draws"],
                int(seed),
            )
            baseline = batch.raw_score(
                base, outcome, item["card"], item["assets"], item["horizons"]
            )
            weights = tuple(baseline["weights_effective"])
            scale = {
                k: float(scales.loc[group, k]) if w else 1.0
                for k, w in zip(["marginal", "joint", "tail"], weights, strict=True)
            }
            if any(not np.isfinite(v) or v <= 0 for v in scale.values()):
                raise ValueError("Degenerate pooled scale")
            params = item["card"].get("scoring", {}).get("params", {})
            old = m.baseline_mean(history, item["steps"], item["returns"], monthly).ravel()
            for row in subset.to_dict("records"):
                cfg = cfgs[row["config"]]
                new = m.forecast_mean(history, item["steps"], item["returns"], monthly, cfg)
                samples = base if cfg["model"] == "baseline" else base + (new - old)[None, :]
                metric = scoring._composite(
                    samples,
                    outcome,
                    weights=weights,
                    tail_levels=tuple(params.get("tail_levels", (0.01, 0.05, 0.95, 0.99))),
                    joint=scoring.card_joint_statistic(item["card"]),
                    tail_metric=params.get("tail_metric", "pinball"),
                    ref_scale=scale,
                )
                for key in ["marginal", "joint", "tail"]:
                    if not np.isclose(metric[key], row[key], rtol=1e-9, atol=1e-10):
                        raise ValueError("Replay raw loss changed")
                for key, weight in zip(["marginal", "joint", "tail"], weights, strict=True):
                    row[key + "_ratio"] = metric[key] / scale[key] if weight else 0.0
                    row[key + "_contribution"] = weight * row[key + "_ratio"]
                    row[key + "_baseline"] = scale[key]
                records.append(dict(row, **metric))
            if index % 100 == 0:
                print(f"{frequency}: replayed {index+1} case/seeds", flush=True)
        stable = pd.DataFrame(records)
        if len(stable) != len(frame):
            raise ValueError("Replay coverage changed")
        stable.to_csv(root / ("stable_" + frequency + "_cases.csv"), index=False)
        summary = batch.summarize(stable)
        summary.to_csv(root / ("stable_" + frequency + "_summary.csv"), index=False)
        tables = []
        for year, part in stable.groupby(stable.origin.str[:4]):
            table = batch.summarize(part)
            table.insert(0, "year", year)
            tables.append(table)
        pd.concat(tables).to_csv(root / ("stable_" + frequency + "_by_year.csv"), index=False)
        batch.save_diagnostics(stable, root, "stable_" + frequency)
        compatible = stable.copy()
        compatible.loc[compatible.config == m.BASE, "config"] = batch.BASE
        f.write_json(
            root / ("stable_" + frequency + "_paired.json"),
            {
                c: batch.paired_interval(compatible, c)
                for c in stable.config.unique()
                if c != m.BASE
            },
        )
        decisions[frequency] = dict(winner=str(summary.iloc[0].config), promotion=False)
        print(summary.to_string(index=False), flush=True)
    f.write_json(root / "scale_sensitivity_decision.json", decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
