"""## Executive summary (read this first)

Freeze a private, event-blocked three-policy comparison before opening outcomes.
Stage only checksum-verified input files; answers never enter model snapshots.
Seal forecast bytes before scoring through the unchanged official gate stack.
Missing participant outputs count eight, while missing references stop evaluation.
Unavailable/mock House runs are diagnostics and cannot select a text policy.
This local research protocol is not an official ranking or Final uploader.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import shutil
import sys
import tomllib
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from qfbench2_track_forecasting import candidate, scoring, text_candidate
from qfbench2_track_forecasting.grid import GridSpec, grid_from_card
from qfbench2_track_forecasting.normalization import NormalizationMode, _parse_ref_scale

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("numeric", "previous_text", "timed_text")
PHASES = ("select", "confirm", "test")
OUTPUTS = ("forecast.parquet", "forecast_meta.json", "forecast_rationale.md")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_hashes(folder):
    result = {}
    for p in folder.rglob("*"):
        key = str(p.relative_to(folder))
        if p.is_symlink():
            result[key] = "link:" + str(p.readlink())
        elif p.is_file():
            result[key] = sha(p)
        else:
            result[key] = "directory"
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def private_path(path):
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("Private research inputs/results must stay outside the public checkout")
    return path


def fingerprint():
    files = list((ROOT / "qfbench2_track_forecasting").glob("*.py")) + [Path(__file__)]
    return dict(
        code={str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
        python=sys.version,
        packages={
            n: importlib.metadata.version(n)
            for n in ["numpy", "pandas", "scipy", "pyarrow", "qfbench2-common"]
        },
    )


def allowed_inputs(unit):
    manifest = json.loads((unit / "manifest.json").read_text(encoding="utf-8"))
    keep = []
    for item in manifest["files"]:
        rel = PurePosixPath(item["path"])
        if rel.is_absolute() or ".." in rel.parts or "\\" in str(rel):
            raise ValueError("Unsafe manifest path")
        if item.get("role") not in ("input", "fixture", "corpus"):
            continue
        if any(p in ("reference", "realized", "solution") for p in rel.parts):
            continue
        if rel.name == "realized.parquet" or rel.name.startswith(("answer_key", "oracle_")):
            continue
        if item["role"] == "fixture" and str(rel) not in (
            "card.toml",
            "forecast_spec.json",
            "forecast_card.md",
        ):
            continue
        file = unit.joinpath(*rel.parts)
        if any(p.is_symlink() for p in [file, *file.parents] if p != unit.parent):
            raise ValueError("Linked input is not a standalone snapshot")
        if not file.is_file() or sha(file) != item["sha256"]:
            raise ValueError("Input checksum mismatch")
        keep.append(item)
    if "card.toml" not in {x["path"] for x in keep}:
        raise ValueError("A manifested card is required")
    return manifest, keep


def event_splits(units):
    """Join overlapping future windows and explicit shared-event labels transitively."""
    parent = list(range(len(units)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(units):
        for j, b in enumerate(units[:i]):
            overlaps = max(a["start"], b["start"]) <= min(a["end"], b["end"])
            shared = a.get("event") and a.get("event") == b.get("event")
            if overlaps or shared:
                parent[find(i)] = find(j)
    # A shared event can span two separated windows. Merge every intervening
    # block too, so chronological training can never extend past a later split.
    changed = True
    while changed:
        changed = False
        envelopes = {}
        for i, u in enumerate(units):
            envelopes.setdefault(find(i), []).append(u)
        keys = list(envelopes)
        for pos, i in enumerate(keys):
            a = envelopes[i]
            for j in keys[:pos]:
                b = envelopes[j]
                if max(min(u["start"] for u in a), min(u["start"] for u in b)) <= min(
                    max(u["end"] for u in a), max(u["end"] for u in b)
                ) and find(i) != find(j):
                    parent[find(i)] = find(j)
                    changed = True
    blocks = {}
    for i, unit in enumerate(units):
        blocks.setdefault(find(i), []).append(unit)
    groups = sorted(blocks.values(), key=lambda g: min(u["start"] for u in g))
    if len(groups) < 3:
        raise ValueError("Need at least three complete event/date blocks for the three phases")
    cuts = np.array_split(np.arange(len(groups)), 3)
    assigned = {}
    for phase, indices in zip(PHASES, cuts, strict=True):
        for gi in indices:
            for unit in groups[int(gi)]:
                assigned[unit["handle"]] = dict(phase=phase, block=int(gi))
    return assigned


def prepare(catalog_path, run_dir, draws=4096, seeds=(0, 17, 41)):
    run = private_path(run_dir)
    if run.exists() and any(run.iterdir()):
        raise ValueError("Use a new comparison directory")
    if (
        not seeds
        or draws < 256
        or draws > 20000
        or draws & (draws - 1)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Use unique seeds and power-of-two draws within 256-20000")
    catalog_path = private_path(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("normalization") != "m0-expected-error":
        raise ValueError(
            "Declare expected-error M0 scales; raw/realized-error scales are incompatible"
        )
    units, sources = [], {}
    for entry in catalog["units"]:
        source = (catalog_path.parent / entry["unit"]).resolve()
        manifest, keep = allowed_inputs(source)
        card = tomllib.loads((source / "card.toml").read_text(encoding="utf-8"))
        ident = card["task"]["id"]
        handle = hashlib.sha256(ident.encode()).hexdigest()[:16]
        cutoff = str(card["provenance"]["data_cutoff"])[:10]
        target = card["targets"]
        end = entry.get("target_end")
        if end is None:
            frequency = target.get(
                "target_frequency", card.get("metadata", {}).get("target_frequency")
            )
            if frequency not in (None, "daily", "business_daily"):
                raise ValueError(
                    "Release/monthly cards require an explicit target_end in the catalog"
                )
            end = str((pd.Timestamp(cutoff) + pd.offsets.BDay(max(target["horizons"]))).date())
        start = str((pd.Timestamp(cutoff) + pd.offsets.BDay(1)).date())
        end = str(pd.Timestamp(end).date())
        if pd.Timestamp(end) < pd.Timestamp(start):
            raise ValueError("Target window ends before it starts")
        references = {}
        for key in ("realized", "scale"):
            path = private_path(catalog_path.parent / entry[key]) if entry.get(key) else None
            references[key] = dict(path=str(path), sha256=sha(path)) if path else None
        units.append(
            dict(
                id=ident,
                handle=handle,
                asof=cutoff,
                start=start,
                end=end,
                event=entry.get("event"),
                references=references,
            )
        )
        sources[handle] = (source, manifest, keep)
    if len({u["handle"] for u in units}) != len(units):
        raise ValueError("Duplicate card ID")
    assignment = event_splits(units)
    run.mkdir(parents=True, exist_ok=True)
    for unit in units:
        unit.update(assignment[unit["handle"]])
        source, manifest, keep = sources[unit["handle"]]
        snapshot = run / "inputs" / unit["handle"]
        for item in keep:
            dest = snapshot / item["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / item["path"], dest)
        filtered = dict(manifest, files=keep)
        write_json(snapshot / "manifest.json", filtered)
        unit["input_hashes"] = {
            str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob("*") if p.is_file()
        }
        unit["input_digest"] = digest(unit["input_hashes"])
    plan = dict(
        schema_version=1,
        units=units,
        variants=VARIANTS,
        seeds=list(seeds),
        draws=draws,
        environment=fingerprint(),
        rankable=False,
        selection_min_gain=0.01,
        component_regression_limit=1.03,
        event_regression_limit=1.05,
        kit=catalog.get("kit_revision", "unspecified"),
    )
    plan["digest"] = digest(plan)
    write_json(run / "plan.json", plan)
    return plan


def load_plan(run_dir):
    run = private_path(run_dir)
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    if plan["digest"] != digest({k: v for k, v in plan.items() if k != "digest"}):
        raise ValueError("Plan changed after it was frozen")
    if plan["environment"] != fingerprint():
        raise ValueError("Code or runtime changed after the plan was frozen")
    for unit in plan["units"]:
        folder = run / "inputs" / unit["handle"]
        now = {str(p.relative_to(folder)): sha(p) for p in folder.rglob("*") if p.is_file()}
        if now != unit["input_hashes"]:
            raise ValueError("A frozen input snapshot changed")
    return run, plan


def smoke_generate(run_dir):
    """Exercise the three entry points with House disabled. Never create quality evidence."""
    run, plan = load_plan(run_dir)
    root = run / "forecasts"
    if root.exists():
        raise ValueError("Forecasts already exist; never overwrite a comparison")
    saved = {
        k: os.environ.pop(k)
        for k in ("MODEL_ENDPOINT", "MODEL_TOKEN", "MODEL_NAME")
        if k in os.environ
    }
    evidence = {}
    try:
        for unit in plan["units"]:
            source = run / "inputs" / unit["handle"]
            for variant in VARIANTS:
                for seed in plan["seeds"]:
                    key = f"{variant}/{seed}/{unit['handle']}"
                    out = root / key / "forecast.parquet"
                    panels = source / "panels" if (source / "panels").is_dir() else source
                    args = [
                        "--panels",
                        str(panels),
                        "--text",
                        str(source / "text"),
                        "--asof",
                        unit["asof"],
                        "--out",
                        str(out),
                        "--seed",
                        str(seed),
                        "--n-draws",
                        str(plan["draws"]),
                    ]
                    entry = candidate.main if variant == "numeric" else text_candidate.main
                    if variant == "timed_text":
                        args += ["--event-timing", "--event-clock", "ecb-ust"]
                    with contextlib.redirect_stdout(io.StringIO()):
                        entry(args)
                    evidence[key] = dict(
                        mode="numeric" if variant == "numeric" else "unavailable",
                        input_digest=unit["input_digest"],
                        successful_replies=0,
                    )
    finally:
        os.environ.update(saved)
    write_json(run / "execution.json", evidence)


def seal(run_dir):
    run, plan = load_plan(run_dir)
    if (run / "sealed.json").exists():
        raise ValueError("Forecast archive is already sealed")
    files = {}
    for unit in plan["units"]:
        for variant in VARIANTS:
            for seed in plan["seeds"]:
                key = f"{variant}/{seed}/{unit['handle']}"
                folder = run / "forecasts" / key
                files[key] = output_hashes(folder)
    files["execution"] = sha(run / "execution.json") if (run / "execution.json").is_file() else None
    write_json(run / "sealed.json", dict(plan_digest=plan["digest"], files=files))


def evaluate(run_dir, phase):
    if phase not in PHASES:
        raise ValueError("Unknown evaluation phase")
    run, plan = load_plan(run_dir)
    archive = json.loads((run / "sealed.json").read_text(encoding="utf-8"))
    if archive["plan_digest"] != plan["digest"]:
        raise ValueError("Forecast archive belongs to another plan")
    execution_path = run / "execution.json"
    current = sha(execution_path) if execution_path.is_file() else None
    if current != archive["files"]["execution"]:
        raise ValueError("Execution evidence changed after forecasts were sealed")
    execution = json.loads(execution_path.read_text(encoding="utf-8")) if current else {}
    if phase == "select":
        variants = list(VARIANTS)
    else:
        previous = "select" if phase == "confirm" else "confirm"
        decision = json.loads((run / f"{previous}-decision.json").read_text(encoding="utf-8"))
        if (
            decision["plan_digest"] != plan["digest"]
            or decision["phase"] != previous
            or decision["digest"] != digest({k: v for k, v in decision.items() if k != "digest"})
        ):
            raise ValueError("The preceding decision changed or belongs to another plan")
        variants = list(dict.fromkeys(["numeric", decision["selected"]]))
    marker = run / f"{phase}-started.json"
    with marker.open("x", encoding="utf-8") as f:
        json.dump(dict(plan=plan["digest"], variants=variants), f)
    rows, horizons_raw = [], []
    for unit in [u for u in plan["units"] if u["phase"] == phase]:
        source = run / "inputs" / unit["handle"]
        card = tomllib.loads((source / "card.toml").read_text(encoding="utf-8"))
        refs = unit["references"]
        for ref in refs.values():
            if ref is None or sha(ref["path"]) != ref["sha256"]:
                raise ValueError("Missing or changed reference: do not drop this card")
        spec = grid_from_card(card)
        y = scoring._load_realized_vector(Path(refs["realized"]["path"]), spec)
        normalizer = _parse_ref_scale(
            Path(refs["scale"]["path"]).read_bytes(),
            cell_count=spec.cell_count,
            joint_statistic=scoring.card_joint_statistic(card),
        )
        scale = normalizer.as_mapping(joint_weight=0.0 if normalizer.joint is None else 1.0)
        for variant in variants:
            for seed in plan["seeds"]:
                key = f"{variant}/{seed}/{unit['handle']}"
                folder = run / "forecasts" / key
                now = output_hashes(folder)
                if now != archive["files"][key]:
                    raise ValueError("Forecast changed after the archive was sealed")
                ctx = dict(
                    card=card,
                    unit_dir=source,
                    output_dir=folder,
                    unit_handle=unit["id"],
                    realized=y,
                    ref_scale=normalizer,
                    normalization_mode=NormalizationMode.REF_SCALE,
                )
                verdict = scoring.build_verifier(ctx).run(ctx)
                ok = verdict.admissible
                meta = {}
                if ok:
                    meta = json.loads((folder / "forecast_meta.json").read_text(encoding="utf-8"))
                    ok = meta["n_draws"] == plan["draws"]
                detail = verdict.detail if ok else {}
                ev = execution.get(key, {})
                genuine = (
                    ev.get("mode") in ("house_live", "house_replay")
                    and ev.get("successful_replies", 0) > 0
                    and ev.get("input_digest") == unit["input_digest"]
                    and meta.get("house_requests_attempted", 0) > 0
                )
                if variant == "timed_text":
                    rationale = meta.get("rationale", {})
                    genuine = (
                        genuine
                        and rationale.get("event_timing_enabled") is True
                        and rationale.get("event_clock") == "ecb-ust"
                    )
                if ok:
                    for horizon in spec.horizons:
                        indices = [i for i, (_, h) in enumerate(spec.cells()) if h == horizon]
                        raw = scoring._score(
                            dict(
                                ctx,
                                expected_grid=GridSpec(spec.assets, (horizon,)),
                                _samples=ctx["_samples"][:, indices],
                                realized=y[indices],
                                ref_scale=None,
                                normalization_mode=NormalizationMode.RAW_UNRANKABLE,
                            )
                        )
                        horizons_raw.append(
                            dict(
                                unit=unit["id"],
                                variant=variant,
                                seed=seed,
                                horizon=horizon,
                                raw_marginal=raw["marginal"],
                                raw_joint=raw["joint"],
                                raw_tail=raw["tail"],
                                rankable=False,
                            )
                        )
                rows.append(
                    dict(
                        unit=unit["id"],
                        block=unit["block"],
                        variant=variant,
                        seed=seed,
                        passed=ok,
                        quality_evidence=variant == "numeric" or genuine,
                        loss=min(float(detail["score"]), 8) if ok else 8.0,
                        marginal=detail.get("marginal", np.nan) / scale["marginal"],
                        tail=detail.get("tail", np.nan) / scale["tail"],
                        joint=detail.get("joint", np.nan) / scale["joint"],
                    )
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("The planned phase has no cards")
    frame.to_csv(run / f"{phase}-cases.csv", index=False)
    pd.DataFrame(horizons_raw).to_csv(run / f"{phase}-horizons-raw.csv", index=False)
    blocks = frame.groupby(["variant", "block"])[["loss", "marginal", "joint", "tail"]].mean()
    summary = blocks.groupby("variant").mean()
    summary["unit_mean_loss"] = frame.groupby("variant").loss.mean()
    summary.to_csv(run / f"{phase}-summary.csv")
    selected = "numeric"
    baseline_ok = bool(frame[frame.variant == "numeric"].passed.all())
    for variant in sorted(variants, key=lambda v: summary.loc[v, "loss"]):
        evidence = frame[frame.variant == variant]
        if (
            not baseline_ok
            or variant == "numeric"
            or not evidence.quality_evidence.all()
            or not evidence.passed.all()
        ):
            continue
        ratio = summary.loc[variant] / summary.loc["numeric"]
        block_ratio = blocks.loc[variant].loss / blocks.loc["numeric"].loss
        if (
            ratio["loss"] <= 1 - plan["selection_min_gain"]
            and max(ratio["marginal"], ratio["tail"]) <= plan["component_regression_limit"]
            and block_ratio.max() <= plan["event_regression_limit"]
        ):
            selected = variant
            break
    result = dict(
        selected=selected,
        phase=phase,
        rankable=False,
        event_blocks=frame.block.nunique(),
        plan_digest=plan["digest"],
        house_provenance="declared by the prediction producer; not provider-attested",
        baseline_passed=baseline_ok,
        selection_valid=baseline_ok,
        aggregation="equal event-block loss; unit-average loss also reported",
        text_quality_evidence={
            v: bool(frame[frame.variant == v].quality_evidence.all()) for v in variants
        },
    )
    result["paired_differences"] = {}
    for v in variants:
        if v == "numeric":
            continue
        differences = (blocks.loc[v].loss - blocks.loc["numeric"].loss).to_numpy()
        interval = None
        if len(differences) >= 3:
            rng = np.random.default_rng(271)
            means = rng.choice(differences, (2000, len(differences)), replace=True).mean(1)
            interval = np.quantile(means, [0.025, 0.975]).tolist()
        result["paired_differences"][v] = dict(
            mean=float(differences.mean()),
            interval95=interval,
            caveat="descriptive event-block bootstrap; shared series and small samples remain",
        )
    result["digest"] = digest(result)
    write_json(run / f"{phase}-decision.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "smoke", "seal", "score"])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--catalog", type=Path)
    p.add_argument("--phase", choices=PHASES, default="select")
    a = p.parse_args()
    if a.action == "plan":
        if a.catalog is None:
            p.error("plan requires --catalog")
        result = prepare(a.catalog, a.run_dir)
        print(json.dumps(dict(cards=len(result["units"]), plan_digest=result["digest"])))
    elif a.action == "smoke":
        smoke_generate(a.run_dir)
    elif a.action == "seal":
        seal(a.run_dir)
    else:
        print(json.dumps(evaluate(a.run_dir, a.phase)))


if __name__ == "__main__":
    main()
