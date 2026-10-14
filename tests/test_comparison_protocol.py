"""## Executive summary (read this first)

Test event isolation, answer-free snapshots, fixed-denominator failures and
immutable phase decisions using synthetic cards and the unchanged official scorer.
These examples are protocol tests, never evidence of forecasting improvement.
"""

import json
import uuid
from pathlib import Path

import pandas as pd
import pytest

from experiments import compare_final as c
from tests.conftest import (
    build_submission,
    write_card,
    write_panels,
    write_reference,
    write_text_corpus,
)


@pytest.fixture
def comparison(tmp_path):
    entries = []
    for i, date in enumerate(["2020-07-01", "2020-09-01", "2020-11-02"]):
        source = tmp_path / f"source-{i}"
        end = str((pd.Timestamp(date) + pd.offsets.BDay(5)).date())
        write_card(
            source,
            unit_id=f"t2-SYN-{i}",
            asof=date,
            target_dates=[str((pd.Timestamp(date) + pd.offsets.BDay(1)).date()), end],
            canary=str(uuid.uuid4()),
        )
        write_panels(source, asof=date)
        write_text_corpus(source, asof=date)
        reference = write_reference(source)
        files = []
        for f in source.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(source))
                role = (
                    "reference"
                    if rel.startswith("reference/")
                    else (
                        "fixture"
                        if rel == "card.toml"
                        else "input"
                        if rel.startswith("panels/")
                        else "corpus"
                    )
                )
                files.append(dict(path=rel, role=role, sha256=c.sha(f), bytes=f.stat().st_size))
        c.write_json(source / "manifest.json", dict(manifest_version="2.0", files=files))
        entries.append(
            dict(
                unit=str(source),
                realized=str(reference / "realized.parquet"),
                scale=str(reference / "ref_scale.json"),
            )
        )
    catalog = tmp_path / "catalog.json"
    c.write_json(catalog, dict(units=entries, normalization="m0-expected-error"))
    run = tmp_path / "comparison"
    plan = c.prepare(catalog, run, draws=256, seeds=(0,))
    return run, plan


def make_predictions(
    run, plan, missing=None, mode="unavailable", mock_advantage=False, best="timed_text"
):
    evidence = {}
    for u in plan["units"]:
        for v in c.VARIANTS:
            key = f"{v}/0/{u['handle']}"
            if v != missing:
                build_submission(
                    run / "forecasts" / key,
                    unit_id=u["id"],
                    asof=u["asof"],
                    n_draws=256,
                    centre=1.0 if mock_advantage and v == best else 3.0,
                    meta_overrides={
                        "house_requests_attempted": 2 if mode != "unavailable" else 0,
                        "rationale": dict(
                            file="forecast_rationale.md",
                            event_timing_enabled=v == "timed_text",
                            event_clock="ecb-ust" if v == "timed_text" else "date",
                        ),
                    },
                )
            evidence[key] = dict(
                mode="numeric" if v == "numeric" else mode,
                input_digest=u["input_digest"],
                successful_replies=0 if mode == "unavailable" else 2,
            )
    c.write_json(run / "execution.json", evidence)
    c.seal(run)


def test_snapshot_excludes_published_outcomes_and_scales(comparison):
    run, plan = comparison
    for u in plan["units"]:
        paths = list((run / "inputs" / u["handle"]).rglob("*"))
        assert not any(p.name in ["realized.parquet", "ref_scale.json", "reference"] for p in paths)
        assert c.digest(u["input_hashes"]) == u["input_digest"]


def test_official_card_role_is_accepted_only_with_its_published_checksum(comparison):
    _, plan = comparison
    source = Path(plan["units"][0]["references"]["realized"]["path"]).parents[1]
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    next(item for item in manifest["files"] if item["path"] == "card.toml")["role"] = "card"
    c.write_json(path, manifest)
    _, selected = c.allowed_inputs(source)
    assert any(item["path"] == "card.toml" for item in selected)
    with (source / "card.toml").open("a") as file:
        file.write("\n# Modified after publication.\n")
    with pytest.raises(ValueError, match="Input checksum mismatch"):
        c.allowed_inputs(source)


def test_mock_cannot_win_even_when_its_synthetic_predictions_are_better(comparison):
    run, plan = comparison
    make_predictions(run, plan, mode="mock", mock_advantage=True)
    result = c.evaluate(run, "select")
    rows = pd.read_csv(run / "select-cases.csv").set_index("variant")
    assert rows.loc["timed_text", "loss"] < rows.loc["numeric", "loss"]
    assert result["selected"] == "numeric"
    assert not result["text_quality_evidence"]["timed_text"]


def test_shared_event_envelopes_cannot_cross_chronological_splits():
    units = [
        dict(handle=str(i), start=f"2020-{i+1:02d}-01", end=f"2020-{i+1:02d}-05") for i in range(6)
    ]
    units[0]["event"] = units[2]["event"] = "one-specific-event"
    splits = c.event_splits(units)
    assert splits["0"] == splits["1"] == splits["2"]


def test_missing_output_counts_eight_and_missing_house_cannot_win(comparison):
    run, plan = comparison
    make_predictions(run, plan, missing="timed_text")
    result = c.evaluate(run, "select")
    rows = pd.read_csv(run / "select-cases.csv")
    assert len(rows) == 3
    assert rows[rows.variant == "timed_text"].loss.iloc[0] == 8
    assert result["selected"] == "numeric"
    assert not result["text_quality_evidence"]["previous_text"]


def test_modified_forecast_is_refused_even_if_it_would_improve_a_score(comparison):
    run, plan = comparison
    make_predictions(run, plan)
    unit = next(u for u in plan["units"] if u["phase"] == "select")
    path = run / "forecasts" / "numeric" / "0" / unit["handle"] / "forecast_rationale.md"
    path.write_text("Changed after predictions were sealed.")
    with pytest.raises(ValueError, match="Forecast changed"):
        c.evaluate(run, "select")


def test_modified_reference_is_not_treated_as_a_missing_participant_result(comparison):
    run, plan = comparison
    make_predictions(run, plan)
    unit = next(u for u in plan["units"] if u["phase"] == "select")
    with open(unit["references"]["realized"]["path"], "ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="Missing or changed reference"):
        c.evaluate(run, "select")


def test_holdout_is_once_only_and_runs_only_locked_candidate(comparison):
    run, plan = comparison
    make_predictions(run, plan)
    assert c.evaluate(run, "select")["selected"] == "numeric"
    assert c.evaluate(run, "confirm")["selected"] == "numeric"
    c.evaluate(run, "test")
    rows = pd.read_csv(run / "test-cases.csv")
    assert set(rows.variant) == {"numeric"}
    with pytest.raises(FileExistsError):
        c.evaluate(run, "test")


def test_only_selected_policy_reaches_confirmation_and_test(comparison):
    run, plan = comparison
    make_predictions(run, plan, mode="house_replay", mock_advantage=True, best="previous_text")
    assert c.evaluate(run, "select")["selected"] == "previous_text"
    assert c.evaluate(run, "confirm")["selected"] == "previous_text"
    c.evaluate(run, "test")
    rows = pd.read_csv(run / "test-cases.csv")
    assert set(rows.variant) == {"numeric", "previous_text"}
    horizons = pd.read_csv(run / "test-horizons-raw.csv")
    assert set(horizons.horizon) == {1, 5}
    assert not horizons.rankable.any()
