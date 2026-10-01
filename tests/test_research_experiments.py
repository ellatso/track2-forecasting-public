"""## Executive summary (read this first)

Verify chronological inputs, monthly asset-specific steps, shared scoring semantics,
configuration balance, and secret-free ZIP inspection on synthetic data only.
"""

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from experiments import batch
from experiments.audit_submission import inspect_zip


def test_configs_unique_and_common_seed():
    cfgs = batch.configs()
    assert len(cfgs) == len({c["id"] for c in cfgs}) == 20
    history = np.cumsum(np.random.default_rng(1).normal(size=(500, 2)), axis=0)
    base = next(c for c in cfgs if c["id"] == batch.BASE)
    a = batch.predict(history, [5, 20], False, "T2-F3", False, base, 200, 17)
    b = batch.predict(history.copy(), [5, 20], False, "T2-F3", False, base, 200, 17)
    assert a.shape == (200, 4)
    np.testing.assert_array_equal(a, b)


def test_monthly_steps_are_per_asset():
    history = np.cumsum(np.random.default_rng(2).normal(size=(80, 2)), axis=0)
    cfg = next(c for c in batch.configs() if c["id"] == batch.BASE)
    result = batch.predict(history, np.array([[2, 4], [3, 5]]), False, "T2-F1", True, cfg, 200, 17)
    assert result.shape == (200, 4)
    assert np.isfinite(result).all()


def synthetic_item():
    dates = pd.bdate_range("2000-01-03", periods=1000)
    values = np.arange(1000, dtype=float)
    return dict(
        frame=pd.DataFrame({"X": values}, index=dates),
        steps=np.array([5, 20]),
        monthly=False,
        returns=False,
    )


def test_future_never_enters_predictor_history():
    item = synthetic_item()
    origin = str(item["frame"].index[500].date())
    history, outcome, _, _ = batch.case_at(item, origin)
    assert len(history) == 501
    np.testing.assert_array_equal(outcome, [505, 520])
    modified = synthetic_item()
    modified["frame"].iloc[501:] += 10000
    changed_history, changed_outcome, _, _ = batch.case_at(modified, origin)
    np.testing.assert_array_equal(changed_history, history)
    assert np.all(changed_outcome != outcome)


def test_long_gap_is_refused():
    item = synthetic_item()
    dates = item["frame"].index.to_numpy().copy()
    dates[400:] += np.timedelta64(3650, "D")
    item["frame"].index = pd.to_datetime(dates)
    with pytest.raises(ValueError, match="gap"):
        batch.case_at(item, str(item["frame"].index[500].date()))


def test_shared_single_cell_weights_and_baseline_identity():
    samples = np.random.default_rng(0).normal(size=(200, 1))
    card = {"scoring": {"params": {}}}
    baseline = batch.raw_score(samples, np.array([0.5]), card, ["X"], [5])
    assert baseline["weights_effective"] == pytest.approx([5 / 7, 0, 2 / 7])
    compared = batch.relative_score(samples, np.array([0.5]), card, baseline)
    assert compared["composite"] == pytest.approx(1)


def test_equal_basket_weight_ignores_repeated_rows():
    frame = pd.DataFrame(
        {"config": ["a"] * 4, "cluster": ["X", "X", "X", "Y"], "composite": [1, 1, 1, 3]}
    )
    assert batch.summarize(frame).iloc[0].local_relative_loss == 2


def test_zip_audit_does_not_read_team_claim_or_print_secrets(tmp_path):
    path = tmp_path / "submission.zip"
    descriptor = dict(
        image={"digest": "sha256:abc"},
        models=[],
        competition_id="test",
        phase="dev",
        track="forecasting",
        team_id="PRIVATE",
        MODEL_TOKEN="SECRET",
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("submission.json", json.dumps(descriptor))
        z.writestr("team-claim.json", "this is deliberately not JSON; SECRET-TEAM-KEY")
    result = inspect_zip(path)
    assert result["image"]["digest"] == "sha256:abc"
    assert "SECRET" not in json.dumps(result)
    assert "PRIVATE" not in json.dumps(result)


def test_selection_lock_and_only_selected_holdout(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "inventory", lambda root: ([], [], {}))

    def fake_run(items, origins, cfgs, seeds, draws):
        calls.append([c["id"] for c in cfgs])
        rows = []
        for c in cfgs:
            for basket in ("X", "Y"):
                rows.append(
                    dict(
                        config=c["id"],
                        cluster=basket,
                        composite=0.8 if c["id"] == "ewma60" else 1.0,
                        family="T2-F1",
                        monthly=False,
                        seed=17,
                        coverage90=0.9,
                    )
                )
        return pd.DataFrame(rows), []

    monkeypatch.setattr(batch, "run_cases", fake_run)
    out = tmp_path / "run"
    assert batch.main(["--run-dir", str(out)]) == 0
    assert json.loads((out / "selected.json").read_text())["config"] == "ewma60"
    assert batch.main(["--run-dir", str(out), "--phase", "holdout"]) == 0
    assert set(calls[-1]) == {batch.BASE, "ewma60"}
    with pytest.raises(SystemExit):
        batch.main(["--run-dir", str(out), "--phase", "holdout"])


def test_modified_plan_refuses_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "inventory", lambda root: ([], [], {}))
    frame = pd.DataFrame(
        [
            dict(
                config=batch.BASE,
                cluster="X",
                composite=1.0,
                family="T2-F1",
                monthly=False,
                seed=17,
                coverage90=0.9,
            )
        ]
    )
    monkeypatch.setattr(batch, "run_cases", lambda *args: (frame, []))
    out = tmp_path / "run"
    batch.main(["--run-dir", str(out)])
    plan = json.loads((out / "plan.json").read_text())
    plan["draws"] = 500
    (out / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(SystemExit):
        batch.main(["--run-dir", str(out), "--phase", "holdout"])
    assert not (out / "holdout_started.json").exists()
