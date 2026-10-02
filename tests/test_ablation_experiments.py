"""## Executive summary (read this first)

Check paired effect signs, interactions, basket weighting and incomplete-case
refusal. Ensure the four-arm protocol preserves baseline and prior coverage.
"""

import numpy as np
import pandas as pd
import pytest

from experiments import ablation as b
from experiments import adaptive as a
from experiments import batch, frequency


def synthetic():
    rows = []
    for group, cluster, origin in [
        ("a", "A", "2023-01-03"),
        ("b", "A", "2021-01-04"),
        ("c", "B", "2023-01-03"),
    ]:
        for cfg, loss in [(b.BASE, 1), (b.VOL, 0.9), (b.CORR, 1.2), (b.BOTH, 1.05)]:
            rows.append(
                dict(
                    group=group,
                    cluster=cluster,
                    origin=origin,
                    seed=17,
                    config=cfg,
                    composite=loss,
                    marginal_contribution=loss * 0.5,
                    joint_contribution=loss * 0.3,
                    tail_contribution=loss * 0.2,
                )
            )
    return pd.DataFrame(rows)


def test_effects_are_paired_and_interaction_has_correct_sign():
    report = b.effects(synthetic())
    expected = {
        "variance_without_shrinkage": -0.1,
        "shrinkage_without_variance": 0.2,
        "variance_with_shrinkage": -0.15,
        "shrinkage_with_variance": 0.15,
        "combined_vs_baseline": 0.05,
        "interaction": -0.05,
    }
    for effect, value in expected.items():
        table = report[report.effect == effect]
        np.testing.assert_allclose(table.composite, value)
        np.testing.assert_allclose(table[b.METRICS[1:]].sum(axis=1), table.composite)


def test_missing_or_duplicate_case_is_refused():
    frame = synthetic()
    with pytest.raises(ValueError, match="Incomplete"):
        b.effects(frame.iloc[1:])
    with pytest.raises(ValueError, match="Duplicate"):
        b.effects(pd.concat([frame, frame.iloc[:1]]))


def test_baskets_are_weighted_equally_not_by_row_count():
    report = b.effects(synthetic())
    report.loc[report.cluster == "B", b.METRICS] *= 3
    value = (
        b.summarize_effects(report)
        .set_index("effect")
        .loc["variance_without_shrinkage", "composite"]
    )
    assert value == pytest.approx(-0.2)


def test_original_baseline_and_previous_both_arm_are_preserved():
    history = np.cumsum(np.random.default_rng(5).normal(size=(450, 3)), axis=0)
    cfgs = b.configs()
    assert {(c["mix"], c["shrink"]) for c in cfgs} == {(0, 0), (0.5, 0), (0, 0.25), (0.5, 0.25)}
    np.testing.assert_array_equal(
        b.predict(history, [21, 63], False, "T2-F3", False, cfgs[0], 200, 17),
        batch.predict(
            np.ascontiguousarray(history),
            [21, 63],
            False,
            "T2-F3",
            False,
            frequency.configs(False)[0],
            200,
            17,
        ),
    )
    np.testing.assert_array_equal(
        b.predict(history, [21, 63], False, "T2-F3", False, cfgs[3], 200, 17),
        a.predict(history, [21, 63], False, "T2-F3", False, a.configs()[1], 200, 17),
    )


def test_runner_preserves_adaptive_dates_bounds_and_gap_checks(monkeypatch):
    received = {}

    def capture(*args, **kwargs):
        received.update(kwargs)
        assert args[1] == a.ORIGINS

    monkeypatch.setattr(frequency, "run_cases", capture)
    b.run_cases([], [0, 17, 41], 1000)
    assert received["validation_bounds"] == a.BOUNDS
    assert received["history_validator"] == a.validate_history


def test_existing_run_refused_before_any_work(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "plan.json").write_text("{}")
    with pytest.raises(SystemExit):
        b.main(["--run-dir", str(out)])
