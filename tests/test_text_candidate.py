"""## Executive summary (read this first)

Check V13 evidence-only adjustments, exact V12 fallback, cutoff isolation,
shared-path mixtures and bounded malformed replies. No outcome or live model.
"""

import json

import numpy as np
import pandas as pd
import pytest

from qfbench2_track_forecasting import candidate, text_candidate, text_scenarios


@pytest.fixture
def unit(tmp_path):
    root = tmp_path / "unit"
    (root / "text").mkdir(parents=True)
    dates = pd.bdate_range("2001-01-01", periods=400)
    x = np.random.default_rng(60).normal(0, 0.04, 400)
    pd.DataFrame(dict(date=dates, asset="UST_2Y", value=3 + x.cumsum())).to_parquet(
        root / "panel.parquet"
    )
    (root / "card.toml").write_text(
        '[task]\nid="synthetic"\n[targets]\nasset_ids=["UST_2Y"]\nhorizons=[21,63]\ntarget_type="level"\n'
    )
    asof = str(dates[-1].date())
    quote = "The committee raised the policy rate and expects further adjustment."
    (root / "text/doc.txt").write_text(quote)
    (root / "text/corpus_index.json").write_text(
        json.dumps(
            dict(
                documents=[dict(doc_id="doc", timestamp=asof, file="doc.txt", doc_type="statement")]
            )
        )
    )
    return root, asof, quote


def args(root, asof, out):
    return ["--panels", str(root), "--text", str(root / "text"), "--asof", asof, "--out", str(out)]


def reply(quote):
    return dict(
        evidence=[
            dict(
                doc_id="doc",
                quote=quote,
                assets=["UST_2Y"],
                status="inferred",
                interpretation="policy uncertainty",
            )
        ]
    )


def scenarios(shift=0.8):
    return dict(
        scenarios=[
            dict(
                label="policy",
                weight=1,
                assets={
                    "UST_2Y": dict(
                        shift_risk_sd=shift,
                        vol_scale=1.2,
                        confidence=0.6,
                        regime_changed=False,
                        evidence_ids=["e0"],
                    )
                },
            )
        ]
    )


def test_event_clock_cli_keeps_every_pre_event_draw_exact(unit, tmp_path, monkeypatch):
    root, asof, _ = unit
    date = str((pd.Timestamp(asof) + pd.offsets.BDay(3)).date())
    quote = f"A decision is scheduled for {date} at 09:00 UTC. Its outcome is unknown."
    (root / "text/doc.txt").write_text(quote)
    (root / "card.toml").write_text(
        '[task]\nid="synthetic"\n[targets]\nasset_ids=["UST_2Y"]\n'
        'horizons=[4,1,3,2]\ntarget_type="level"\n'
    )

    def ask(self, prompt):
        self.requests += 1
        if self.requests == 1:
            return reply(quote)
        result = scenarios(0.4)
        result["scenarios"][0]["assets"]["UST_2Y"].update(
            activation_date=date, activation_time="09:00", activation_timezone="UTC"
        )
        return result

    monkeypatch.setattr(text_candidate.House, "ask", ask)
    old, new = tmp_path / "old/f.parquet", tmp_path / "new/f.parquet"
    candidate.main(args(root, asof, old))
    text_candidate.main(args(root, asof, new) + ["--event-timing", "--event-clock", "ecb-ust"])
    before, after = (pd.read_parquet(p).value.to_numpy().reshape(4096, 1, 4) for p in [old, new])
    np.testing.assert_array_equal(before[:, :, [1, 3]], after[:, :, [1, 3]])
    assert np.sum((before == after).all(axis=(1, 2))) == 2048


def test_absent_house_is_exact_v12_even_with_dated_docs(unit, tmp_path, monkeypatch):
    root, asof, _ = unit
    for name in ["MODEL_ENDPOINT", "MODEL_NAME", "MODEL_TOKEN"]:
        monkeypatch.delenv(name, raising=False)
    old, new = tmp_path / "old/f.parquet", tmp_path / "new/f.parquet"
    candidate.main(args(root, asof, old))
    text_candidate.main(args(root, asof, new))
    pd.testing.assert_frame_equal(pd.read_parquet(old), pd.read_parquet(new), check_exact=True)
    meta = json.loads((new.parent / "forecast_meta.json").read_text())
    assert not meta["reasoning_applied"] and meta["house_requests_attempted"] == 0


def test_verified_house_changes_half_paths_and_does_not_send_hindsight_labels(
    unit, tmp_path, monkeypatch
):
    root, asof, quote = unit
    prompts = []

    def ask(self, prompt):
        prompts.append(prompt)
        self.requests += 1
        return reply(quote) if self.requests == 1 else scenarios()

    monkeypatch.setattr(text_candidate.House, "ask", ask)
    old, new = tmp_path / "old/f.parquet", tmp_path / "new/f.parquet"
    candidate.main(args(root, asof, old))
    text_candidate.main(args(root, asof, new))
    a, b = (pd.read_parquet(p).value.to_numpy().reshape(4096, 1, 2) for p in [old, new])
    assert np.sum((a == b).all(axis=(1, 2))) == 2048
    assert np.all(b[:, -1, :].mean(0) > a[:, -1, :].mean(0))
    assert "synthetic" not in "".join(prompts) and "T2-F4" not in "".join(prompts)
    assert json.loads((new.parent / "forecast_meta.json").read_text())["reasoning_applied"]


@pytest.mark.parametrize("bad", [True, "3", float("nan"), float("inf"), 10**500])
def test_nonfinite_or_non_numeric_model_values_are_rejected(bad):
    s = scenarios(bad)
    ev = [dict(id="e0", assets=["UST_2Y"])]
    assert text_scenarios.validate(s, ev, ["UST_2Y"], [63]) == []


def test_unknown_quotes_keep_v12(unit, tmp_path, monkeypatch):
    root, asof, _ = unit

    def ask(self, prompt):
        self.requests += 1
        return reply("This invented statement is absent from every admitted document.")

    monkeypatch.setattr(text_candidate.House, "ask", ask)
    old, new = tmp_path / "old/f.parquet", tmp_path / "new/f.parquet"
    candidate.main(args(root, asof, old))
    text_candidate.main(args(root, asof, new))
    pd.testing.assert_frame_equal(pd.read_parquet(old), pd.read_parquet(new), check_exact=True)


def test_future_panel_and_text_changes_cannot_affect_admitted_inputs(unit, tmp_path, monkeypatch):
    root, asof, quote = unit

    def ask(self, prompt):
        self.requests += 1
        return reply(quote) if self.requests == 1 else scenarios()

    monkeypatch.setattr(text_candidate.House, "ask", ask)
    first, second = tmp_path / "a/f.parquet", tmp_path / "b/f.parquet"
    text_candidate.main(args(root, asof, first))
    frame = pd.read_parquet(root / "panel.parquet")
    frame = pd.concat(
        [
            frame,
            pd.DataFrame(
                dict(date=[pd.Timestamp(asof) + pd.Timedelta(days=1)], asset="UST_2Y", value=1e8)
            ),
        ]
    )
    frame.to_parquet(root / "panel.parquet")
    index = json.loads((root / "text/corpus_index.json").read_text())
    index["documents"].append(dict(doc_id="future", timestamp="2099-01-01", file="future.txt"))
    (root / "text/future.txt").write_text("A future price must never be used.")
    (root / "text/corpus_index.json").write_text(json.dumps(index))
    text_candidate.main(args(root, asof, second))
    pd.testing.assert_frame_equal(pd.read_parquet(first), pd.read_parquet(second), check_exact=True)


def test_scenarios_use_authored_horizon_order_and_keep_uncited_assets():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(512, 2, 2))
    mean = base.mean(0)
    sd = base.std(0)
    summary = {"assets": {a: {"stale_fx": False} for a in ["UST_2Y", "UST_10Y"]}}
    sc = text_scenarios.validate(
        scenarios(), [dict(id="e0", assets=["UST_2Y"])], ["UST_2Y", "UST_10Y"], [63, 21]
    )
    out = text_scenarios.apply(base, ["UST_2Y", "UST_10Y"], [63, 21], summary, mean, sd, sd, sc, 0)
    np.testing.assert_array_equal(out[:, 1, :], base[:, 1, :])
    assert (out[:, 0, 0] - base[:, 0, 0]).mean() > (out[:, 0, 1] - base[:, 0, 1]).mean()


def test_short_horizon_shifts_cannot_exceed_the_smaller_bound():
    sc = text_scenarios.validate(
        scenarios(100), [dict(id="e0", assets=["UST_2Y"])], ["UST_2Y"], [2, 12]
    )
    assert sc[0]["assets"]["UST_2Y"]["shift_risk_sd"] == 0.75 * 0.6


def test_stale_fx_risk_is_bounded_uses_current_peers_and_ignores_future():
    rng = np.random.default_rng(15)
    dates = pd.bdate_range("2000-01-01", periods=2100)
    values = np.exp(rng.normal(0, 0.005, (2100, 3)).cumsum(0))
    values[:, 0] = np.exp(rng.normal(0, 0.0003, 2100).cumsum())
    frame = pd.DataFrame(values, index=dates, columns=["EUR", "GBP", "JPY"])
    data = (
        frame.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="asset", value_name="value")
    )
    data = data[~((data.asset == "EUR") & (data.date > dates[500]) & (data.date < dates[1999]))]
    base = rng.normal(1, 0.001, (4096, 1, 2))
    target = dict(target_type="level")
    first = text_scenarios.summarize(
        {"fx": data}, ["EUR"], [21, 63], str(dates[1999].date()), base, target
    )
    assert set(first[0]["peer_daily_log_volatility"]) == {"GBP", "JPY"}
    assert first[0]["assets"]["EUR"]["stale_fx"]
    assert (first[3] >= first[2]).all() and (first[3] <= 6 * first[2]).all()
    data.loc[data.date > dates[1999], "value"] *= 10000
    after = text_scenarios.summarize(
        {"fx": data}, ["EUR"], [21, 63], str(dates[1999].date()), base, target
    )
    assert first[0] == after[0]
    np.testing.assert_array_equal(first[3], after[3])


def test_peer_risk_does_not_change_residuals_without_a_regime_flag():
    base = np.random.default_rng(18).normal(1, 0.01, (512, 1, 2))
    mean, sd = base.mean(0), base.std(0)
    summary = {"assets": {"EUR": {"stale_fx": True}}}
    ev = [dict(id="e0", assets=["EUR"])]
    s = dict(
        scenarios=[
            dict(
                weight=1,
                assets={
                    "EUR": dict(
                        shift_risk_sd=0,
                        vol_scale=1,
                        confidence=0.5,
                        regime_changed=False,
                        evidence_ids=["e0"],
                    )
                },
            )
        ]
    )
    sc = text_scenarios.validate(s, ev, ["EUR"], [21, 63])
    out = text_scenarios.apply(base, ["EUR"], [21, 63], summary, mean, sd, 6 * sd, sc, 0)
    np.testing.assert_allclose(out, base, rtol=0, atol=0)
