"""## Executive summary (read this first)

Validate release extraction with synthetic dated documents. Refuse future news,
nonconsecutive months, ambiguous copies and wrong seasonal or annual measures.
Check that adding a published observation preserves the authored target endpoint.
"""

import json

import numpy as np
import pandas as pd
import pytest

from qfbench2_track_forecasting import cli
from qfbench2_track_forecasting.release_inputs import augment, parse

CPI = (
    "CONSUMER PRICE INDEX - JUNE 2023\n"
    "The Consumer Price Index for All Urban Consumers (CPI-U) rose 0.2 percent "
    "in June on a seasonally adjusted basis. Over the last 12 months it rose 3.0 percent."
)
UNRATE = (
    "THE EMPLOYMENT SITUATION -- JULY 2024\n" "The unemployment rate rose to 4.3 percent in July."
)


def setup(tmp_path, text=CPI, timestamp="2023-07-12", last="2023-05-01", asset="CPI_ALL"):
    dates = pd.date_range(end=last, periods=36, freq="MS")
    panel = pd.DataFrame(dict(date=dates, asset=asset, value=np.linspace(290, 300, 36)))
    (tmp_path / "release.txt").write_text(text)
    (tmp_path / "corpus_index.json").write_text(
        json.dumps(
            dict(
                documents=[
                    dict(
                        timestamp=timestamp,
                        doc_type="macro_release",
                        source="BLS",
                        file="release.txt",
                    )
                ]
            )
        )
    )
    return {"macro": panel}


def test_extract_month_and_correct_units():
    month, values = parse(CPI)
    assert str(month) == "2023-06"
    assert values["CPI_ALL"]["value"] == 0.2
    assert parse(UNRATE)[1]["UNRATE"]["value"] == 4.3
    assert not parse(CPI.replace("seasonally adjusted", "unadjusted"))[1]
    assert not parse(CPI.replace("0.2 percent in June", "0.2 percent in May"))[1]


def test_append_without_altering_published_history(tmp_path):
    panels = setup(tmp_path)
    result, ledger = augment(panels, ["CPI_ALL"], np.array([[4]]), tmp_path, "2023-07-12")
    assert len(ledger) == 1
    assert len(panels["macro"]) == 36
    assert len(result["macro"]) == 37
    history = cli._series(result, "CPI_ALL", "2023-07-12")
    assert history.iloc[-1] == pytest.approx(300.6)
    assert pd.Timestamp(history.index[-1]) == pd.Timestamp("2023-06-01")
    assert ledger[0]["interpretation"] == "monthly_percent"


@pytest.mark.parametrize(
    "timestamp,last,steps",
    [
        ("2023-07-13", "2023-05-01", 4),
        ("2023-07-12", "2023-04-01", 4),
        ("2023-07-12", "2023-06-01", 4),
        ("2023-07-12", "2023-05-01", 1),
    ],
)
def test_unavailable_or_inapplicable_release_is_ignored(tmp_path, timestamp, last, steps):
    panels = setup(tmp_path, timestamp=timestamp, last=last)
    result, ledger = augment(panels, ["CPI_ALL"], np.array([[steps]]), tmp_path, "2023-07-12")
    assert ledger == []
    pd.testing.assert_frame_equal(result["macro"], panels["macro"])


def test_duplicate_documents_are_not_silently_selected(tmp_path):
    panels = setup(tmp_path)
    idx = tmp_path / "corpus_index.json"
    data = json.loads(idx.read_text())
    data["documents"] *= 2
    idx.write_text(json.dumps(data))
    _, ledger = augment(panels, ["CPI_ALL"], np.array([[4]]), tmp_path, "2023-07-12")
    assert ledger == []


def test_missing_text_and_daily_inputs_are_exact_noops(tmp_path):
    panels = setup(tmp_path)
    assert augment(panels, ["CPI_ALL"], None, tmp_path, "2023-07-12") == (panels, [])
    (tmp_path / "corpus_index.json").unlink()
    assert augment(panels, ["CPI_ALL"], np.array([[4]]), tmp_path, "2023-07-12") == (panels, [])
