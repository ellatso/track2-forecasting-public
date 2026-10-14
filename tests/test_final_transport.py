"""## Executive summary (read this first)

Exercise the current text forecaster through a real local HTTP proxy and a child
process. Valid synthetic replies change half the paths; HTTP failures, malformed
JSON and truncated replies preserve the numerical forecast exactly. No live model
or historical realized outcome is used in these transport tests.
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from qfbench2_track_forecasting import candidate
from scripts.check_final_container import synthetic_transport
from tests import test_text_candidate as fixtures
from tests.test_text_candidate import args

unit = fixtures.unit


@pytest.mark.parametrize(
    "mode,timed",
    [
        ("valid", False),
        ("valid", True),
        ("http_error", False),
        ("invalid_json", False),
        ("truncated", False),
    ],
)
def test_real_http_text_cli_keeps_valid_mixture_or_exact_failure_fallback(
    unit, tmp_path, mode, timed
):
    root, asof, _ = unit
    old, new = tmp_path / "numeric/f.parquet", tmp_path / "transport/f.parquet"
    candidate.main(args(root, asof, old))
    command = [sys.executable, "-m", "qfbench2_track_forecasting.text_candidate"]
    command += args(root, asof, new)
    if timed:
        command += ["--event-timing", "--event-clock", "ecb-ust"]
    evidence = synthetic_transport(command, mode, ["UST_2Y"])
    before, after = (pd.read_parquet(p) for p in [old, new])
    meta = json.loads((new.parent / "forecast_meta.json").read_text())
    assert not evidence["live_house_quality_test"]
    assert evidence["synthetic_http_requests"] == meta["house_requests_attempted"]
    if mode == "valid":
        assert meta["reasoning_applied"]
        assert meta["rationale"]["event_timing_enabled"] is timed
        equal = (before.value.to_numpy() == after.value.to_numpy()).reshape(4096, -1).all(1)
        assert np.sum(equal) == 2048
    else:
        assert not meta["reasoning_applied"]
        pd.testing.assert_frame_equal(before, after, check_exact=True)
