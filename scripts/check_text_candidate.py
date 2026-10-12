"""## Executive summary (read this first)

Check V13 on every published card twice: exact V12 fallback and synthetic House
replies quoting that card's admitted corpus. Check actual output gates. These
checks do not call a live model, read outcomes or claim forecast-quality gains.
"""

import argparse
import contextlib
import io
import json
import os
import tempfile
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from qfbench2_track_forecasting import candidate, scoring, text_candidate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--units", type=Path, required=True)
    a = p.parse_args()
    for name in ["MODEL_ENDPOINT", "MODEL_TOKEN", "MODEL_NAME"]:
        os.environ.pop(name, None)
    failures, passed, applied = [], 0, 0
    original = text_candidate.House
    with tempfile.TemporaryDirectory() as tmp:
        for cp in sorted(a.units.glob("*/card.toml")):
            card = tomllib.loads(cp.read_text())
            assets = card["targets"]["asset_ids"]
            asof = str(card["provenance"]["data_cutoff"])[:10]
            root = Path(tmp) / cp.parent.name
            try:

                def run(entry, out, cp=cp, asof=asof):
                    with contextlib.redirect_stdout(io.StringIO()):
                        entry(
                            [
                                "--panels",
                                str(cp.parent),
                                "--text",
                                str(cp.parent / "text"),
                                "--asof",
                                asof,
                                "--out",
                                str(out),
                            ]
                        )

                old, off = root / "numeric/forecast.parquet", root / "offline/forecast.parquet"
                run(candidate.main, old)
                run(text_candidate.main, off)
                np.testing.assert_array_equal(
                    pd.read_parquet(old).value, pd.read_parquet(off).value
                )

                class MockHouse:
                    def __init__(self):
                        self.requests = 0
                        self.failures = []

                    def ask(self, prompt, assets=assets):
                        self.requests += 1
                        if self.requests == 1:
                            docs = json.loads(prompt.split("\n", 1)[-1])
                            quote = " ".join(docs[0]["text"].split())[:180]
                            return dict(
                                evidence=[
                                    dict(
                                        doc_id=docs[0]["doc_id"],
                                        quote=quote,
                                        status="inferred",
                                        assets=assets,
                                        interpretation="synthetic interface check",
                                    )
                                ]
                            )
                        return dict(
                            scenarios=[
                                dict(
                                    label="synthetic",
                                    weight=1,
                                    assets={
                                        asset: dict(
                                            shift_risk_sd=0.2,
                                            vol_scale=1.1,
                                            confidence=0.5,
                                            regime_changed=False,
                                            evidence_ids=["e0"],
                                        )
                                        for asset in assets
                                    },
                                )
                            ]
                        )

                text_candidate.House = MockHouse
                on = root / "mock/forecast.parquet"
                run(text_candidate.main, on)
                meta = json.loads((on.parent / "forecast_meta.json").read_text())
                applied += int(meta["reasoning_applied"])
                for output in [off, on]:
                    captured = io.StringIO()
                    with contextlib.redirect_stdout(captured):
                        code = scoring._main(
                            ["score", "--card", str(cp), "--forecast", str(output)]
                        )
                    verdict = json.loads(captured.getvalue())
                    if code or not verdict.get("admissible"):
                        raise ValueError("Output gate refused candidate")
                passed += 1
            except (Exception, SystemExit) as e:
                failures.append(dict(unit=cp.parent.name, error=type(e).__name__))
            finally:
                text_candidate.House = original
    print(
        json.dumps(
            dict(
                cards_passed_both_paths=passed,
                mock_adjustments_applied=applied,
                live_house_quality_test=False,
                failures=failures,
            ),
            indent=2,
        )
    )
    if failures or passed == 0 or applied == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
