"""## Executive summary (read this first)

Run every published card using the fixed numeric candidate and verify its actual public gates.
No realized market outcome is read, and no accuracy score is claimed by this check.
"""

import argparse
import contextlib
import io
import json
import os
import tempfile
import tomllib
from pathlib import Path

from qfbench2_track_forecasting import candidate, scoring


def main():
    for name in ("MODEL_ENDPOINT", "MODEL_TOKEN", "MODEL_NAME"):
        os.environ.pop(name, None)
    passed, failures = 0, []
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, default=root / "units")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        for card_path in sorted(args.units.glob("*/card.toml")):
            unit = card_path.parent
            card = tomllib.loads(card_path.read_text(encoding="utf-8"))
            asof = str(card.get("provenance", {}).get("data_cutoff", ""))[:10]
            if not asof:
                failures.append({"unit": unit.name, "failure": "missing cutoff"})
                continue
            out = Path(temp) / unit.name / "forecast.parquet"
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    candidate.main(
                        [
                            "--panels",
                            str(unit),
                            "--text",
                            str(unit / "text"),
                            "--asof",
                            asof,
                            "--out",
                            str(out),
                        ]
                    )
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    code = scoring._main(
                        ["score", "--card", str(card_path), "--forecast", str(out)]
                    )
                verdict = json.loads(captured.getvalue())
                if code or not verdict.get("admissible"):
                    failures.append({"unit": unit.name, "failure": verdict})
                else:
                    passed += 1
            except (Exception, SystemExit) as exc:
                failures.append({"unit": unit.name, "failure_type": type(exc).__name__})
    print(json.dumps({"published_cards_passed": passed, "failures": failures}, indent=2))
    if failures or passed == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
