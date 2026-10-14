"""## Executive summary (read this first)

Run the real forecast verb in fresh, restricted Linux containers for every card.
Check output gates without outcomes. Exercise the HTTP/proxy path on four daily
cards using a loopback synthetic server, including unavailable and invalid replies.
These are runtime checks, never live House calls or forecasting-quality evidence.
Keep all generated files outside the public checkout; never upload a submission.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import time
import tomllib
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = {"forecast.parquet", "forecast_meta.json", "forecast_rationale.md"}


def synthetic_transport(command, mode, assets):
    """Use real HTTP and authentication headers, but only a local synthetic proxy."""
    requests, errors = [], []

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assert self.path == "http://house-test.invalid:8443/v1/chat/completions"
                assert self.headers.get("Authorization") == "Bearer synthetic-bearer"
                assert self.headers.get("Proxy-Authorization", "").startswith("Basic ")
                assert body["model"] == "synthetic-transport-only"
                assert body["max_tokens"] <= 4000
                assert body["chat_template_kwargs"]["enable_thinking"] is False
                requests.append(True)
                if mode == "http_error":
                    self.send_error(503)
                    return
                if len(requests) == 1:
                    docs = json.loads(body["messages"][-1]["content"].split("\n", 1)[-1])
                    reply = dict(
                        evidence=[
                            dict(
                                doc_id=docs[0]["doc_id"],
                                quote=" ".join(docs[0]["text"].split())[:180],
                                status="inferred",
                                assets=assets,
                                interpretation="Synthetic transport check; no quality claim",
                            )
                        ]
                    )
                else:
                    reply = dict(
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
                raw = (
                    b"invalid JSON"
                    if mode == "invalid_json"
                    else json.dumps(
                        dict(
                            choices=[
                                dict(
                                    finish_reason="length" if mode == "truncated" else "stop",
                                    message=dict(content=json.dumps(reply)),
                                )
                            ]
                        )
                    ).encode()
                )
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except (AssertionError, ValueError, KeyError, IndexError):
                errors.append("HTTP request did not match the published contract")
                self.send_error(400)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = dict(
        os.environ,
        MODEL_ENDPOINT="http://house-test.invalid:8443",
        MODEL_TOKEN="synthetic-bearer",
        MODEL_NAME="synthetic-transport-only",
    )
    proxy = f"http://synthetic:password@127.0.0.1:{server.server_port}"
    env.update(
        http_proxy=proxy,
        https_proxy=proxy,
        HTTP_PROXY=proxy,
        HTTPS_PROXY=proxy,
        NO_PROXY="*",
        no_proxy="*",
    )
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=75)
        if result.returncode or errors or not 1 <= len(requests) <= 3:
            raise RuntimeError("Synthetic HTTP forecast failed: " + result.stderr[-1000:])
        if mode == "valid" and len(requests) != 2:
            raise RuntimeError("Expected successful extraction and scenario HTTP requests")
        return dict(synthetic_http_requests=len(requests), live_house_quality_test=False)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def forecast_args(card, root, output):
    panels = root / "panels" if (root / "panels").is_dir() else root
    return [
        "--panels",
        str(panels),
        "--text",
        str(root / "text"),
        "--asof",
        str(card["provenance"]["data_cutoff"])[:10],
        "--out",
        str(output),
    ]


def container_command(image, unit, output, mode, timed=False):
    name = "t2-runtime-" + uuid.uuid4().hex[:12]
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--read-only",
        "--user",
        "65534:65534",
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges",
        "--network=none",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=64m",
        "--pids-limit",
        "256",
        "--ulimit",
        "nofile=1024:1024",
        "--ulimit",
        "nproc=256:256",
        "--ulimit",
        "fsize=67108864:67108864",
        "--cpus",
        "2",
        "--memory",
        "2g",
        "--memory-swap",
        "2g",
        "-e",
        "HOME=/nonexistent",
        "-e",
        "QFBENCH_SEED=0",
        "-e",
        "QFBENCH_NETWORK=none",
        "-v",
        f"{unit}:/input:ro",
        "-v",
        f"{output}:/output",
    ]
    card = tomllib.loads((unit / "card.toml").read_text(encoding="utf-8"))
    if mode in ("offline", "numeric"):
        command += [image]
        command += (
            ["forecast"]
            if mode == "offline"
            else ["python", "-m", "qfbench2_track_forecasting.candidate"]
        )
        args = forecast_args(card, Path("/input"), Path("/output/forecast.parquet"))
        # The container path has no host directory to inspect.
        args[1] = "/input/panels" if (unit / "panels").is_dir() else "/input"
        command += args
    else:
        command += [
            "-v",
            f"{Path(__file__).resolve()}:/preflight.py:ro",
            image,
            "python",
            "/preflight.py",
            "--inside",
            "--mode",
            mode,
        ]
    if timed:
        command += ["--event-timing"]
        if mode in ("offline", "numeric"):
            command += ["--event-clock", "ecb-ust"]
    return name, command


def host_main(image, units, output):
    import pandas as pd

    from experiments.compare_final import allowed_inputs
    from qfbench2_track_forecasting import scoring

    output = output.resolve()
    if output == ROOT or ROOT in output.parents or (output.exists() and any(output.iterdir())):
        raise ValueError("Use a new, private runtime-check directory outside the checkout")
    output.mkdir(parents=True)
    inspection = json.loads(subprocess.check_output(["docker", "image", "inspect", image]))[0]
    config = inspection["Config"]
    if (
        inspection["Architecture"] != "amd64"
        or inspection["Os"] != "linux"
        or config.get("Volumes")
        or config.get("Labels", {}).get("qfbench2.interface_version") != "2.0"
    ):
        raise ValueError("Image is not a linux/amd64 interface-2.0 image without VOLUME")
    thread_vars = {
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }
    report = dict(
        image_id=inspection["Id"],
        image_thread_settings={
            key: value
            for key, value in (item.split("=", 1) for item in config.get("Env", []))
            if key in thread_vars
        },
        live_house_quality_test=False,
        runtime="ordinary Docker; not organizer attestation",
        cases=[],
        failures=[],
    )
    cards = []
    for original in sorted(units.resolve().glob("*/card.toml")):
        manifest, keep = allowed_inputs(original.parent)
        snapshot = output / "inputs" / original.parent.name
        for item in keep:
            destination = snapshot / item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original.parent / item["path"], destination)
        (snapshot / "manifest.json").write_text(
            json.dumps(dict(manifest, files=keep), indent=2) + "\n", encoding="utf-8"
        )
        cards.append(snapshot / "card.toml")
    if not cards:
        raise ValueError("No cards found")
    representatives = {}
    for cp in cards:
        c = tomllib.loads(cp.read_text(encoding="utf-8"))
        if c["targets"].get("target_frequency", "daily") == "daily" and c.get("text", {}).get(
            "n_documents", 0
        ):
            representatives.setdefault(c["metadata"]["category"], cp.parent)
    if set(representatives) != {f"T2-F{i}" for i in range(1, 5)}:
        raise ValueError("Need one daily HTTP-test card for each of the four families")
    tasks = [(cp.parent, "offline", False) for cp in cards]
    for unit in representatives.values():
        tasks += [
            (unit, mode, False)
            for mode in ["numeric", "valid", "http_error", "invalid_json", "truncated"]
        ]
        tasks += [(unit, "valid", True)]
    for unit, mode, timed in tasks:
        label = mode + ("-timed" if timed else "")
        folder = output / unit.name / label
        folder.mkdir(parents=True)
        folder.chmod(0o777)
        name, command = container_command(image, unit, folder, mode, timed)
        start = time.monotonic()
        row = dict(unit=unit.name, mode=label)
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=120)
            if result.returncode:
                raise RuntimeError("Container failed: " + result.stderr[-1000:])
            if {p.name for p in folder.iterdir()} != REQUIRED:
                raise RuntimeError("Output must contain exactly the three forecast files")
            total = sum(p.stat().st_size for p in folder.iterdir())
            if total > 64 * 1024 * 1024:
                raise RuntimeError("Output tree exceeds 64 MiB")
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                code = scoring._main(
                    [
                        "score",
                        "--card",
                        str(unit / "card.toml"),
                        "--forecast",
                        str(folder / "forecast.parquet"),
                    ]
                )
            verdict = json.loads(captured.getvalue())
            if code or not verdict.get("admissible"):
                raise RuntimeError("Canonical output gates refused the forecast")
            meta = json.loads((folder / "forecast_meta.json").read_text(encoding="utf-8"))
            if mode in ("offline", "numeric"):
                if meta["house_requests_attempted"] or meta["reasoning_applied"]:
                    raise RuntimeError("Offline/numeric mode unexpectedly called a model")
            else:
                transport = json.loads(result.stdout)
                row.update(transport)
                expected_requests = 2 if mode == "valid" else 1
                if meta["house_requests_attempted"] != expected_requests:
                    raise RuntimeError("Unexpected HTTP request count")
                base = pd.read_parquet(output / unit.name / "numeric/forecast.parquet")
                actual = pd.read_parquet(folder / "forecast.parquet")
                if mode == "valid":
                    draws = meta["n_draws"]
                    equal = (
                        (actual.value.to_numpy() == base.value.to_numpy()).reshape(draws, -1).all(1)
                    )
                    if not meta["reasoning_applied"] or equal.sum() != draws // 2:
                        raise RuntimeError(
                            "Synthetic adjustment must retain exactly half the paths"
                        )
                else:
                    pd.testing.assert_frame_equal(base, actual, check_exact=True)
                    if meta["reasoning_applied"]:
                        raise RuntimeError("Invalid HTTP response changed the numeric forecast")
            row.update(passed=True, seconds=round(time.monotonic() - start, 3), bytes=total)
        except (Exception, SystemExit) as exc:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
            row.update(passed=False, error=str(exc))
            report["failures"].append(row.copy())
        report["cases"].append(row)
        (output / "runtime-report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(row), flush=True)
    if report["failures"]:
        raise SystemExit(1)
    print(
        json.dumps(dict(cases_passed=len(tasks), cards=len(cards), live_house_quality_test=False))
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image")
    p.add_argument("--units", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--inside", action="store_true")
    p.add_argument("--mode", choices=["valid", "http_error", "invalid_json", "truncated"])
    p.add_argument("--event-timing", action="store_true")
    a = p.parse_args()
    if a.inside:
        if not a.mode:
            p.error("--inside requires --mode")
        root = Path("/input")
        card = tomllib.loads((root / "card.toml").read_text(encoding="utf-8"))
        command = ["forecast"] + forecast_args(card, root, Path("/output/forecast.parquet"))
        if a.event_timing:
            command += ["--event-timing", "--event-clock", "ecb-ust"]
        print(json.dumps(synthetic_transport(command, a.mode, card["targets"]["asset_ids"])))
    else:
        if not a.image or not a.units or not a.out:
            p.error("Host mode requires --image, --units and --out")
        host_main(a.image, a.units, a.out)


if __name__ == "__main__":
    main()
