"""## Executive summary (read this first)

Read only submission.json from a local ZIP and print its image identity and model
names. Never extract team-claim.json or print Team Keys, tokens or team identifiers.
Use this to associate a CodaBench submission ID with the actual uploaded image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def inspect_zip(path):
    with zipfile.ZipFile(path) as archive:
        entry = archive.getinfo("submission.json")
        if entry.file_size > 1024 * 1024:
            raise ValueError("descriptor too large")
        descriptor = json.loads(archive.read(entry))
    return dict(
        file=str(Path(path).resolve()),
        zip_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        image=descriptor["image"],
        model_names=[m["name"] for m in descriptor.get("models", [])],
        competition_id=descriptor["competition_id"],
        phase=descriptor["phase"],
        track=descriptor["track"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_zip(args.zip), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
