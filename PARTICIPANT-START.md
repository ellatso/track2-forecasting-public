## Executive summary (read this first)

This branch tests a coherent-horizon numerical forecast candidate after submission #952729 scored -1.2618. Each draw now shares shocks across its daily horizons. The GitHub Action builds a Linux/amd64 image, runs the public exemplar, checks admissibility, and publishes `ghcr.io/ellatso/agenthon-t2:coherent`. The candidate still ignores text. Its CodaBench score is unknown; the public checks establish admissibility, not improvement over the first submission.

## Submission steps

1. The [candidate build](https://github.com/ellatso/track2-forecasting-public/actions/runs/36678488300) passed 29 producer regression checks, the exemplar, and admissibility checks. Any later image-code edit needs a new green build.
2. The template identifies the immutable digest of `ghcr.io/ellatso/agenthon-t2:coherent`. Confirm anonymous pull access before a CodaBench upload.
3. Register a team at agenthon.net. Copy `participant/submission-dev.template.json` to a private working directory as `submission.json`. It names the real image digest, with `category: api` and `models: []` for this model-free numerical candidate. The template intentionally omits `team_id` and `descriptor_digest`: the packing tool derives and seals both. Do not add the example `team_id` from the official fixture; the packer refuses a mismatched ID.
4. On Windows, install Python 3.13 and run `powershell -ExecutionPolicy Bypass -File .\participant\pack-windows.ps1` from a checkout of this branch. The script defaults to Team Number 609, installs the pinned toolkit, and writes `submission.zip` under `Downloads\Agenthon-T2-609`. Enter the Team Key at the hidden prompt. For another team number, pass `-TeamNumber N`. The toolkit derives the real team ID, seals the descriptor and writes the ZIP. Keep the key and ZIP private. On another operating system, install the pinned toolkit with Python 3.13 and run `qfbench2 submission pack --descriptor submission.json --team-number 609 --out submission.zip` yourself.
5. From the signed-in Agenthon site, follow the CodaBench competition link, request access, and upload `submission.zip` under Development with the one CodaBench account selected for your team. Verify the processing status there and later check the Agenthon Development leaderboard.

Never commit a Team Key, token, private proof, or `submission.zip` to this public fork. When iterating, modify the agent, rerun the checks, publish a new image, and repack the descriptor with its new digest.
