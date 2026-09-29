## Executive summary (read this first)

This fork's first submission candidate is the repo's existing numerical `forecast` implementation. It creates valid joint forecast draws from the supplied historical panels and deliberately makes no text adjustment. The `Participant T2 image` GitHub Action builds a Linux/amd64 image, runs the public exemplar, checks admissibility, and publishes it to `ghcr.io/ellatso/agenthon-t2:dev`. This is a structural baseline, not a claim of a competitive score.

## Submission steps

1. Open the Actions tab and allow workflows for this fork if GitHub asks. Run `Participant T2 image` on the `submission/t2-baseline` branch (or inspect its push-triggered run). A green run is required before using its image.
2. On GitHub, open the `agenthon-t2` package settings and set the package **Public**. Verify it can be pulled without a GitHub login. The evaluation service cannot use your private login.
3. Copy the immutable `sha256:` image digest from the successful run or from `docker buildx imagetools inspect ghcr.io/ellatso/agenthon-t2:dev`. Do not submit the mutable `:dev` tag as the image identity.
4. Register a team at agenthon.net. Start from the current Track 2 Development descriptor in the official shared toolkit. Set `category` to `api`, `track` to `forecasting`, `phase` to `dev`, `image_access` to `public`, the GHCR registry/repository/digest to the published image, and `models` to `[]` for this model-free numerical baseline. Do not copy an example team ID.
5. Install the officially pinned `qfbench2-common` toolkit with Python 3.13. Run `qfbench2 submission alias --team-number YOUR_TEAM_NUMBER`, enter the Team Key in its hidden prompt, and use the derived team ID in `submission.json`. Run the track's local checks, then `qfbench2 submission pack --descriptor submission.json --team-number YOUR_TEAM_NUMBER --out submission.zip`. Enter the key only at the hidden prompt. Keep the key and ZIP private.
6. From the signed-in Agenthon site, follow the CodaBench competition link, request access, and upload `submission.zip` under Development with the one CodaBench account selected for your team. Verify the processing status there and later check the Agenthon Development leaderboard.

Never commit a Team Key, token, private proof, or `submission.zip` to this public fork. When iterating, modify the agent, rerun the checks, publish a new image, and repack the descriptor with its new digest.
