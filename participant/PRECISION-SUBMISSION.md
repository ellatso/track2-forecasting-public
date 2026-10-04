## Executive summary (read this first)

This numeric Development candidate keeps the half-drift/sample-EWMA50 daily distribution and uses 4096 scrambled Sobol draws. Monthly means blend the original full-drift centre equally with a trend fitted on the last 12 consecutive monthly observations and damped with a 12-month half-life. Covariance and coherent paths are retained. No House call, packaged trained model or practice-answer lookup is used.

## Packaging

After the image descriptor is pinned to a successful tested digest, run `participant/pack-precision-windows.ps1`. The toolkit 2.4.4 asks for the Team Key through a hidden terminal prompt. Team 609's output is `Downloads/Agenthon-T2-609-precision-monthly50/submission.zip`. Never commit the signing key or signed ZIP.

## Validation and limits

The frozen historical precision batch and separately frozen supplementary dates support a monthly trial, with much smaller and uncertain changes from daily sampling precision. Historical months use snapshots without historical release-vintage reconstruction. These are diagnostics, not independent real-time OOS evidence. The documented M0 method replica does not expose official private scales or establish a leaderboard score. All numeric diagnostic results stay private. The actual container must pass the all-public-card offline/read-only check before publication.

This descriptor remains a Development submission. Formal Final designation follows the organizer's separate Final phase instructions. No specific competition score is promised.
