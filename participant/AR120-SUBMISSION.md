# Executive summary (read this first)

This numeric candidate changes only daily conditional means. It equally blends the existing half-drift mean and a 120-observation autoregressive mean. Monthly sampling, daily covariance, seed 0 and 1,000 default draws are unchanged. Large-gap or short histories retain the original mean. No House calls, packaged fitted model, or cross-unit outcome lookup is used.

## Run

After the descriptor is pinned to the successful tested image digest, run `participant/pack-ar120-windows.ps1` in Windows PowerShell. The official toolkit 2.4.4 asks for the Team Key through its hidden prompt and creates `Downloads/Agenthon-T2-609-ar120-blend50/submission.zip` for Team 609.

## Evidence and limits

The selection comes from explored historical mean-model diagnostics, not untouched holdout or official private M0 scales. Per-case normalization was unstable near zero joint loss; pooled-scale sensitivity suggested this candidate, but does not establish an official score improvement. Official performance requires an upload. Keep the signing key and signed ZIP private.
