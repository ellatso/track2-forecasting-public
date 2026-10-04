## Executive summary (read this first)

This candidate keeps the precision/monthly-trend forecast and adds eligible
published monthly observations from the current unit's dated BLS corpus.
It does not change daily forecasts or calibrate a new distribution.
See `ARTIFACT_PROVENANCE.md` for interpretation and limitations.

After the image workflow passes and the descriptor is pinned, run
`participant/pack-release-windows.ps1` from branch
`submission/t2-release-inputs` in PowerShell. The official 2.4.4 toolkit reads
the Team Key at its hidden prompt. The script never reads or stores the key.

The output is `Downloads/Agenthon-T2-609-release-inputs/submission.zip`.
Keep the ZIP and Team Key out of GitHub. Old team claims cannot be copied to a
new descriptor. Packaging checks and public card gates establish admissibility;
only CodaBench can establish the official score. No target score is promised.
