## Executive summary (read this first)

This branch tests the next numerical forecast after CodaBench #952974 scored -1.1400. It uses each asset's last 300 observations and a mean step drift for level targets, including monthly data. The previous coherent-horizon path is retained. The candidate's private CodaBench score is unknown.

## Evidence and packaging

A local walk-forward check used only published panel rows before their cards' as-of dates. Against a 300-observation, full-drift Gaussian reference, the prior full-history zero-drift forecast's mean Gaussian CRPS ratio was 1.127 for 702 daily asset/time checks and 2.067 for 36 monthly checks. These are diagnostic marginal checks, not the competition's joint score. The new implementation passed 31 focused tests locally; see the Action for the image smoke check and public admissibility gates.

Once the Action succeeds, this branch's `participant/submission-dev.template.json` pins the immutable image digest. On Windows check out this branch and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\participant\pack-windows.ps1
```

The script uses Team Number 609 and prompts for the Team Key privately. It writes `Downloads\Agenthon-T2-609\submission.zip` for the Development CodaBench upload. Keep the Team Key and ZIP out of GitHub. It installs the pinned public qfbench2 toolkit using Python 3.13. The image is numerical and text-blind, so the descriptor remains `category: api` and `models: []`. A public gate passing does not predict the leaderboard score.
