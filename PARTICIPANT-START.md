## Executive summary (read this first)

This branch tests a family-calibrated forecast following Team 609's Development score of -1.0042 for submission #953358. It retains the 300-observation coherent Gaussian path and monthly macro drift, while calibrating daily forecasts from each unit's declared card family.

F1 and F2 daily cards multiply empirical drift by 0.5 and spread by 0.85. F3 daily cards retain empirical drift and multiply spread by 0.85. F4 and monthly cards retain the previous forecast. These choices were evaluated by rolling cutoffs strictly within published pre-as-of panel histories; they are a candidate, not a known improvement on the sealed outcomes. No text is used.

The public historical check covers 99 cards and 1,751 marginal horizon checks. On the last two synthetic cutoffs per card and asset, a 0.5 drift and 0.85 spread had a mean normalized CRPS-plus-pinball ratio of 0.891 (F1) and 0.928 (F2), relative to the previous 1.0. F3's 21 multi-asset cards had 42 synthetic cutoff checks using the published CRPS, variogram and pinball scorers; 0.85 spread yielded a mean normalized composite of 0.969. These historical diagnostics do not identify the private Development score, and the particular event/shock outcomes remain sealed. F4 is unchanged because its text-cued tail risk is not represented by a generic spread shrinkage.

After the [candidate Action](https://github.com/ellatso/track2-forecasting-public/actions) passes and the template pins its image digest, on Windows check out this branch and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\\participant\\pack-windows.ps1
```

The Team Key is requested at a hidden prompt. Team Number 609's ZIP is written under `Downloads\\Agenthon-T2-609-calibrated\\submission.zip`; keep it out of the public repository. Submit it to Development CodaBench to measure whether this candidate improves on #953358.
