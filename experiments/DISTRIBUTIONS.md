## Executive summary (read this first)

Run 17 frozen daily distribution methods on the same explored historical origins. The baseline reproduces the numeric half-drift submission candidate exactly. This research branch changes no image, submission descriptor or monthly forecast. Keep all output outside GitHub.

### Fixed controls
Every component uses drift multiplier 0.5, 300 observations and correlation shrinkage 0. Defaults are 1,000 draws and seeds 0/17/41. Baseline volatility is equal sample/EWMA60 variance. No House calls or learned artifacts are used.

### 17 methods
Nine pure components cross normal/Student-t(df=5)/Student-t(df=8) with variance-clock powers 0.8/1.0/1.2. The clock leaves the first 20 observations unchanged and uses 20*(h/20)^power afterward. Drift remains linear in actual observation count. Positive clock increments preserve coherent joint paths.

Four volatility mixtures allocate 25%/50% of paths to sample-only or EWMA-only volatility. Two tail mixtures allocate 25%/50% to Student-t(df=5) with baseline volatility. One clock mixture allocates 50% each to powers 0.8/1.2. One joint mixture uses 50% baseline, 25% Student-t(df=5)/clock0.8 and 25% Student-t(df=8)/clock1.2. All mixture weights are fixed before execution.

Student-t components multiply the Gaussian path residual by sqrt((df-2)/ChiSquare(df)). This preserves theoretical covariance and drift. A draw shares its latent scale across every asset and horizon. Mixtures select complete paths with a separate deterministic random stream; they do not average quantiles or sample values. Finite-draw empirical variance need not be exactly equal across arms.

### Diagnostic protocol
Reuse the previous ten origins from 2011 to 2023, plus 2024-01-02. January 2024 has already been inspected: it is diagnostic, not untouched validation. Outcomes stay within each origin's calendar year and the source card cutoff. All candidates share eligible cases; failed cases are reported. Public history coverage can differ across years.

Shared organizer functions compute marginal, joint and tail losses. Normalize per case/seed to this round's current baseline, then average cases within asset baskets and baskets equally. Relative loss below 1 is better. These scales are not the private organizer M0 scales; local loss is not an official leaderboard score. Component/family/horizon tables are descriptive case or cell means, as indicated by their names. Overlapping baskets and many comparisons limit inference; no method is promoted automatically.

Input, source, environment and parameter hashes are frozen before execution. Outputs cannot be overwritten. A checkpoint is saved after each basket. Monthly retains the prior baseline and is outside this daily comparison.

### Windows reproduction
Clone into a separate checkout:

```powershell
git clone --branch research/t2-distributions --single-branch https://github.com/ellatso/track2-forecasting-public.git "C:\Users\ella.tso\Downloads\agenthon-t2-distributions"
powershell -ExecutionPolicy Bypass -File "C:\Users\ella.tso\Downloads\agenthon-t2-distributions\experiments\run-distributions-windows.ps1" -Phase benchmark
```

Outputs appear in Downloads/Agenthon-T2-Research-V7. Add `-Phase all` to open the notebook after a successful benchmark; `-Phase notebook` opens the latest successful run. QuickCheck runs one explored date, 200 draws and seed 0; it is only a runtime check. Do not upload the research-results ZIP to CodaBench.
