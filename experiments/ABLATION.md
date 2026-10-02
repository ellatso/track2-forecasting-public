## Executive summary (read this first)

Run a matched four-arm comparison that separates daily variance mixing from
correlation shrinkage. Drift, history length, variance clock, seeds and draw counts
stay fixed. Monthly forecasts stay unchanged. These dates were explored before;
this experiment diagnoses mechanisms and cannot establish independent improvement.

### Run on Windows

```powershell
$repo = "C:\Users\ella.tso\Downloads\agenthon-t2"
git -C "$repo" pull --ff-only origin research/t2-experiments
powershell -ExecutionPolicy Bypass -File "$repo\experiments\run-ablation-windows.ps1" -Phase all
```

Results stay under Downloads/Agenthon-T2-Research-V5, outside the repository.
After the notebook opens, run all cells and save it. Phase benchmark runs numeric
experiments only; phase notebook opens the latest successful result.

### Four arms

| Arm | Variance mixture | Correlation shrinkage |
| --- | --- | --- |
| Baseline | 0 | 0 |
| Variance only | 0.5 | 0 |
| Correlation only | 0 | 0.25 |
| Both | 0.5 | 0.25 |

A mixture of 0.5 gives equal weight to sample variance and exponentially weighted
variance with a 60-observation half-life. Shrinkage of 0.25 blends the sample
correlation matrix with 25% identity. The original baseline path is preserved.
All arms use a 300-observation window, drift multiplier 1, linear variance clock,
1000 draws and seeds 0, 17, 41. Use exactly the adaptive round's ten outer dates,
year-end target limits and historical-gap checks, keeping case coverage comparable.
There is no inner optimization or winner selection.

### Read the effects

For example, shrinkage_with_variance equals the loss of both minus variance only.
A negative difference means the added shrinkage reduced loss. Each effect is
paired by grid, basket, origin and seed. Duplicate or incomplete pairs are refused.
Average cases within each asset basket first, then weight baskets equally.
Component columns are weighted contributions using the existing scorer's effective
weights, including its single-cell weight adjustment. No scoring formula is added.

The interaction equals both minus variance only minus correlation only plus baseline.
It measures whether their joint change differs from the sum of isolated changes.
It does not establish a market causal mechanism. Correlation changes can also alter
finite Monte Carlo marginal samples; common seeds reduce noise but do not eliminate it.

The plan records source/input hashes and environment before execution. Existing
runs cannot be overwritten. Outputs include diagnostic_cases/cells/summary/by_year,
effect_cases/summary/by_year, coverage, and descriptive paired intervals. Overlapping
baskets and previously explored dates prevent independent statistical claims.
Keep all outputs private and outside GitHub. No image or competition ZIP is built.
