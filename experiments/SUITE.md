## Executive summary (read this first)

Run one complete, frozen 21-method daily comparison and a later-period check.
The batch includes drift, variance and correlation controls, marginal and composite
inner selection, and a calibrated numeric control. Monthly forecasts and official
images stay unchanged. Public data only support limited January 2024 validation.
This is research evidence, not an official leaderboard score.

### One command sequence on Windows

```powershell
$repo = "C:\Users\ella.tso\Downloads\agenthon-t2"
git -C "$repo" pull --ff-only origin research/t2-experiments
powershell -ExecutionPolicy Bypass -File "$repo\experiments\run-suite-windows.ps1" -Phase all
```

The benchmark automatically runs diagnostics, writes a frozen shortlist, checks
source/input/plan locks, then evaluates the shortlisted methods on January 2024.
It opens the notebook afterward. Run all notebook cells and save. Outputs remain
outside GitHub in Downloads/Agenthon-T2-Research-V6. Phase notebook opens the latest
successful run; phase benchmark runs both numeric stages without opening Jupyter.
QuickCheck is a CI exercise on one explored date, 200 draws and seed 0; it does not
run later validation and its decision explicitly refuses promotion.

### Frozen comparisons

The fixed grid is drift multiplier 0/0.5/1 times variance mixture 0/0.5/1 times
correlation shrinkage 0/0.25: 18 configurations. Variance mixing interpolates sample
variance and exponentially weighted variance with a 60-observation half-life.
Correlation shrinkage moves the sample matrix toward identity. All use 300 history
observations and a linear variance clock. The unchanged baseline is one of these arms.
Three controls bring the total to 21:

* dynamic_marginal preserves the previous adaptive implementation exactly.
* dynamic_composite chooses the same weights 0/0.125/0.25/0.375/0.5 with the shared
  marginal, joint and tail composite. Its drift stays 1 and shrinkage stays 0.25,
  matching dynamic_marginal so only the inner objective changes.
* calibrated_numeric_control reproduces the earlier family-dependent numeric policy.
  It is not identified with a particular official submission: ZIP/image mapping
  remains unverified. It contains no House reasoning.

Both dynamic controls use up to six completed inner forecasts, spaced 21 observations,
256 draws, seed 71 and a smaller-weight tie preference. They receive only their own
case's pre-origin history. The composite control averages baseline component scales
across completed inner cases and imports effective weights/composite math from the
existing scorer. Degenerate scales or insufficient inner history fall back to mix 0
without dropping the outer case. These scales differ from private organizer M0 scales.

### Diagnostic and later-period evidence

Outer defaults are 1000 draws and seeds 0/17/41. Diagnostics reuse the ten origins
from 2011/2013/2015/2017/2018/2019/2020/2021/2022/2023 with within-year target limits.
Their parameter ranking is exploratory and subject to multiple comparison bias.
The shortlist is baseline, the three controls and the top three diagnostic methods,
deduplicated. It is saved before any outer January 2024 target is accessed.
The later origin is 2024-01-02, with all targets ending before 2025-01-01 and before
the original card cutoff. At most eight daily baskets have histories reaching 2024;
long horizons and gaps can reduce actual eligible coverage further. One later origin
and overlapping baskets cannot establish robust improvement; promotion is always false.
No later target is used to adjust the shortlist or parameters.

Summaries average cases within baskets and then baskets equally. Family/seed reports
are descriptive case means and need not equal basket-weighted headline scores.
Paired intervals are descriptive; baskets overlap. All arms use identical eligible
cases. Monthly data remain a separate, unchanged baseline and are not evaluated here.

### Private outputs

plan.json, shortlist.json, validation_started.json and decision.json record the protocol
and decision. diagnostic_/validation_ prefixes include cases, cells, summaries, yearly
and family/seed breakdowns, weight audits, coverage and paired intervals. Inputs and code
hashes are recorded before execution. Existing runs cannot be overwritten. Keep all
result files outside the public repository. No Docker image or submission ZIP is built.
