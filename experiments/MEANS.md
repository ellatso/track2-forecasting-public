## Executive summary (read this first)

Compare 17 daily and 17 monthly mean models with the current numerical submission. Hold Gaussian volatility, covariance, seeds and draws fixed. Only conditional means change. Historical dates were explored before; this round does not create a competition submission.

### Frozen comparisons
Both frequencies include the current baseline, zero drift, six autoregressive mean-reversion settings, six damped trends and three half-strength trend blends. An autoregressive mean reverts a stretched level toward an equilibrium fitted only on the prefix. A damped trend continues a recent slope with progressively smaller monthly or business-observation increments.

Daily reference: 300 observations, half sample drift, equal sample/EWMA60 variance, empirical correlation, linear variance clock. Monthly reference: unchanged production monthly CLI with full sample drift and sample volatility. Defaults remain 1,000 draws and seeds 0/17/41.

Daily reversion windows are 120/300; monthly windows are 60/120. Reversion strengths are 0.25/0.5/1 relative to the current reference mean. Level autoregression persistence is clipped to [0,0.995]. Factor log-return persistence is clipped to [-0.5,0.5], reverting toward the reference mean return rate; cumulative targets remain sums of log(1+return).

Daily trend windows are 60/120 with damping half-lives 20/60/120 observations. Monthly trend windows are 12/24 with half-lives 3/6/12 months. Daily slopes are scaled by 0.5; monthly slopes by 1. Linear regression fits level slopes; recent mean log return supplies factor slopes. Three additional settings blend the shorter-window trend with the reference mean at 50%.

Every candidate adds a deterministic mean shift to the same baseline samples. Mean-reversion uncertainty is deliberately not reduced here: this first experiment isolates the conditional mean. Volatility and correlation are never re-estimated from autoregression residuals.

### Monthly limitations and target keys
Use the explicit per-asset monthly observation-step grid from each source card. Preserve authored horizon keys in all reports; never divide the keys by assumed business days per month. The baseline calls the actual production CLI, including noise in months without targets.

Historical panels are fixed snapshots, without a historical release/vintage ledger. Monthly backtests are observation-index diagnostics, not forecasts using proven then-available releases. Original step patterns are reused for older synthetic observation origins; this is not a reconstruction of old task metadata. A winning monthly method cannot establish causal OOS improvement and is not automatically promoted.

### Balanced diagnostic protocol
Use the same eleven explored origins from 2011 through January 2024. Targets remain within the origin year and source cutoff. Evaluate frequencies separately; no pooled daily/monthly winner. All candidates within a frequency use the same eligible cases. Inputs, code, environment and settings are hashed before execution; changed inputs/code refuse completion. Use a fresh directory; existing outputs cannot be overwritten.

Import shared organizer scoring. Normalize each case/seed to the current frequency-specific baseline, then average within baskets and baskets equally. Local relative loss below one is better but cannot be converted into an official score. Component/family/horizon tables are descriptive case/cell means; unused component ratios may be zero. Overlapping baskets and repeated model comparisons limit inference.

### Windows reproduction
```powershell
git clone --branch research/t2-mean-models --single-branch https://github.com/ellatso/track2-forecasting-public.git "C:\Users\ella.tso\Downloads\agenthon-t2-mean-models"
powershell -ExecutionPolicy Bypass -File "C:\Users\ella.tso\Downloads\agenthon-t2-mean-models\experiments\run-means-windows.ps1" -Phase benchmark
```
Outputs remain in Downloads/Agenthon-T2-Research-V8. Use Phase all to open the notebook afterward, or Phase notebook for the latest successful run. QuickCheck uses one explored date, 200 draws and seed 0; it is a runtime exercise. Do not upload the research-results ZIP to CodaBench.
