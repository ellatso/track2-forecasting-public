## Executive summary (read this first)

V12 addresses two previously unsupported numerical paths: cumulative factor
returns and long-horizon FX histories with a missing middle. This is a
Development follow-up. It does not change the short UST/G10 policy and does
not claim a leaderboard rank or a Final improvement.

`factor_followup.py` compares conditional volatility, Student-t tails and
horizon-dependent variance with a fixed half-strength drift. A second
configuration file can compare means while holding the distribution fixed.
`long_gap_followup.py` creates synthetic target-history gaps of 2, 5 and 10
years, retains the current anchor and reads only other currencies' prefixes.
It compares old-history drift, zero drift, current-anchor rescaling,
context adaptation, tails and volatility floors.

Both call the organizer's unchanged composite loss with outcome-free M0
normalizers, clipping each synthetic task at eight. The factor comparison
weights single-asset groups and the joint basket equally. FX results weight
currencies equally. Their numbers are not official leaderboard scores.
Private target windows are never reconstructed. Historical targets overlapping
any published practice target for that asset are omitted entirely.

Example training commands, from the repository with its dependencies installed:

```bash
python -m experiments.factor_followup --official ../official-kit --out ../private-factor-train
python -m experiments.long_gap_followup --official ../official-kit --out ../private-gap-train
```

Use `--start 2015-01-05 --end 2019-12-02 --draws 4096 --seeds 0,17`
for confirmation. Pass a JSON list of explicit configurations with `--configs`.
The selected factor pair is `v9_t0_d0.5` (vol=v9, df=0, drift=0.5, window=300)
and `garch_t8_d0.5` (vol=garch, df=8, drift=0.5, window=300).
`--production` runs this pair through the real CLI fallback and factor module.
The selected gap pair is `v9` (drift=0.5) and `context50_t8`
(logscale=true, context=0.5, df=8). No floor is enabled.

Keep scores and selection records outside this checkout. Report the previously
explored status of 2020-2024 rather than calling those years unseen. Synthetic
G10 gap evidence does not establish emerging-market transfer quality. An
official run is still needed to evaluate the selected image on Development.
