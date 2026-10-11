# Short-horizon research

## Executive summary (read this first)

The October 6 scoring change and October 11 Final notice require a new comparison.
The ranked Final uses UST and G10 FX at 2-12 business days. This workflow rebuilds
M0's published outcome-free divisors and delegates every loss to the official
scorer. It does not ship data, outcomes or research outputs in the image.

Use Python 3.13 and toolkit 2.6.0. Run from this repository with an independent
checkout of official commit a2c93e7b66b8c6a6c429fda81d6bc7873b7c5dec:

```bash
python -m experiments.final_study --official ../t2-official --out ../private-study \
  --start 2003-01-06 --end 2014-12-01 --stride 10 --draws 1024
```

Select a small fixed shortlist on these dates, then pass its JSON to `--configs`
for 2015-2019 confirmation. Freeze the deployment rule before the 2020-2024
comparison. Do not choose another runner-up from the final comparison if a frozen
change fails. Those years were used in prior explorations; do not call them unseen.
`transfer_deployment.py` validates the missing-middle path using actual production
functions and synthetic missing histories. All results belong outside this repo.

The shape weighting for research is 75% full joint grid and 25% a rotating single
cell. It is not the organizers' undisclosed Final roster weighting. Group and year
breakdowns, paired differences and seeds must accompany the aggregate. Overlapping
horizons and related currencies are correlated samples, not independent trials.

The normalizer tests include exact single-cell identities, Monte Carlo joint
moments and the public exemplar values in M0-BASELINE section 4. Forecast tests cover
horizon order, fixed seeds, full-path mixtures and post-cutoff data invariance.
