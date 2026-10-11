# Participant artifact provenance

## Executive summary (read this first)

This image contains numerical forecasting code, fixed algorithm choices and pinned
libraries. It contains no fitted offline model, model checkpoint, data panel,
practice outcome, lookup table or language-model adapter. It makes no network calls.
All runtime means, covariance matrices and context adjustments are computed from
that invocation's own panels at or before its as-of. Use category `api`, models `[]`.

## Input and research provenance

Public source: Agenthon-2026/track2-forecasting-public, commit
`a2c93e7b66b8c6a6c429fda81d6bc7873b7c5dec`, MIT organizer code. Third-party
panel licenses remain those in the source manifest and THIRD-PARTY-NOTICES.
Research used the rates panel in `t2-F1-hawkish-cut-2024` (through 2024-12-18)
and the G10 FX panel in `t2-F3-election-2024-joint` (through 2024-10-31).
No panels are included in the image. Synthetic historical origins were chosen on
fixed business-day schedules; every prediction received only its historical prefix.
Short-horizon comparisons excluded windows overlapping published practice target
windows. The transfer stress test created its own missing-middle histories.
It was not a scored practice-card target lookup.

Candidate screening covered 2003-2014, confirmation 2015-2019 and a frozen
comparison 2020-2024. The last period had appeared in earlier research at other
horizons, so it is not claimed as pristine unseen data. Hyperparameters were chosen
by these historical comparisons, not by reconstructing official hidden answers.
The two source panel hashes, complete selection plans and local results are in the
private research report. Research scripts are public under `experiments/` but never
copied into the container. No dated empirical forecast, coefficient or outcome is
packaged. These historical inputs precede the announced 2026 ranked Final cutoff.
The image is also usable for Development; historical practice calibration is
covered by the organizer's Track 2 issue 24 ruling, subject to this disclosure.

## Frozen policy

- Use 4096 scrambled Sobol draws, seed 0, raising to the next power of two if the
  card requires more draws. Every draw contains one joint path across horizons.
- Retain V9 for rate levels at horizons up to 12 business days: half the trailing
  mean change, equal sample/EWMA60 variance, empirical correlation.
- Short FX level forecasts: mix 25% of that V9 distribution with 75% zero-drift,
  EWMA20 variance and Student-t8 full-path noise, normalized to the same variance.
- Longer daily rate and FX level forecasts: zero drift with a fixed GARCH(1,1)
  recursion, coefficients omega=0.02 times sample variance, alpha=0.06, beta=0.92,
  variance reverting at alpha+beta=0.98, and variance-normalized Student-t8 paths.
  These are algorithm constants, not stored fitted coefficients.
- When a short FX target has only old increments and a current anchor, use old
  log changes rescaled to that anchor. Mix the old variance equally with its
  adaptation by the median current/old variance ratio of at least two other FX
  series from the same input, ratio clipped to [0.25,4]. No stale context is used.
- Retain V9 for monthly, return and unsupported mixed-domain targets. Monthly
  horizons continue to use explicit observation periods and publication lag.
- No card id, title, difficulty or design note selects forecasts. No document is
  read and no House call is made. Rationale states the exact numerical policy.

## Software

Python 3.13 slim Bookworm; qfbench2-common 2.6.0; NumPy 2.1.3;
SciPy 1.15.3; pandas 2.2.3; pyarrow 18.1.0; jsonschema 4.23.0;
requests 2.34.2. Organizer scorer is imported unchanged for local validation.
The image is pinned by digest after the actual Linux container passes the checks.
