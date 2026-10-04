## Executive summary (read this first)

Compare distribution shape around the precision/monthly-trend candidate.
Use 4096 Sobol normal draws, five seeds and unchanged means. Compare Student-t
path scales, widths, volatility estimates, variance clocks and whole-path mixtures.
Use organizer scoring and a documented M0 replica on historical pseudo-cases.
These repeatedly explored panel snapshots are diagnostic, not pristine OOS data.
Save all results privately, outside this repository.

Run `python -m experiments.shape_sweep --run-dir ../private-shapes` from the
repository root in the pinned Python 3.13 environment. The primary schedule is
in `distributions.ORIGINS`. A quick structural check uses `--quick-check`.
The baseline is V9, not the older 1000-draw submission.

### Confirmation schedule

The additional origins are 2006-03-01, 2009-03-02, 2010-09-01,
2014-09-01, 2017-09-01, 2019-09-02, 2021-09-01 and 2023-09-01.
Freeze a short list before running them. The comparison shortlist was
V9 / t5-clock1.15 / tailmix0.25 for daily inputs, and
V9 / width0.85 / t5 for monthly inputs. Override `configs` in `shape_sweep`
and `ORIGINS` / `BOUNDS` in `distributions` before calling `shape_sweep.main`.
The private run plan records the overrides; do not present this as untouched data.

### Transfer input diagnostic

Run `python -m experiments.transfer_sweep --run-dir ../private-transfer`.
This holds back two, five or ten years of a target's history, retaining its current
anchor. Its context series are from the same single unit. No practice outcomes
are reconstructed. It compares stale absolute volatility, early log volatility,
rescaling and contemporary context blends. The separate dates are
2010-09-01, 2012-09-03, 2016-09-01, 2018-09-03 and 2022-09-01.
Synthetic G10 performance does not establish EM transfer performance.

### Exploratory return-target subgroup

After inspecting the first two runs, separately test cumulative log-return targets.
Compare V9, t5-clock1.15 and a 50% whole-path mixture. Extra origins are
2007-03-15, 2012-03-15, 2016-03-15, 2018-11-15, 2020-09-15,
2022-03-15 and 2023-05-15. This subgroup was noticed after the earlier results;
do not retrospectively call it a pre-registered hypothesis.

### Scoring and limits

`shape_sweep` calls the unchanged organizer `_score` and `_composite` functions.
It records both per-case clipped M0 normalization and pooled-scale diagnostics.
Pooled scales are a stability diagnostic, not the competition's private scales.
Group averages first average historical cases within an asset basket; baskets
then receive equal weight. The official board instead averages scored cards.

Monthly panels have no historical release vintages. Explicit observation-period
steps are retained. Practice asset/cutoff pairs within seven days are excluded.
All outcomes remain within the original single source unit's published cutoff.
No forecast function receives an outcome, practice ID or another unit's future data.
