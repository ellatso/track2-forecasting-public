## Executive summary (read this first)

`experiments.final_daily --audit-production` evaluates the actual numeric deployment
dispatch before considering a forecast change. It also adds the 7Y/20Y Treasury
basket and two Treasury/FX baskets to the existing seven groups. This is historical
research, not a selected Final image, an official rank estimate, or validation of
House reasoning or official release-value cards. Keep results outside this repository.

### Fixed comparisons

Use 4096 draws and seeds 0, 17, 41 at 1–4 business-day horizons. Compare the previous
research sampler with the actual `candidate.numeric_forecast` function. Two optional
rank templates preserve every marginal sample exactly: 60-observation half-life
correlation and 50% correlation shrinkage toward independence. They change the
dependence among assets and horizons. Two further candidates multiply deviations
from the current sample centre by 0.8 and 1.2, retaining that centre and seed.
These are research settings; the forecast image imports none of this module.

`--audit-online-width` is a subsequent exploratory comparison, motivated by interval
coverage in those same periods. It compares the unchanged production forecast with
per-asset width factors from 60 past one-step forecast errors, bounded to 0.8–1.2.
Each error's mean and variance use only the history before that increment. It keeps
the current forecast centre and does not fit against the origin's future targets.
Its results are exploratory; the inspected periods are not a new independent holdout.
Use `--groups rates_bills,mixed_em` to repeat a denser robustness diagnostic, not to
claim that selecting favorable groups demonstrates an overall Final improvement.

Use `--origin-weekday 0 --stride 1` for a Monday-only Final analogue. The runner
requires an actual observation on that Monday rather than silently moving a missing
Monday back to Friday. Without this flag, a stride of 5 or 20 business days repeatedly
selects the same weekday, so changing a split's start date can confound a comparison
with weekday effects. A Monday-only check remains an explored historical diagnostic.

The runner calls the canonical Track 2 composite with the documented M0 expected
normalizers. Report marginal, joint and pinball-tail terms separately. Calibration
diagnostics include 50/80/95% interval coverage, mean empirical probability-integral
transform (PIT), and mean error divided by forecast standard deviation. Coverage is
a diagnostic, not the competition's tail loss. Missing output is not an improvement.

### Example

```bash
python -m experiments.final_daily \
  --raw /private/raw --official . --out /private/audit-2017-2021 \
  --start 2017-01-03 --end 2021-12-31 --cutoff 2026-10-09 \
  --draws 4096 --seeds 0,17,41 --stride 20 --workers 4 --audit-production
```

Use a second output directory for 2022-01-03 through 2026-10-09. Both periods have
been explored previously; neither is a new untouched holdout. Do not tune further
against the second block and then describe it as independent confirmation.

### Limits that matter

- Only each historical prefix enters estimation. ECB USD crosses retain the published
  quote directions. Mixed baskets use the intersection of actual source dates; no fill.
- Monday–Friday offsets skip origins with absent target dates. This is not a complete
  exchange/publication holiday calendar. Asynchronous and sparse input panels need
  separate testing. Current downloaded histories are not a vintage archive.
- The diagnostic weighting is 75% full joint grid and 25% rotating one-cell slice.
  That slice is not a separately invoked singleton card. Final roster weights are unknown.
- A sample-equivalent numerical fallback can have different Sobol points from the
  old research sampler. Distinguish finite-sample variation from a policy change.
- Release-value cards still require the actual contract: series identity, first-release
  vintage, observation period, seasonality, units and release timing. Passing old
  monthly-level cards does not establish this coverage.
- Mock House responses establish interface robustness only. They do not establish
  predictive value of a prompt or choose whether text should alter a Final forecast.

### Sources

- [Official Final notice](https://github.com/Agenthon-2026/track2-forecasting-public/issues/4).
- [Allowed dependence modelling and pinball tail](https://github.com/Agenthon-2026/track2-forecasting-public/issues/25#issuecomment-5985154153).
- [Guaranteed fields and excluded hints](https://github.com/Agenthon-2026/track2-forecasting-public/issues/20).
- [ECB download](https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip).
- [Treasury XML feed](https://home.treasury.gov/treasury-daily-interest-rate-xml-feed).
- [Scheuerer and Hamill, 2015](https://repository.library.noaa.gov/view/noaa/22327/):
  variogram scores can distinguish correlation errors. This motivates a controlled
  comparison; it does not establish that a new coupling improves financial forecasts.
