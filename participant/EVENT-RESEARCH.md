# Executive summary (read this first)

This is an opt-in event-time research extension, not a selected Final image.
It adds causally timed scenario paths and separate release-unit utilities.
Neither a clock check nor a unit conversion establishes forecast-quality gains.
The official short-horizon kit is still awaited as of 14 October 2026.

## Event dates and market observations

The original text policy changes all horizons. The new option can preserve every
pre-event draw exactly and start a conditional event effect later. The same draw
keeps one joint scenario across assets and horizons, with half the draws retained.

```bash
python -m qfbench2_track_forecasting.text_candidate \
  --panels /input/panels --text /input/text --asof YYYY-MM-DD \
  --out /output/forecast.parquet --event-timing --event-clock ecb-ust
```

Select the source profile explicitly, after checking the actual card. `ecb-ust`
uses 14:10 Europe/Berlin for FX and approximately 15:30 America/New_York for
Treasury. This gives the announced Final ECB fix of 13:10 UTC. The two regions
can change daylight-saving time on different dates. The profile is unsuitable
for H.10 noon observations or unknown sources.

An event dated Wednesday at 14:00 New York time can affect Wednesday's Treasury
observation but only Thursday's ECB fix. The scenario's `activation_date`,
`activation_time` and `activation_timezone` must occur together in a verified
quote. Supported clock zones are UTC, America/New_York and Europe/Berlin.
Missing, invented, past or out-of-window activation evidence rejects that asset's
event adjustment. It is not replaced by a guessed event time from model memory.

The simpler `--event-clock date` is a date-only diagnostic. It cannot distinguish
an event before versus after the same day's fix and should not be taken as a
verified intraday Final policy. Without `--event-timing`, the previous behavior
is retained. The Docker entry point has not been switched to either research option.

In the event scenario, the mean shift is a persistent level jump. Only later
increments are scaled. For an authored interval straddling an event, variance
is divided by business-day steps, a Brownian interpolation assumption. Weekends
are rolled forward; no market-holiday or intraday path is invented. New kit dates
and clock conventions must be verified before adoption.

References: [ECB reference-rate timing](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html),
[Treasury observation timing](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/).

## Official-release units and vintages

`release_units.latest_vintage` requires publication dates and observation periods
for one series. It excludes later releases/revisions, future observation periods
and conflicting same-timestamp values. A current revised series without its
publication history cannot pass as a historical real-time vintage.

`release_units.transform_draws` converts hypothetical level draws to:

- unchanged levels or explicitly scaled counts;
- period-to-period percentage change;
- year-over-year percentage change using the matching earlier period;
- annualized quarterly percentage change, compounded to the fourth power.

It refuses missing comparison periods, nonpositive growth denominators and
unknown unit conventions. The caller must select the correct series, SA/NSA
adjustment, unit scale and vintage. A future revision of the comparison period is
not modeled here. These functions are not a nowcasting model, are not wired into
the forecast CLI, and do not guess the unreleased macro-card contract.

References: [BEA quarterly annualization](https://www.bea.gov/help/faq/122),
[BLS CPI methodology and FAQs](https://www.bls.gov/cpi/questions-and-answers.htm).

## Selection condition

Use new historical cards in complete event/date blocks. Fix the numeric baseline,
sample count, seeds and candidate flags before evaluating held blocks. Compare
text off, the previous text policy and the timed policy under the same inputs.
Report marginal, joint and tail loss separately. Mocks prove execution only.
No Final upload or image publication is performed by this branch.
