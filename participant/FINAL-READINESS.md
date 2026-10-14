# Executive summary (read this first)

This research branch prepares the revised Final. It is not a selected Final image,
does not supply a submission ZIP, and never uploads to CodaBench. Preserve the one
Final upload while the promised short-horizon practice kit is still unavailable.
Outcome tables and research scores stay outside this public repository.

## Binding notice checked on 14 October

The [12 October correction, updated 13 October](https://github.com/Agenthon-2026/track2-forecasting-public/issues/4#issuecomment-6129776242)
replaces the earlier ranked-card description. The main README still describes older
arrangements. The Final now uses 1-4 business-day forecasts and USD-converted ECB
currencies, including MXN, BRL, ZAR, KRW and INR. Treasury adds 3M, 6M and 1Y par
curve tenors. Official release cards require forecasts of the defined release value.
They are not ordinary monthly level extrapolations.

The [13 October limits correction](https://github.com/Agenthon-2026/track2-forecasting-public/issues/4#issuecomment-6150556139)
reduces the whole run allowance to 43,200 seconds, while each unit retains 1,800
seconds. The ranked roster arrangement is still to be announced.

## Changes in this branch

- Add MXN, ZAR and KRW to the existing input-only currency policy and quote descriptions.
- Bound each Linux House request to 90 seconds of elapsed time, including slow streams.
  All requests share 180 seconds; retain the numerical forecast on failure. At most
  three requests are attempted. Windows signing does not need the model client.
- Supply `experiments/final_daily.py` for a reproducible source-consistent study.
  It reads local official downloads, uses the organizer's scoring functions, and
  excludes absent holiday targets and windows overlapping existing practice cards.
- Supply opt-in cited event timing and explicit ECB/UST observation clocks, with
  release-unit/vintage helpers. See [EVENT-RESEARCH.md](EVENT-RESEARCH.md). These
  are research options; the Docker entry point and selected image are unchanged.

## Numeric experiment

Download historical data from the official
[ECB reference-rate page](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html)
and [Treasury XML feed](https://home.treasury.gov/treasury-daily-interest-rate-xml-feed).
The research runner accepts `ecb-history.zip` and `treasury-YYYY.xml` files in `--raw`.
Treasury requests must use `data=daily_treasury_yield_curve`, not bill discount rates.
These research downloads are not copied into the submission image.

```bash
OPENBLAS_NUM_THREADS=1 python -m experiments.final_daily \
  --raw /absolute/private/raw --official /absolute/official-kit \
  --out /absolute/private/new-selection \
  --start 2007-01-01 --end 2014-12-31
```

Use a new, private directory for every output. Never commit outcomes, summaries,
cached data or input credentials. The runner writes its configuration and source
hashes before scoring. `--locked` accepts a JSON mapping of basket names to one
frozen candidate ID and compares it with the legacy baseline. Keep 1, 2, 3 and 4
as authored horizon keys. ECB conversions use the same date's USD/EUR reference;
do not splice H.10 observations into these panels.

## Remaining selection work

1. Verify the organizer's new kit and exact asset IDs, units, release definitions,
   information vintages and forecast-grid requirements.
2. Reserve complete historical event/date blocks before evaluating outcomes.
   Cards that share an event or future window stay in the same block.
3. Compare the fixed numeric baseline with a small set of cited-text mixtures.
   Use the same 4096 draws and seeds 0, 17 and 41. Record marginal, joint and tail
   components separately; don't select from a few favorable cards.
4. Treat event timing explicitly. A future scheduled decision is not a known
   surprise and must not be substituted for a Treasury yield or FX movement.
5. Validate release forecasts in their specified units. GDP annualized growth,
   a price index, a monthly inflation change and a weekly claims count require
   different transformations and uncertainty scales.
6. Check the actual Linux image on every new card and the previous card suite.
   Measure live House latency and failures; mocks establish interface behavior only.
7. Select one verified digest, complete the required model disclosure, and sign
   locally with the hidden Team Key prompt. Keep the image anonymously pullable
   until the organizer publishes results. No Final upload is made by this branch.

No Development ranking or local diagnostic score establishes a Final rank.
