# Executive summary (read this first)

Freeze the three policies, input bytes, runtime and event-block splits before
scoring. Published outcomes stay outside the forecast input snapshots and outside
this repository. Missing outputs remain in the denominator at loss 8.0. No House
access or mock responses can establish text uplift. This runner never creates a
submission ZIP and never uploads to CodaBench.

## What is ready and what is still required

`compare_final.py` supplies input isolation, a fixed comparison, official gates
and scoring, diagnostic tables and sequential candidate locks. It accepts the
new kit through an explicit private catalog. The kit is not yet published as of
14 October; its actual layout, release-card contract and M0 scale availability
must be checked when it arrives.

Real text-quality evaluation needs actual authorized House-produced forecasts.
This runner does not create per-unit House credentials or borrow one unit's
allocation for other cards. `smoke` deliberately removes House configuration and
only checks numerical fallback and execution. A user-supplied replay/live label
is producer provenance, not a provider signature; the result states this limitation.

## Private catalog

Create a UTF-8 JSON file outside the checkout. Relative paths resolve beside it.
Use each unit's expected-error M0 divisors under the October 6 scoring rule,
never raw losses or M0's realized error on that card. Scales are not reconstructed
by guessing a release card's numerical units. Missing references or scales prevent
scoring; they never remove a difficult card from an average.

```json
{
  "normalization": "m0-expected-error",
  "kit_revision": "the actual organizer revision",
  "units": [
    {
      "unit": "kit/units/card-directory",
      "realized": "private/card-outcomes.parquet",
      "scale": "private/card-scales.json",
      "event": "one specific historical episode",
      "target_end": "YYYY-MM-DD"
    }
  ]
}
```

Supply every intended card. At least three distinct complete date/event blocks
are required; more are needed for persuasive evidence. `event` is optional and
identifies a specific episode, not a broad family such as "FOMC". Daily target
endpoints can be derived from authored business-day horizons. Monthly/release
cards require their explicit endpoint. Planning-only catalogs may omit reference
paths, but cannot later be scored without making a new complete plan.

Download organizer archives without changing text line endings. Windows Git
autocrlf can change bytes and therefore fail the published checksums. The planner
checks manifested inputs and makes a separate filtered research snapshot; original
organizer files are never edited. Answer files and reference scales are not copied.

## Commands

Use the pinned Python 3.13/common-2.6.0 research environment. The PowerShell wrapper
creates a separate environment and keeps everything under Downloads by default:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run-final-comparison-windows.ps1 -Action plan -Catalog 'C:\private\catalog.json'
```

The path above is a format example, not an existing catalog. Later commands can
use the last successful plan, or accept an explicit `-RunDir`. On any platform:

```bash
python -m experiments.compare_final plan --catalog /private/catalog.json --run-dir /private/run
python -m experiments.compare_final smoke --run-dir /private/run
python -m experiments.compare_final seal --run-dir /private/run
python -m experiments.compare_final score --phase select --run-dir /private/run
python -m experiments.compare_final score --phase confirm --run-dir /private/run
python -m experiments.compare_final score --phase test --run-dir /private/run
```

`smoke` writes forecasts for execution checks only. For an actual text comparison,
omit it and place already produced predictions at
`forecasts/<numeric|previous_text|timed_text>/<seed>/<handle>/` before sealing.
The plan lists every handle. Each output directory has exactly the three standard
forecast files, with 4096 draws and seeds 0, 17 and 41 fixed in the plan.

Keep producer provenance in a separate `execution.json`, keyed by
`variant/seed/handle`, with `mode`, `input_digest` and `successful_replies`.
Modes `house_live` or `house_replay` require genuine corresponding outputs;
`mock` and `unavailable` are never selection evidence. The forecast metadata must
also record attempted House calls. Timed outputs must declare their event timing
and ECB/UST clock profile. Absence of a verified adjustment is a permissible
abstention, not automatically a model failure.

## Split, score and selection semantics

Overlapping future windows and explicit shared-event labels form connected groups.
Their entire date envelopes also stay together, preventing a shared event from
placing a later training card after an earlier test card. Chronological groups
are divided into selection, confirmation and held test thirds.

All forecast bytes are sealed before outcomes are scored. Changed input, output,
reference, runtime or code bytes cause refusal. Each phase starts once. After a
failed data check, inspect the error and preserve the run; do not relabel an
already observed period as a fresh holdout in a replacement plan.

The unchanged canonical scorer applies all four gates, exact grids, single-cell
weight redistribution and component normalization. Card losses are clipped at 8;
failed/missing participant forecasts count 8. Reference defects abort the phase.
Results remain explicitly unrankable because there is no signed Final roster.

Selection compares the three policies; confirmation and test score only the
numeric baseline and the previously locked candidate. Require at least 1% lower
event-block mean loss, marginal/tail ratios at most 1.03, no event-block loss ratio
above 1.05, complete admissibility, and text execution evidence. Otherwise retain
the numerical policy. A failed numerical baseline marks the comparison invalid.
The held test is a final go/no-go check; no alternative is searched there.

The summary reports both equal-event and unit-average loss. Per-horizon marginal,
joint and tail tables are **raw**, for paired comparison within the same card and
horizon only. Do not average their differently scaled units into a leaderboard
score. The event-block bootstrap is descriptive; it supplies no interval with
fewer than three blocks and does not make shared financial series independent.
