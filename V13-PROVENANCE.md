## Executive summary (read this first)

V13 is a cited-text Development trial around the unchanged V12 numeric forecast.
It packages code and fixed prompts, no panels, outcomes, learned weights or
offline fitted artifacts. It calls only the organizer's injected House endpoint,
at most three times per unit. A missing or invalid reply preserves V12 exactly.
Its live-model forecast quality has not been validated off-platform.

## Approved model and route

The descriptor discloses the approved House model as
`nvidia/nemotron-3-super-120b-a12b`, version and revision `rl-030326-fp8`,
training cutoff `unpublished`, access `api`. These values come from the official
`Agenthon2026-public/docs/HOUSE-MODEL.md`. The approved-base exception applies
to that House revision; no participant tuning or model weights are packaged.

Runtime calls use only injected MODEL_ENDPOINT, MODEL_TOKEN, MODEL_NAME and
the authenticated audited proxy. Calls post to `/v1/chat/completions`, use
temperature zero, a fixed seed and disabled thinking, and request at most
3,600 output tokens. No vendor tools or third-party APIs are used. No endpoint,
proxy login, bearer credential or Team Key is baked into the image.

## Inputs and numerical policy

Numerical calibration and data sources are documented in the retained
`ARTIFACT_PROVENANCE.md` for V12. This trial retains that complete numerical
policy, including its horizon handling and 4,096 correlated Sobol draws.
The cited-text wrapper shares the same numeric function; offline checks compare
every released card's fallback with the numeric output exactly.

Every numerical observation and document comes only from the current task's
mounted files at or before asof. Corpus timestamps and paths are checked;
unknown, future-dated and out-of-directory documents are rejected. At most
12 admitted documents and 24,000 excerpt characters enter the model prompt.
No card identifier, title, difficulty, design note or family category is sent.

## Text policy and fixed bounds

House first extracts statements with verbatim quotes and source document IDs.
Only quotes found in the admitted text become evidence. House then proposes
joint scenarios with probabilities and per-asset effects citing that evidence.
There are at most four scenarios. Invalid numbers, weights, quotes or references
are rejected. Source occurrence checks do not establish causal truth.

Half of the joint draw distribution remains exactly V12. The other half can
shift its centre or increase its conditional uncertainty. Confidence is capped
at 0.75. Longest-horizon shift requests are capped at 0.75 standard deviations
for horizons at most 12 business days and 3 for longer horizons, then multiplied
by confidence. Conditional standard-deviation requests are clipped to [1,3].
An unsupported asset retains its original draws. Scenarios are shared across
all requested horizons and assets, with shifts scaled by authored horizon.
Monthly inputs keep the complete numeric policy and make no House calls.

Stale FX histories may form a contemporary risk bound from at least two other
currently observed FX series in that same input. It is clipped to six times
the actual V12 forecast standard deviation. This bound is used only inside an
evidence-cited scenario explicitly asserting a current regime change; it is
not an unconditional volatility floor. H.10 quote directions respect the
original series convention. The known H.10 four USD-per-foreign-unit series
are AUD, EUR, GBP and NZD; other supported FX series are foreign-units-per-USD.

These are fixed experimental guards, not fitted or validated optimal parameters.
The wrapper was motivated by Development feedback that numerical variance
changes left important capped losses unresolved. It contains no per-unit
choice, hidden target reconstruction, outcome-derived constant or answer table.
We do not claim an expected official score, top-ten rank or Final gain.

## Validation scope

Synthetic unit tests cover exact fallback, quote rejection, malformed replies,
cutoff isolation, stale peer risk, joint paths and bounded effects. Actual
container checks run all released cards with no House credentials and then
with synthetic, source-cited replies, verifying both output paths through the
organizer's unmodified public gates. HTTP/proxy tests verify the route shape.
Those checks validate execution and the numerical transformation, not the
quality or reliability of real House interpretations.
