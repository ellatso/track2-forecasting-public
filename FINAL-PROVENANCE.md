# Executive summary (read this first)

This Final candidate retains the researched production numeric policy and its
bounded, cited House scenario wrapper. It contains code and fixed prompts only.
No panels, practice answers, historical downloads, credentials, learned weights or
outcome tables enter the image. Runtime tests establish execution, not Final rank.

## Code and policy

The release workflow builds one linux/amd64 image, runs the real forecast verb on
all 126 cards from official revision
`0ddb8896a942b9d44c64927f74755e7012c3e5f9`, and exercises synthetic HTTP replies.
It publishes that same local image, verifies its anonymous pull by digest, and
records the source commit and tested image identity in the handoff artifact.
Toolkit v2.6.1 is pinned. The forecast entry point is `text_candidate`.

The numeric sampler uses 4096 joint Sobol draws, horizon-dependent daily FX and
Treasury policies and the existing monthly damped trend. The new-card and longer
historical comparisons did not justify replacing it. See participant/PS1-REVIEW.md
for the protocol. Results remain outside this public repository.

MXN, BRL, ZAR, KRW and INR use their mounted foreign-units-per-USD series.
AUD, EUR, GBP and NZD use their mounted USD-per-foreign-unit convention.
No source conversion or download happens inside the image. Monthly observation
periods, publication lag and authored output horizon keys are kept distinct;
see docs/MONTHLY-HORIZONS.md. No hidden release schema is guessed.

## House disclosure and resource bounds

The approved model is `nvidia/nemotron-3-super-120b-a12b`, version and revision
`rl-030326-fp8`, training cutoff `unpublished`, access `api`. This is disclosed in
the generated submission descriptor. Runtime calls use injected MODEL_NAME,
MODEL_ENDPOINT and MODEL_TOKEN through the audited proxy at /v1/chat/completions.
No external model, participant adapter or third-party data API is used.

At most three requests are attempted. Each request has a 90-second elapsed-time
ceiling, and all requests share 180 seconds. Thinking is disabled. Responses must
cite quotations found in admitted pre-cutoff text. Half the joint paths remain
exactly numeric; supported shifts are bounded and uncertainty cannot be narrowed.
The detailed fixed scenario bounds are documented in V13-PROVENANCE.md. Monthly
targets retain the numeric policy without House calls. Invalid or unavailable
responses retain the exact numerical forecast.

Synthetic HTTP success, malformed/truncated replies and HTTP errors test these
execution paths. They do not establish the quality of real House predictions.
Only an official run receives live per-unit credentials. Published monthly cards
establish format compatibility, not accuracy on future official release values.

## Submission and verification

The handoff includes an unsigned Final descriptor and Windows packer. The user's
Team Key is entered only at the official local hidden prompt. Nothing here uploads
to CodaBench. Two Final uploads are currently allowed on different UTC days; the
later replaces the earlier. Close is 2026-10-26 11:59 UTC. Keep the exact digest
anonymously pullable until results are published.

Sources: official issue 4, comment 6252030411; shared toolkit docs/HOUSE-MODEL.md
and docs/DEVELOPMENT-RUNTIME.md. Ordinary Docker checks do not reproduce gVisor
or guarantee the organizer's whole-roster runtime.
