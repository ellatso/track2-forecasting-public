# Executive summary (read this first)

Evaluate the organizer's published short-horizon cards before selecting a Final image.
The official kit revision is `0ddb8896a942b9d44c64927f74755e7012c3e5f9`.
This research retains the existing production numerical policy. Measured results,
forecast files and source downloads belong in the private evidence bundle.

## Reproduce the comparison

Use Python 3.13 with the Dockerfile's numerical dependency versions and toolkit
v2.6.1. The latter fixes native Windows packing and preserves the scoring contract.
Check out the official kit separately, including its published `practice-answers/`.

```bash
OPENBLAS_NUM_THREADS=1 python -m experiments.practice_ps1 \
  --official /absolute/official-kit --out /absolute/private/ps1-initial
```

The seven fixed policies compare production, bounded online width calibration on
bills or all daily targets, exponentially weighted or shrunk correlation, a
zero-drift historical-covariance forecast, and an EWMA20 Student-t8 forecast.
Every policy uses 4096 draws and seeds 0, 17 and 41. It preserves card horizon and
asset order. All forecast bytes are sealed before the answer loader runs.
The canonical organizer verifier checks every generated output. Component losses
use the canonical scorer and the published input-only M0 expected-error divisors.
These reconstructed local scores are diagnostic, not an official rank.

`--mixtures` adds exploratory 25% and 50% whole-path mixtures of production and
the zero-drift historical-covariance forecast. These were motivated after the
initial ps1 comparison; do not call them preregistered ps1 candidates or claim
the same 22 cards are an untouched validation set.

```bash
python -m experiments.download_final_history --out /absolute/private/raw
# Download eurofxref-hist.zip from the ECB and save it as raw/ecb-history.zip.
OPENBLAS_NUM_THREADS=1 python -m experiments.ps1_confirmation \
  --raw /absolute/private/raw --official /absolute/official-kit \
  --out /absolute/private/monday-diagnostics
```

The second runner compares production, zero drift and both mixtures on six
asset baskets at Monday origins, with 1–4-business-day targets. It excludes ps1
target weeks, requires actual target observations and reports both temporal
blocks. The periods have been explored before; this is a robustness check.
The complete group-and-period evidence does not justify replacing production.

## Final preparation implications

- The 18 October UTC organizer announcement permits two Track 2 Final uploads,
  at most one per UTC day. The later upload becomes the entry. Close remains
  26 October at 11:59 UTC, or 19:59 in Taiwan.
- New practice G10 panels still use H.10; Final G10 uses ECB. The historical
  diagnostic uses source-consistent ECB USD crosses with the published quote
  directions.
- Successful local gates establish interface compatibility. This environment
  cannot establish the predictive quality of live House reasoning. Disabling
  model access checks only the numerical fallback and document-processing path.
- No monthly official release-value accuracy is established by the 22 daily
  cards. The monthly contract still requires observation period, unit and vintage
  correctness. See the organizer's `docs/MONTHLY-HORIZONS.md`.
- This comparison does not publish a Docker image, choose a new image digest,
  sign a team claim or upload to CodaBench. No Team Key is needed for research.

Official announcement:
https://github.com/Agenthon-2026/track2-forecasting-public/issues/4#issuecomment-6252030411
