## Executive summary (read this first)

This image contains numerical forecasting code and a deterministic reader for
dated BLS releases in the current unit. It contains no fitted model file,
neural weights, practice answers, cross-unit lookup or external data. The
descriptor uses category `api` and `models: []`. No House call is made.

### Numerical method and selection history

The numerical forecast is the unchanged precision/monthly-trend candidate from
branch `submission/t2-precision-monthly50`, commit
`9d24c59d5e4d4049ff6b14e59b04de75700868ea`.
It uses at most 300 observations, half daily drift, an equal mixture of sample
and EWMA60 variance, empirical correlation and 4096 scrambled Sobol paths.
Monthly means blend full trailing drift with a 12-observation slope, damped with
a 12-month half-life, at equal weight. Missing monthly history keeps the original
mean. There are no per-unit or date-specific model parameters.

These design choices were compared on prefixes of the public panel histories
in the official Track 2 repository, covering 2000–2024. Historical pseudo-cases
use only observations preceding their origin; outcomes remain within the source
unit's original cutoff. Practice cutoff/asset pairs within seven days are excluded.
The research repeatedly explored these years and is not pristine out-of-sample
validation. Monthly data are fixed snapshots without historical publication vintages.
This is development research, not a claim of sealed-set performance.

The current research compared width, Student-t path scales, variance clocks,
volatility mixtures and simulated missing-history inputs. A separately scheduled
set of historical origins did not support replacing the existing parameters.
No new fitted parameter or calibration file from those experiments is packaged.

### Published observations from the current input

`release_inputs.py` uses authored parsing rules, with no fitted coefficients.
It reads only `macro_release` documents labelled `BLS` in the current input's
`text/corpus_index.json`. The publication must be on or before `--asof`.
The release must identify the month immediately after the panel's last month,
and the month must precede every requested target observation.

Unemployment is read as a stated level. CPI monthly seasonally adjusted percent
change is applied to the previous panel level. That produces an approximate
index: rounding and historical revisions can differ from a later vintage.
Annual rates and unadjusted index levels are not substituted. Duplicate or
ambiguous releases are ignored. No existing observation is overwritten.

The candidate resolves the explicit target periods again and requires exactly
one fewer monthly step for every augmented asset. Authored output horizon keys
stay unchanged. Source quotes, publication dates, reference months and numerical
interpretations are written to the per-run rationale. Text is read at inference
time; none is bundled in the image.

The opportunity to use newly published figures in lagging monthly inputs was
identified while reviewing `mask123123/agenthon`, commit
`5594a751792d0651e7aa517e7347d0174170afde`. The parser here is separately implemented;
no fitted coefficients or forecast outputs from that project are used.

### Components

The image pins qfbench2-common 2.4.4, NumPy 2.1.3, SciPy 1.15.3,
pandas 2.2.3, PyArrow 18.1.0, jsonschema 4.23.0 and requests 2.34.2.
Organizer scoring and contract validation are imported unchanged.
The build tests the actual linux/amd64 image with no network, read-only root,
temporary writable output and a non-root user before publication.
