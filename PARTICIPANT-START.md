# Executive summary
The integrated candidate combines the evaluated recent-window numerical method (-1.0042, submission #953358) with evidence-conditioned House-model scenarios. It is a new, unscored candidate; passing admissibility gates does not establish improved forecasting accuracy.

## Forecasting behavior
The agent first extracts verbatim evidence from indexed documents published by the card cutoff, then requests weighted scenarios linked to validated evidence. FX quote conventions, target units, monthly publication lag and history gaps are explicit. Unsupported adjustments are discarded; bounded shifts, conditional volatility and fat-tailed common shocks preserve coherent trajectories. Half of draws retain the numerical baseline. Invalid or unavailable House responses fall back to that same numerical method.

House requests use only the injected authenticated proxy and official endpoint. There are at most three attempts per card, temperature zero, thinking disabled and at most 3600 output tokens per request. No endpoint credentials, Team Key or hidden reasoning trace enter output files. The submission descriptor declares the approved House model.

## Validation and limits
44 focused tests pass, including an actual HTTP exchange with a local mock proxy, malformed responses, fabricated citations, monthly steps and fallback equality. All 104 published cards pass official g0-g3 checks with House disabled; no realized outcome is read. GitHub Actions validates Python 3.13 and the read-only container before publishing:
https://github.com/ellatso/track2-forecasting-public/actions/runs/36823807050

The actual House service and its judgement quality cannot be tested without organizer-injected credentials. Earlier public-history calibration was exploratory, not an untouched final holdout. This candidate does not promise to beat -0.6301.

## Package on Windows
Use your existing checkout:
```powershell
$repo = "C:\Users\ella.tso\Downloads\agenthon-t2"
git -C "$repo" fetch origin
git -C "$repo" switch submission/t2-integrated
git -C "$repo" pull --ff-only origin submission/t2-integrated
powershell -ExecutionPolicy Bypass -File "$repo\participant\pack-windows.ps1"
```
Enter Team 609's Team Key only at the hidden terminal prompt. Upload:
`C:\Users\ella.tso\Downloads\Agenthon-T2-609-integrated\submission.zip`

The template image digest is pinned after the container passes CI. Never upload a ZIP generated before that pin is committed. Keep the Team Key and packed ZIP outside the public repository.
