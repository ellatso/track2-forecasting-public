# Numeric trial submission
Executive summary: submit one fixed numerical forecasting candidate, packaged locally with the official hidden Team Key prompt. Daily forecasts use half the sample drift and an equal mixture of sample variance and EWMA variance (60-observation half-life). Monthly forecasts retain the reference baseline. This is an experiment, not a promise of a leaderboard improvement.

## Fixed configuration
- Daily history: 300 observations; drift multiplier 0.5; variance mixture 0.5; correlation shrinkage 0; linear variance clock.
- Monthly: unchanged reference method and the card's explicit publication steps.
- Default seed: 0. Default draws: 1,000, increased to the card minimum when required.
- Pure numerical model: no House API calls, no learned model artifact; descriptor category `api`, `models: []`.
- The descriptor pins the tested linux/amd64 image by immutable digest.

## Windows packaging
Clone this branch into a separate folder so previous work and ZIPs remain available:

```powershell
git clone --branch submission/t2-halfdrift-mix50 --single-branch https://github.com/ellatso/track2-forecasting-public.git "C:\Users\ella.tso\Downloads\agenthon-t2-halfdrift"
powershell -ExecutionPolicy Bypass -File "C:\Users\ella.tso\Downloads\agenthon-t2-halfdrift\participant\pack-halfdrift-windows.ps1"
```

Enter the original Team Key only at the hidden terminal prompt. Team 609 is the script default. The output is:

```text
C:\Users\ella.tso\Downloads\Agenthon-T2-609-halfdrift-mix50\submission.zip
```

Upload that ZIP at https://www.codabench.org/competitions/17766/ from the existing competition account. Research-result ZIPs are not submissions. A claim from an older submission cannot authenticate a changed descriptor; the packer generates a fresh claim bound to this descriptor.

Keep the Team Key and submission ZIP out of the public repository. Record the new CodaBench submission ID, score and the descriptor image digest so the official result can be traced to this exact candidate. The official score is unknown until that run finishes.

## Verification
The image workflow runs regression tests and checks every published card inside the actual container with networking disabled and a read-only filesystem. A separate Windows workflow checks PowerShell syntax and official ZIP creation using ephemeral synthetic credentials. Neither workflow uses the real Team Key.
