# Executive summary (read this first): set up Python 3.13 and run paired variance and correlation experiments.
# Results and the environment stay under Downloads, outside the public repository.
param(
    [ValidateSet('benchmark','notebook','all')][string]$Phase = 'all',
    [string]$RunDir = ''
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
$Research = Join-Path (Join-Path $env:USERPROFILE 'Downloads') 'Agenthon-T2-Research-V5'
$Venv = Join-Path $Research '.venv'
$Python = Join-Path $Venv 'Scripts/python.exe'
$Latest = Join-Path $Research 'latest-run.txt'
New-Item -ItemType Directory -Force -Path $Research | Out-Null
if (-not (Test-Path -LiteralPath $Python)) {
    & py -3.13 -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 is required.' }
}
& $Python -m pip install --disable-pip-version-check `
    'numpy==2.1.3' 'pandas==2.2.3' 'pyarrow==18.1.0' 'jsonschema==4.23.0' 'requests==2.34.2' `
    'qfbench2-common @ https://github.com/Agenthon-2026/Agenthon2026-public/archive/refs/tags/v2.4.4.tar.gz#subdirectory=common'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if (-not $RunDir) {
    if ($Phase -eq 'benchmark' -or $Phase -eq 'all') {
        $RunDir = Join-Path $Research ('run-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
    } elseif (Test-Path -LiteralPath $Latest) {
        $RunDir = (Get-Content -LiteralPath $Latest -Raw).Trim()
    } else { throw 'No prior run. Run the benchmark phase first.' }
}
Push-Location -LiteralPath $Repo
try {
    $Phases = if ($Phase -eq 'all') { @('benchmark','notebook') } else { @($Phase) }
    foreach ($CurrentPhase in $Phases) {
        if ($CurrentPhase -eq 'notebook') {
            & $Python -m pip install notebook matplotlib
            if ($LASTEXITCODE -ne 0) { throw 'Notebook installation failed.' }
            $Notebook = Join-Path $RunDir 'ablation-results.ipynb'
            Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'ablation-results.ipynb') -Destination $Notebook -Force
            & $Python -X utf8 -m notebook $Notebook
            if ($LASTEXITCODE -ne 0) { throw 'Notebook launch failed.' }
        } else {
            & $Python -X utf8 -m experiments.ablation --run-dir $RunDir
            if ($LASTEXITCODE -ne 0) { throw 'Experiment failed. Read the terminal and coverage reports.' }
            if ($CurrentPhase -eq 'benchmark') { Set-Content -LiteralPath $Latest -Value $RunDir -Encoding UTF8 }
            Write-Host "Results: $RunDir"
            Write-Host 'These are historical diagnostics; do not commit outputs or claim independent validation.'
        }
    }
} finally { Pop-Location }
