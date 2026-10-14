# Executive summary: run private comparison stages; never package or upload a Final submission.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('plan', 'smoke', 'seal', 'score')]
    [string]$Action,
    [string]$Catalog,
    [string]$RunDir,
    [ValidateSet('select', 'confirm', 'test')]
    [string]$Phase = 'select',
    [string]$Python = 'python',
    [string]$ResearchRoot = (Join-Path $env:USERPROFILE 'Downloads\Agenthon-T2-Final-Research')
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$comparisonRepo = Split-Path -Parent $PSScriptRoot
if ($Action -eq 'plan' -and (-not $Catalog -or -not (Test-Path -LiteralPath $Catalog))) {
    throw 'plan requires -Catalog with an existing private catalog JSON file.'
}
if ($Catalog) { $Catalog = [System.IO.Path]::GetFullPath($Catalog) }
if ($RunDir) { $RunDir = [System.IO.Path]::GetFullPath($RunDir) }
$ResearchRoot = [System.IO.Path]::GetFullPath($ResearchRoot)
$venvPython = Join-Path $ResearchRoot '.venv\Scripts\python.exe'
$readyFile = Join-Path $ResearchRoot 'environment-ready.txt'
$latestFile = Join-Path $ResearchRoot 'latest-comparison.txt'
New-Item -ItemType Directory -Force -Path $ResearchRoot | Out-Null
if (-not (Test-Path -LiteralPath $readyFile)) {
    & $Python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,13) else 1)'
    if ($LASTEXITCODE -ne 0) { throw 'Select a Python 3.13 interpreter with -Python.' }
    & $Python -m venv (Join-Path $ResearchRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Research environment creation failed.' }
    & $venvPython -m pip install numpy==2.1.3 scipy==1.15.3 pandas==2.2.3 pyarrow==18.1.0 jsonschema==4.23.0 requests==2.34.2 'qfbench2-common @ https://github.com/Agenthon-2026/Agenthon2026-public/archive/refs/tags/v2.6.0.tar.gz#subdirectory=common'
    if ($LASTEXITCODE -ne 0) { throw 'Pinned dependency installation failed.' }
    & $venvPython -m pip install --no-deps -e $comparisonRepo
    if ($LASTEXITCODE -ne 0) { throw 'Research package installation failed.' }
    'Pinned common 2.6.0 comparison environment' | Set-Content -LiteralPath $readyFile -Encoding UTF8
}
if (-not $RunDir) {
    if ($Action -eq 'plan') {
        $RunDir = Join-Path $ResearchRoot ('run-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
    } elseif (Test-Path -LiteralPath $latestFile) {
        $RunDir = (Get-Content -LiteralPath $latestFile -Raw -Encoding UTF8).Trim()
    } else {
        throw 'Pass -RunDir or first create a plan.'
    }
}
$comparisonArgs = @('-m', 'experiments.compare_final', $Action, '--run-dir', $RunDir)
if ($Action -eq 'plan') { $comparisonArgs += @('--catalog', $Catalog) }
if ($Action -eq 'score') { $comparisonArgs += @('--phase', $Phase) }
$previousLocation = Get-Location
try {
    Set-Location -LiteralPath $comparisonRepo
    & $venvPython @comparisonArgs
    if ($LASTEXITCODE -ne 0) { throw "Comparison stage '$Action' failed." }
    if ($Action -eq 'plan') {
        $RunDir | Set-Content -LiteralPath $latestFile -Encoding UTF8
    }
} finally {
    Set-Location -LiteralPath $previousLocation
}
Write-Host "Comparison stage complete: $RunDir"
if ($Action -eq 'smoke') {
    Write-Host 'House access was disabled. These outputs establish execution only, not text quality.'
}
