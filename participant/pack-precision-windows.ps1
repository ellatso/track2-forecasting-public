# Executive summary (read this first): package the tested numeric precision and monthly-trend T2 image for Team 609.
# Run this script in PowerShell from a checkout of the submission/t2-precision-monthly50 branch.
# The Team Key is read only by qfbench2's hidden terminal prompt, never by this script.
param([int]$TeamNumber = 609)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$Template = Join-Path $PSScriptRoot 'submission-dev.template.json'
$OutDir = Join-Path (Join-Path $env:USERPROFILE 'Downloads') "Agenthon-T2-$TeamNumber-precision-monthly50"
$Venv = Join-Path $OutDir '.venv'
$Python = Join-Path $Venv 'Scripts/python.exe'
$Qfbench = Join-Path $Venv 'Scripts/qfbench2.exe'
$Descriptor = Join-Path $OutDir 'submission.json'
$Zip = Join-Path $OutDir 'submission.zip'

if (-not (Test-Path -LiteralPath $Template)) {
    throw "Template not found: $Template. Check out the submission/t2-precision-monthly50 branch."
}
$ModelDescriptor = Get-Content -LiteralPath $Template -Raw -Encoding UTF8 | ConvertFrom-Json
if ($ModelDescriptor.image.repository -ne 'ellatso/agenthon-t2' -or
    @($ModelDescriptor.models).Count -ne 0 -or
    $ModelDescriptor.category -ne 'api') {
    throw 'Wrong descriptor: this trial requires the pinned numeric-only candidate.'
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Python Launcher (py) was not found. Install Python 3.13 for Windows, then rerun this script.'
}
& py -3.13 --version
if ($LASTEXITCODE -ne 0) {
    throw 'Python 3.13 is required. Install it, then rerun this script.'
}
if (-not (Test-Path -LiteralPath $Python)) {
    & py -3.13 -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python 3.13 environment.' }
}

& $Python -m pip install --disable-pip-version-check `
    'qfbench2-common @ https://github.com/Agenthon-2026/Agenthon2026-public/archive/refs/tags/v2.4.4.tar.gz#subdirectory=common'
if ($LASTEXITCODE -ne 0) { throw 'Could not install the pinned Agenthon toolkit.' }

Copy-Item -LiteralPath $Template -Destination $Descriptor -Force
Write-Host "Packing Team $TeamNumber. Enter the Team Key at the hidden prompt."
& $Qfbench submission pack --descriptor $Descriptor --team-number $TeamNumber --out $Zip --force
if ($LASTEXITCODE -ne 0) { throw 'Packing failed. Check the Team Number and Team Key.' }

Write-Host "Ready to upload on CodaBench: $Zip"
Write-Host 'Keep the Team Key and submission.zip out of the public GitHub repository.'
