# Executive summary: sign the verified Final descriptor locally for Team 609.
# Enter the Team Key only at the official hidden prompt. This script never uploads.
param(
    [string]$DescriptorPath = (Join-Path $PSScriptRoot 'submission-final.json'),
    [int]$TeamNumber = 609
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
if (-not (Test-Path -LiteralPath $DescriptorPath -PathType Leaf)) {
    throw 'Put the verified submission-final.json beside this script.'
}
$Body = Get-Content -LiteralPath $DescriptorPath -Raw | ConvertFrom-Json
if ($Body.phase -ne 'final' -or $Body.track -ne 'forecasting' -or
    $Body.competition_id -ne 'agenthon2026-forecasting-final' -or
    $Body.image.registry -ne 'ghcr.io' -or
    $Body.image.repository -ne 'ellatso/agenthon-t2' -or
    $Body.image.digest -notmatch '^sha256:[0-9a-f]{64}$') {
    throw 'The descriptor is not the verified forecasting Final descriptor.'
}
$Root = Join-Path (Join-Path $env:USERPROFILE 'Downloads') "Agenthon-T2-$TeamNumber-Final"
$OutDir = Join-Path $Root (Get-Date -Format 'yyyyMMdd-HHmmss')
$Venv = Join-Path $Root '.venv'
$Python = Join-Path $Venv 'Scripts/python.exe'
$Qfbench = Join-Path $Venv 'Scripts/qfbench2.exe'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Install Python 3.13 for Windows with the Python Launcher.'
}
& py -3.13 --version
if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 is required.' }
if (-not (Test-Path -LiteralPath $Python)) {
    & py -3.13 -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
& $Python -m pip install --disable-pip-version-check `
    'qfbench2-common @ https://github.com/Agenthon-2026/Agenthon2026-public/archive/refs/tags/v2.6.1.tar.gz#subdirectory=common'
if ($LASTEXITCODE -ne 0) { throw 'Could not install the pinned official packer.' }
$Descriptor = Join-Path $OutDir 'submission.json'
$Zip = Join-Path $OutDir 'submission.zip'
Copy-Item -LiteralPath $DescriptorPath -Destination $Descriptor
Write-Host "Packing Team $TeamNumber, FINAL. Enter the Team Key at the hidden prompt."
& $Qfbench submission pack --descriptor $Descriptor --team-number $TeamNumber --out $Zip
if ($LASTEXITCODE -ne 0) { throw 'Packing failed. No successful ZIP is claimed.' }
Write-Host "Ready for manual Final upload: $Zip"
Write-Host 'The later Final upload replaces the earlier one. Keep the Team Key private.'
