# Executive summary: package the verified V11 image for Team 609 using the official hidden-key prompt.
# Default is Development. Pass -Phase final only for the one Final upload.
param(
    [ValidateSet('dev', 'final')][string]$Phase = 'dev',
    [int]$TeamNumber = 609
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$Body = @'
{
  "schema_version": "1.1.0",
  "interface_version": "2.0",
  "competition_id": "agenthon2026-forecasting-dev",
  "track": "forecasting",
  "phase": "dev",
  "category": "api",
  "image": {
    "registry": "ghcr.io",
    "repository": "ellatso/agenthon-t2",
    "digest": "sha256:c7705573c662e911d5b81f7b5a00a74f6e0570edce71b329d1ad3b0f3765d9a9"
  },
  "image_access": "public",
  "models": [],
  "license": "MIT"
}
'@ | ConvertFrom-Json
if ($Body.image.digest -notmatch '^sha256:[0-9a-f]{64}$') {
    throw 'Image verification is not complete. Use the published final packer.'
}
$Body.phase = $Phase
$Body.competition_id = "agenthon2026-forecasting-$Phase"
$Root = Join-Path (Join-Path $env:USERPROFILE 'Downloads') "Agenthon-T2-$TeamNumber-V11"
$OutDir = Join-Path $Root $Phase
$Venv = Join-Path $Root '.venv'
$Python = Join-Path $Venv 'Scripts/python.exe'
$Qfbench = Join-Path $Venv 'Scripts/qfbench2.exe'
$Descriptor = Join-Path $OutDir 'submission.json'
$Zip = Join-Path $OutDir 'submission.zip'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Python Launcher (py) was not found. Install Python 3.13 for Windows.'
}
& py -3.13 --version
if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 is required.' }
if (-not (Test-Path -LiteralPath $Python)) {
    & py -3.13 -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
& $Python -m pip install --disable-pip-version-check `
    'qfbench2-common @ https://github.com/Agenthon-2026/Agenthon2026-public/archive/refs/tags/v2.6.0.tar.gz#subdirectory=common'
if ($LASTEXITCODE -ne 0) { throw 'Could not install the pinned official packer.' }
[System.IO.File]::WriteAllText($Descriptor, ($Body | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
Write-Host "Packing Team $TeamNumber, phase $Phase. Enter your Team Key at the hidden prompt."
& $Qfbench submission pack --descriptor $Descriptor --team-number $TeamNumber --out $Zip --force
if ($LASTEXITCODE -ne 0) { throw 'Packing failed. No successful ZIP is claimed.' }
Write-Host "Ready to upload: $Zip"
Write-Host "Phase: $Phase. Final allows one upload. Keep the Team Key private."
