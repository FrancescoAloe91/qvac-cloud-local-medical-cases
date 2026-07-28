# Download MedPsy 4B GGUF from Hugging Face into models/ (~2.5 GB).
# Pins revision + verifies sha256 when MEDPSY_GGUF_SHA256 is set.
# Default: no baked digest (empty Expected) — set MEDPSY_GGUF_SHA256 in env for bit-pin.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Quant = if ($env:MEDPSY_QUANT) { $env:MEDPSY_QUANT } else { "medpsy-4b-q4_k_m-imat.gguf" }
$Repo = if ($env:MEDPSY_REPO) { $env:MEDPSY_REPO } else { "qvac/MedPsy-4B-GGUF" }
$Revision = if ($env:MEDPSY_REVISION) { $env:MEDPSY_REVISION } else { "main" }
$Expected = if ($env:MEDPSY_GGUF_SHA256) { $env:MEDPSY_GGUF_SHA256 } else { "" }
$ModelsDir = Join-Path $Root "models"
$Target = Join-Path $ModelsDir $Quant
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

function Get-Sha256([string]$Path) {
  (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant()
}

if (Test-Path $Target) {
  Write-Host "==> MedPsy GGUF already present: $Target"
  $got = Get-Sha256 $Target
  if ($Expected) {
    if ($got -ne $Expected.ToLowerInvariant()) {
      throw "sha256 mismatch: expected $Expected got $got"
    }
    Write-Host "==> sha256 OK ($got)"
  } else {
    Write-Host "==> sha256: $got (set MEDPSY_GGUF_SHA256 to enforce)"
  }
  exit 0
}

if (-not (Test-Path ".venv")) {
  python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
pip install -q huggingface_hub

Write-Host "==> Downloading $Repo@$Revision · $Quant (~2.5 GB)…"
python -c @"
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='$Repo', filename='$Quant', local_dir=r'$ModelsDir', revision='$Revision')
print('Downloaded:', path)
"@

if (-not (Test-Path $Target)) {
  throw "Download finished but $Target missing"
}
$got = Get-Sha256 $Target
Write-Host "==> Ready: $Target"
Write-Host "    sha256: $got"
Write-Host "    revision: $Revision"
if ($Expected) {
  if ($got -ne $Expected.ToLowerInvariant()) {
    throw "sha256 mismatch after download: expected $Expected got $got"
  }
  Write-Host "==> sha256 OK"
}
