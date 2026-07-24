# Download MedPsy 4B GGUF from Hugging Face into models/ (~2.5 GB).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Quant = if ($env:MEDPSY_QUANT) { $env:MEDPSY_QUANT } else { "medpsy-4b-q4_k_m-imat.gguf" }
$ModelsDir = Join-Path $Root "models"
$Target = Join-Path $ModelsDir $Quant
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

if (Test-Path $Target) {
  Write-Host "==> MedPsy GGUF already present: $Target"
  exit 0
}

if (-not (Test-Path ".venv")) {
  python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
pip install -q huggingface_hub

Write-Host "==> Downloading qvac/MedPsy-4B-GGUF · $Quant (~2.5 GB)…"
python -c @"
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='qvac/MedPsy-4B-GGUF', filename='$Quant', local_dir=r'$ModelsDir')
print('Downloaded:', path)
"@

Write-Host "==> Ready under models\"
