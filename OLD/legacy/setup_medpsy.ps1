# One-time setup: Ollama + qvac/MedPsy-4B GGUF + embedding model (Windows).
$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectDir

$ModelsDir = Join-Path $ProjectDir "models"
$OllamaModels = Join-Path $ProjectDir ".ollama-models"
$Quant = if ($env:MEDPSY_QUANT) { $env:MEDPSY_QUANT } else { "medpsy-4b-q4_k_m-imat.gguf" }
$ModelTag = if ($env:QVAC_OLLAMA_MODEL) { $env:QVAC_OLLAMA_MODEL } else { "medpsy-4b-cpu" }
$EmbedTag = if ($env:QVAC_EMBED_MODEL) { $env:QVAC_EMBED_MODEL } else { "all-minilm-cpu" }
$OllamaHost = if ($env:QVAC_OLLAMA_HOST_BIND) { $env:QVAC_OLLAMA_HOST_BIND } else { "127.0.0.1:11434" }

function Get-OllamaExe {
    $local = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $local) { return $local }
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

Write-Host "==> Checking for Ollama..."
$OllamaExe = Get-OllamaExe
if (-not $OllamaExe) {
    Write-Host "==> Ollama not found. Installing via winget (or download from https://ollama.com/download)..."
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
        Start-Sleep -Seconds 5
        $OllamaExe = Get-OllamaExe
    }
}
if (-not $OllamaExe) {
    Write-Host "ERROR: Install Ollama manually, then re-run install.ps1"
    exit 1
}
& $OllamaExe --version

Write-Host "==> Python venv + huggingface_hub..."
if (-not (Test-Path ".venv")) { python -m venv .venv }
& ".\.venv\Scripts\Activate.ps1"
pip install -q -r requirements.txt
pip install -q huggingface_hub

Write-Host "==> Downloading qvac/MedPsy-4B-GGUF ($Quant)..."
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
python -c @"
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='qvac/MedPsy-4B-GGUF', filename='$Quant', local_dir=r'$ModelsDir')
print('Downloaded:', path)
"@

@"
FROM ./$Quant

PARAMETER temperature 0.6
PARAMETER top_k 20
PARAMETER top_p 0.95
PARAMETER num_predict 2400
PARAMETER num_ctx 4096
PARAMETER num_gpu 0
"@ | Set-Content -Path (Join-Path $ModelsDir "Modelfile") -Encoding UTF8

$env:OLLAMA_HOST = $OllamaHost
$env:OLLAMA_MODELS = $OllamaModels
New-Item -ItemType Directory -Force -Path $OllamaModels | Out-Null

Write-Host "==> Starting Ollama server (if needed)..."
try {
    Invoke-WebRequest -Uri "http://$OllamaHost/api/version" -UseBasicParsing -TimeoutSec 2 | Out-Null
} catch {
    Start-Process -FilePath $OllamaExe -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        try {
            Invoke-WebRequest -Uri "http://$OllamaHost/api/version" -UseBasicParsing -TimeoutSec 2 | Out-Null
            break
        } catch { }
    }
}

Write-Host "==> Creating Ollama model '$ModelTag'..."
Push-Location $ModelsDir
& $OllamaExe create $ModelTag -f Modelfile
Pop-Location

Write-Host "==> Embedding model for semantic KPI..."
& $OllamaExe pull all-minilm
@"
FROM all-minilm
PARAMETER num_gpu 0
"@ | Set-Content -Path "$env:TEMP\qvac-embed-modelfile" -Encoding UTF8
& $OllamaExe create $EmbedTag -f "$env:TEMP\qvac-embed-modelfile"
Remove-Item "$env:TEMP\qvac-embed-modelfile" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Setup complete. Start with launch_dashboard.bat"
