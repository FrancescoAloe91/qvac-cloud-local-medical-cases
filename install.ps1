# Thin PowerShell wrapper — real Windows install lives in install.bat
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
$bat = Join-Path $Root "install.bat"
if (-not (Test-Path $bat)) {
  throw "install.bat missing next to install.ps1"
}
cmd /c "`"$bat`""
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
