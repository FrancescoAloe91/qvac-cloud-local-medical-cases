#!/usr/bin/env bash
# One-shot LOCAL install: Python + GGUFs from Hugging Face + QVAC SDK sidecar.
# GGUFs are NOT in git. Private History under artifacts/ is NEVER part of install.
#
# Default: MedPsy-4B Q4 only (~2.5 GB).
# Full 9-model on-device pack (~14 GB): FULL_MODELS=1 ./install.sh
#                              or:     ./install.sh --full-models
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

FULL_MODELS="${FULL_MODELS:-0}"
for arg in "$@"; do
  case "$arg" in
    --full-models|--full) FULL_MODELS=1 ;;
  esac
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  QVAC vs Cloud · Health Test — local install (one time)      ║"
if [[ "$FULL_MODELS" == "1" ]]; then
echo "║  Mode: FULL GGUF pack (~14 GB · 3 MedPsy + 3 peers)          ║"
else
echo "║  Mode: MedPsy-4B Q4 only (~2.5 GB) · add --full-models later ║"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ Python 3.9+ required (hashed locks/CI use 3.9; 3.10+ preferred locally — python.org or: brew install python3)"
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "❌ Node.js >= 22.17 required (https://nodejs.org/)"
  exit 1
fi

echo "==> 1/4 Python virtualenv + deps…"
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q --require-hashes -r requirements.txt

if [[ ! -f .env ]]; then
  cp -n .env.example .env 2>/dev/null || cp .env.example .env
  echo "    Created .env — paste your full OpenRouter key (sk-or-v1-…)."
fi

chmod +x scripts/download_medpsy_gguf.sh scripts/download_local_peers.sh \
  scripts/download_all_ggufs.sh scripts/setup_qvac_sidecar.sh 2>/dev/null || true

if [[ "$FULL_MODELS" == "1" ]]; then
  echo "==> 2/4 Full GGUF pack from Hugging Face (~14 GB, skip if present)…"
  ./scripts/download_all_ggufs.sh
else
  echo "==> 2/4 MedPsy-4B Q4 from Hugging Face (~2.5 GB, skipped if present)…"
  ./scripts/download_medpsy_gguf.sh
  echo "    Tip: ./install.sh --full-models  (or ./scripts/download_all_ggufs.sh)"
  echo "         for all 3 MedPsy + Gemma/Llama/Phi peers."
fi

echo "==> 3/4 QVAC SDK sidecar (npm + OpenSSL 3 on macOS)…"
./scripts/setup_qvac_sidecar.sh

echo "==> 4/4 Launch helpers…"
chmod +x run.sh launch_dashboard.sh stop_dashboard.sh install.sh 2>/dev/null || true
if [[ -x scripts/build_dashboard_app.sh ]]; then
  ./scripts/build_dashboard_app.sh || true
fi

echo ""
echo "✅ Local install ready."
echo ""
echo "   Terminal A — QVAC MedPsy:"
echo "     cd sidecar && npm start"
echo ""
echo "   Terminal B — UI:"
echo "     source .venv/bin/activate && streamlit run app.py"
echo "     → open «Automated Benchmark»"
echo "     (or: ./launch_dashboard.sh)"
echo ""
echo "   Edit .env with OPENROUTER_API_KEY=sk-or-v1-… (full key)."
echo "   History stays ONLY on this machine under artifacts/owners/… (gitignored)."
echo "   Cloud demo (BYOK, no QVAC):"
echo "     https://francescoaloe91-qvac-vs-cloud-llms-health-test-app-wihxyd.streamlit.app"
echo ""
