#!/usr/bin/env bash
# One-shot LOCAL install: Python + MedPsy GGUF (~2.5 GB) + QVAC SDK sidecar.
# Does NOT commit the GGUF to git — downloads it from Hugging Face.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  QVAC vs Cloud · Health Test — local install (one time)      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ Python 3.10+ required (python.org or: brew install python3)"
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
pip install -q -r requirements.txt
pip install -q huggingface_hub

if [[ ! -f .env ]]; then
  cp -n .env.example .env 2>/dev/null || cp .env.example .env
  echo "    Created .env — paste your full OpenRouter key (sk-or-v1-…)."
fi

echo "==> 2/4 MedPsy GGUF from Hugging Face (~2.5 GB, skipped if present)…"
chmod +x scripts/download_medpsy_gguf.sh scripts/setup_qvac_sidecar.sh
./scripts/download_medpsy_gguf.sh

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
echo "   History is private to that key (Cases A/B/C) under artifacts/owners/…"
echo "   Cloud demo (BYOK, no QVAC):"
echo "     https://francescoaloe91-qvac-vs-cloud-llms-health-test-app-wihxyd.streamlit.app"
echo ""
