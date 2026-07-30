#!/usr/bin/env bash
# Full on-device GGUF pack for the live grid (MedPsy + Band B + medical_local).
#
# Downloads from Hugging Face into models/ (skipped if already present).
# SHA pin is optional via env on child scripts (MEDPSY_GGUF_SHA256 / peers) —
# no baked digest by default (same pattern as download_medpsy_gguf.sh).
# Weights are NEVER stored in git — GitHub only ships this script + docs.
#
# Private run history (prompts, answers, judge scores, means) lives under
# artifacts/owners/… and is gitignored. This script does not touch it.
#
# Usage:
#   ./scripts/download_all_ggufs.sh
#   FULL_MODELS=1 ./install.sh
#   SKIP_MEDICAL=1 ./scripts/download_all_ggufs.sh   # MedPsy + Band B only
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

chmod +x scripts/download_medpsy_gguf.sh scripts/download_local_peers.sh \
  scripts/download_medical_peers.sh

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Full GGUF pack · MedPsy ×3 + Band B + medical peers (HF)    ║"
echo "║  ~25 GB · private History is NOT downloaded or overwritten   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

echo "==> 1/5 MedPsy-4B Q4 (default sidecar)…"
./scripts/download_medpsy_gguf.sh

echo "==> 2/5 MedPsy-4B Q8…"
MEDPSY_REPO=qvac/MedPsy-4B-GGUF MEDPSY_QUANT=medpsy-4b-q8_0.gguf \
  ./scripts/download_medpsy_gguf.sh

echo "==> 3/5 MedPsy-1.7B Q4…"
MEDPSY_REPO=qvac/MedPsy-1.7B-GGUF MEDPSY_QUANT=medpsy-1.7b-q4_k_m-imat.gguf \
  ./scripts/download_medpsy_gguf.sh

echo "==> 4/5 Band B peers (Gemma / Llama / Phi)…"
./scripts/download_local_peers.sh

if [[ "${SKIP_MEDICAL:-0}" == "1" ]]; then
  echo "==> 5/5 medical peers skipped (SKIP_MEDICAL=1)"
else
  echo "==> 5/5 medical peers (MedGemma / Med42 / OpenBioLLM)…"
  ./scripts/download_medical_peers.sh
fi

echo ""
echo "✅ Full GGUF pack ready under models/"
ls -lh models/*.gguf 2>/dev/null || true
echo ""
echo "Note: your prompts / outputs / rankings stay only in artifacts/ (local,"
echo "gitignored). Cloning this repo never includes another user's History."
echo ""
