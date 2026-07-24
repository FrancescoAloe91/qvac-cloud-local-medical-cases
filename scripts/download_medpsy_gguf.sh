#!/usr/bin/env bash
# Download MedPsy 4B GGUF from Hugging Face into models/ (~2.5 GB).
# The weights are NOT in git (GitHub file-size limit); this is the local install path.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

QUANT="${MEDPSY_QUANT:-medpsy-4b-q4_k_m-imat.gguf}"
MODELS_DIR="$PROJECT_DIR/models"
TARGET="$MODELS_DIR/$QUANT"

mkdir -p "$MODELS_DIR"

if [[ -f "$TARGET" && -s "$TARGET" ]]; then
  SIZE="$(du -h "$TARGET" | awk '{print $1}')"
  echo "==> MedPsy GGUF already present: $TARGET ($SIZE)"
  exit 0
fi

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q huggingface_hub

echo "==> Downloading qvac/MedPsy-4B-GGUF · $QUANT (~2.5 GB)…"
python3 - "$QUANT" "$MODELS_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download

quant, local_dir = sys.argv[1], sys.argv[2]
path = hf_hub_download(
    repo_id="qvac/MedPsy-4B-GGUF",
    filename=quant,
    local_dir=local_dir,
)
print("Downloaded:", path)
PY

if [[ ! -f "$TARGET" ]]; then
  # hf_hub_download may nest under models/<repo>/ — normalize to models/$QUANT
  FOUND="$(find "$MODELS_DIR" -name "$QUANT" -type f | head -n 1 || true)"
  if [[ -n "$FOUND" && "$FOUND" != "$TARGET" ]]; then
    ln -sfn "$FOUND" "$TARGET" 2>/dev/null || cp -f "$FOUND" "$TARGET"
  fi
fi

echo "==> Ready: $TARGET"
echo "    Sidecar will pick this up via QVAC_MODEL_PATH or default path."
