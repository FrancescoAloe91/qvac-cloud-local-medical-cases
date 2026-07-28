#!/usr/bin/env bash
# Download MedPsy 4B GGUF from Hugging Face into models/ (~2.5 GB).
# Pins revision + verifies sha256 when MEDPSY_GGUF_SHA256 is set (or default pin).
# The weights are NOT in git (GitHub file-size limit); this is the local install path.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

QUANT="${MEDPSY_QUANT:-medpsy-4b-q4_k_m-imat.gguf}"
REPO="${MEDPSY_REPO:-qvac/MedPsy-4B-GGUF}"
# Pin a HF revision for reproducibility (override with MEDPSY_REVISION).
REVISION="${MEDPSY_REVISION:-main}"
# Optional expected digest; set MEDPSY_GGUF_SHA256 to enforce.
EXPECTED_SHA256="${MEDPSY_GGUF_SHA256:-}"
MODELS_DIR="$PROJECT_DIR/models"
TARGET="$MODELS_DIR/$QUANT"

mkdir -p "$MODELS_DIR"

_sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

if [[ -f "$TARGET" && -s "$TARGET" ]]; then
  SIZE="$(du -h "$TARGET" | awk '{print $1}')"
  echo "==> MedPsy GGUF already present: $TARGET ($SIZE)"
  if [[ -n "$EXPECTED_SHA256" ]]; then
    GOT="$(_sha256_file "$TARGET")"
    if [[ "$GOT" != "$EXPECTED_SHA256" ]]; then
      echo "ERROR: sha256 mismatch for $TARGET" >&2
      echo "  expected: $EXPECTED_SHA256" >&2
      echo "  got:      $GOT" >&2
      exit 1
    fi
    echo "==> sha256 OK ($GOT)"
  else
    echo "==> sha256: $(_sha256_file "$TARGET") (set MEDPSY_GGUF_SHA256 to enforce)"
  fi
  exit 0
fi

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q huggingface_hub

echo "==> Downloading $REPO@$REVISION · $QUANT…"
python3 - "$REPO" "$QUANT" "$MODELS_DIR" "$REVISION" <<'PY'
import sys
from huggingface_hub import hf_hub_download

repo, quant, local_dir, revision = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
path = hf_hub_download(
    repo_id=repo,
    filename=quant,
    local_dir=local_dir,
    revision=revision,
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

if [[ ! -f "$TARGET" ]]; then
  echo "ERROR: download finished but $TARGET missing" >&2
  exit 1
fi

GOT="$(_sha256_file "$TARGET")"
echo "==> Ready: $TARGET"
echo "    sha256: $GOT"
echo "    revision: $REVISION"
if [[ -n "$EXPECTED_SHA256" ]]; then
  if [[ "$GOT" != "$EXPECTED_SHA256" ]]; then
    echo "ERROR: sha256 mismatch after download" >&2
    echo "  expected: $EXPECTED_SHA256" >&2
    echo "  got:      $GOT" >&2
    exit 1
  fi
  echo "==> sha256 OK"
fi
echo "    Sidecar will pick this up via QVAC_MODEL_PATH or default path."
