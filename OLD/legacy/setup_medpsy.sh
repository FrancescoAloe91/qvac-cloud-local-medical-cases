#!/bin/bash
# One-time setup: install Ollama locally (no admin/Homebrew needed), download the
# REAL qvac/MedPsy-4B GGUF weights from Hugging Face, and register the model in
# Ollama so the dashboard can run genuine on-device inference.
#
# This project never fabricates QVAC's diagnostic output - if this setup hasn't
# been run, the dashboard will show a clear error instead of a fake answer.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

OLLAMA_DIR="$PROJECT_DIR/.ollama-bin"
OLLAMA_BIN="$OLLAMA_DIR/Ollama.app/Contents/Resources/ollama"
MODELS_DIR="$PROJECT_DIR/models"
QUANT="${MEDPSY_QUANT:-medpsy-4b-q4_k_m-imat.gguf}"
MODEL_TAG="${QVAC_OLLAMA_MODEL:-medpsy-4b-cpu}"

echo "==> Checking for Ollama..."
if [ ! -x "$OLLAMA_BIN" ]; then
  echo "==> Downloading Ollama (darwin)..."
  mkdir -p "$OLLAMA_DIR"
  curl -fsSL -o /tmp/ollama_setup.zip "https://ollama.com/download/Ollama-darwin.zip"
  unzip -o -q /tmp/ollama_setup.zip -d "$OLLAMA_DIR"
  rm -f /tmp/ollama_setup.zip
fi
"$OLLAMA_BIN" --version || true

echo "==> Ensuring Python venv + huggingface_hub..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt
pip install -q huggingface_hub

echo "==> Downloading qvac/MedPsy-4B-GGUF ($QUANT) from Hugging Face..."
mkdir -p "$MODELS_DIR"
python3 - "$QUANT" "$MODELS_DIR" <<'PYEOF'
import sys
from huggingface_hub import hf_hub_download
quant, local_dir = sys.argv[1], sys.argv[2]
path = hf_hub_download(repo_id="qvac/MedPsy-4B-GGUF", filename=quant, local_dir=local_dir)
print("Downloaded:", path)
PYEOF

cat > "$MODELS_DIR/Modelfile" <<EOF
FROM ./$QUANT

PARAMETER temperature 0.6
PARAMETER top_k 20
PARAMETER top_p 0.95
PARAMETER num_predict 2400
PARAMETER num_ctx 4096
PARAMETER num_gpu 0
EOF
# num_gpu 0 works around a known ggml-Metal "Gated Delta Net" assertion crash
# with this GGUF on some Apple Silicon + Ollama builds. If your machine runs
# the model fine on GPU, remove that line for faster generation.

echo "==> Starting Ollama server (background)..."
export OLLAMA_HOST="${QVAC_OLLAMA_HOST_BIND:-127.0.0.1:11434}"
export OLLAMA_MODELS="$PROJECT_DIR/.ollama-models"
mkdir -p "$OLLAMA_MODELS"
if ! curl -sf -o /dev/null "http://${OLLAMA_HOST}/api/version" 2>/dev/null; then
  nohup "$OLLAMA_BIN" serve > /tmp/qvac-ollama-serve.log 2>&1 &
  disown
  for _ in $(seq 1 30); do
    curl -sf -o /dev/null "http://${OLLAMA_HOST}/api/version" 2>/dev/null && break
    sleep 1
  done
fi

echo "==> Creating Ollama model '$MODEL_TAG'..."
(cd "$MODELS_DIR" && OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" create "$MODEL_TAG" -f Modelfile)

echo "==> Setting up the local embedding model (semantic-similarity KPI)..."
EMBED_TAG="${QVAC_EMBED_MODEL:-all-minilm-cpu}"
OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" pull all-minilm
cat > /tmp/qvac-embed-modelfile <<EOF
FROM all-minilm
PARAMETER num_gpu 0
EOF
# Same GPU workaround as MedPsy above: forces CPU-only inference for the
# tiny embedding model too, since the Metal backend crash isn't specific
# to the MedPsy architecture.
OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" create "$EMBED_TAG" -f /tmp/qvac-embed-modelfile
rm -f /tmp/qvac-embed-modelfile

echo ""
echo "Setup complete. Real qvac/MedPsy-4B is ready as Ollama model '$MODEL_TAG',"
echo "and the '$EMBED_TAG' embedding model is ready for the semantic-similarity KPI."
echo "Start the dashboard with ./launch_dashboard.sh"
