#!/usr/bin/env bash
# Download 3 open Q4 GGUFs into models/ for on-device Band B (real local privacy).
# Same QVAC sidecar hot-swaps them via POST /load — not OpenRouter, not Ollama.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
MODELS_DIR="$PROJECT_DIR/models"
mkdir -p "$MODELS_DIR"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q huggingface_hub

# repo|filename|local_name
PEERS=(
  "bartowski/gemma-2-2b-it-GGUF|gemma-2-2b-it-Q4_K_M.gguf|gemma-2-2b-it-Q4_K_M.gguf"
  "bartowski/Llama-3.2-3B-Instruct-GGUF|Llama-3.2-3B-Instruct-Q4_K_M.gguf|Llama-3.2-3B-Instruct-Q4_K_M.gguf"
  "bartowski/Phi-3.5-mini-instruct-GGUF|Phi-3.5-mini-instruct-Q4_K_M.gguf|Phi-3.5-mini-instruct-Q4_K_M.gguf"
)

download_one() {
  local repo="$1" quant="$2" local_name="$3"
  local target="$MODELS_DIR/$local_name"
  if [[ -f "$target" && -s "$target" ]]; then
    echo "==> already present: $local_name ($(du -h "$target" | awk '{print $1}'))"
    return 0
  fi
  echo "==> Downloading $repo · $quant …"
  python3 - "$repo" "$quant" "$MODELS_DIR" "$local_name" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

repo, quant, local_dir, local_name = sys.argv[1:5]
path = Path(
    hf_hub_download(repo_id=repo, filename=quant, local_dir=local_dir)
)
target = Path(local_dir) / local_name
if path.resolve() != target.resolve():
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(path.resolve())
    except OSError:
        import shutil
        shutil.copy2(path, target)
print("Downloaded:", target)
PY
}

for row in "${PEERS[@]}"; do
  IFS='|' read -r repo quant local_name <<<"$row"
  download_one "$repo" "$quant" "$local_name"
done

echo "==> Band B local peers ready under $MODELS_DIR"
ls -lh "$MODELS_DIR"/*{gemma-2-2b,Llama-3.2,Phi-3.5}* 2>/dev/null || true
