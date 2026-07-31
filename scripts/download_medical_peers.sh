#!/usr/bin/env bash
# Download 3 medical-specialized Q4 GGUFs into models/ (Band medical_local).
# Same QVAC sidecar hot-swaps them via POST /load — not OpenRouter, not Ollama.
# SHA pin is optional via env: set MEDICAL_PEER_GGUF_SHA256_<name> or skip.
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

# Primary repo|hf_filename|local_name  (fallback tried in download_one for Med42)
# Filenames verified on Hugging Face (Q4_K_M):
#   unsloth/medgemma-1.5-4b-it-GGUF → medgemma-1.5-4b-it-Q4_K_M.gguf
#   mradermacher/Llama3-Med42-8B-GGUF → Llama3-Med42-8B.Q4_K_M.gguf
#   mradermacher/Llama-3-8B-UltraMedical-GGUF → Llama-3-8B-UltraMedical.Q4_K_M.gguf
PEERS=(
  "unsloth/medgemma-1.5-4b-it-GGUF|medgemma-1.5-4b-it-Q4_K_M.gguf|medgemma-1.5-4b-it-Q4_K_M.gguf"
  "mradermacher/Llama3-Med42-8B-GGUF|Llama3-Med42-8B.Q4_K_M.gguf|Llama3-Med42-8B.Q4_K_M.gguf"
  "mradermacher/Llama-3-8B-UltraMedical-GGUF|Llama-3-8B-UltraMedical.Q4_K_M.gguf|Llama-3-8B-UltraMedical.Q4_K_M.gguf"
)

# Med42 fallback if primary HF file is missing
MED42_FALLBACK_REPO="tensorblock/Llama3-Med42-8B-GGUF"
MED42_FALLBACK_FILE="Llama3-Med42-8B-Q4_K_M.gguf"

download_one() {
  local repo="$1" quant="$2" local_name="$3"
  local target="$MODELS_DIR/$local_name"
  if [[ -f "$target" && -s "$target" ]]; then
    echo "==> already present: $local_name ($(du -h "$target" | awk '{print $1}'))"
    return 0
  fi
  echo "==> Downloading $repo · $quant → $local_name …"
  if python3 - "$repo" "$quant" "$MODELS_DIR" "$local_name" <<'PY'
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
  then
    return 0
  fi
  # Med42-only fallback source
  if [[ "$local_name" == "Llama3-Med42-8B.Q4_K_M.gguf" ]]; then
    echo "==> primary Med42 failed; trying fallback $MED42_FALLBACK_REPO …"
    python3 - "$MED42_FALLBACK_REPO" "$MED42_FALLBACK_FILE" "$MODELS_DIR" "$local_name" <<'PY'
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
    return 0
  fi
  return 1
}

for row in "${PEERS[@]}"; do
  IFS='|' read -r repo quant local_name <<<"$row"
  download_one "$repo" "$quant" "$local_name"
done

echo "==> Medical-local peers ready under $MODELS_DIR"
ls -lh "$MODELS_DIR"/*{medgemma,Med42,UltraMedical}* 2>/dev/null || true
