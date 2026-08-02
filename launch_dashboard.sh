#!/bin/bash
# Avvia QVAC SDK sidecar (MedPsy reale) + dashboard Streamlit, poi apre il browser.
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

PORT=8501
URL="http://localhost:${PORT}"
LOG="/tmp/qvac-cloud-local-medical-cases-streamlit.log"
PIDFILE="/tmp/qvac-cloud-local-medical-cases-streamlit.pid"
SIDECAR_LOG="/tmp/qvac-sidecar.log"
SIDECAR_PIDFILE="/tmp/qvac-sidecar.pid"
SIDECAR_URL="http://127.0.0.1:8787"

if [ ! -d ".venv" ]; then
  echo "Prima esecuzione: avvio installazione automatica (venv)..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -q -r requirements.txt
else
  source .venv/bin/activate
fi

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_SHOW_EMAIL_PROMPT=false
export QVAC_DEVICE="${QVAC_DEVICE:-gpu}"
export QVAC_GPU_LAYERS="${QVAC_GPU_LAYERS:-99}"
export QVAC_WARM_LOAD="${QVAC_WARM_LOAD:-1}"

# Space-free model path (SDK workers break on paths with spaces).
MODEL_LINK="$HOME/.local/qvac-models/medpsy-4b-q4_k_m-imat.gguf"
MODEL_SRC="$PROJECT_DIR/models/medpsy-4b-q4_k_m-imat.gguf"
if [ -f "$MODEL_SRC" ]; then
  mkdir -p "$(dirname "$MODEL_LINK")"
  ln -sfn "$MODEL_SRC" "$MODEL_LINK"
  export QVAC_MODEL_PATH="${QVAC_MODEL_PATH:-$MODEL_LINK}"
fi

start_sidecar() {
  if curl -sf "$SIDECAR_URL/health" 2>/dev/null | grep -q '"modelLoaded":true'; then
    return 0
  fi
  if [ ! -f "$PROJECT_DIR/sidecar/qvac_server.mjs" ]; then
    echo "⚠️  Sidecar mancante. Esegui: ./scripts/setup_qvac_sidecar.sh" >&2
    return 1
  fi
  if [ ! -d "$PROJECT_DIR/sidecar/node_modules/@qvac/sdk" ]; then
    echo "⚠️  @qvac/sdk non installato. Esegui: ./scripts/setup_qvac_sidecar.sh" >&2
    return 1
  fi
  # Kill stale listener if health is broken
  if lsof -iTCP:8787 -sTCP:LISTEN -t >/dev/null 2>&1; then
    if ! curl -sf "$SIDECAR_URL/health" >/dev/null 2>&1; then
      lsof -tiTCP:8787 -sTCP:LISTEN 2>/dev/null | xargs -n1 kill 2>/dev/null || true
      sleep 1
    fi
  fi
  if ! curl -sf "$SIDECAR_URL/health" >/dev/null 2>&1; then
    (
      cd "$PROJECT_DIR/sidecar"
      nohup node qvac_server.mjs >>"$SIDECAR_LOG" 2>&1 &
      echo $! >"$SIDECAR_PIDFILE"
    )
  fi
  for _ in $(seq 1 60); do
    if curl -sf "$SIDECAR_URL/health" 2>/dev/null | grep -q '"modelLoaded":true'; then
      return 0
    fi
    sleep 1
  done
  echo "⚠️  Sidecar non pronto. Log: $SIDECAR_LOG" >&2
  return 1
}

open_browser() {
  if [ -d "/Applications/Safari.app" ]; then
    open -a Safari "$URL"
  else
    open "$URL"
  fi
}

start_sidecar || true

# Gia' in esecuzione → apri solo browser
if lsof -i ":${PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
  open_browser
  exit 0
fi

nohup streamlit run app.py \
  --server.port="${PORT}" \
  --server.headless=true \
  --server.showEmailPrompt=false \
  > "$LOG" 2>&1 &

echo $! > "$PIDFILE"

for _ in $(seq 1 30); do
  if curl -sf -o /dev/null "$URL"; then
    open_browser
    exit 0
  fi
  sleep 1
done

echo "Streamlit non partito. Log: $LOG" >&2
exit 1
