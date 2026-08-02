#!/bin/bash
# Stop Streamlit dashboard + QVAC sidecar. Does not start/stop Ollama.
PIDFILE="/tmp/qvac-medical-cases-streamlit.pid"
SIDECAR_PIDFILE="/tmp/qvac-sidecar.pid"
PORT=8501
SIDECAR_PORT=8787

if [ -f "$PIDFILE" ]; then
  kill "$(cat "$PIDFILE")" 2>/dev/null || true
  rm -f "$PIDFILE"
fi
if [ -f "$SIDECAR_PIDFILE" ]; then
  kill "$(cat "$SIDECAR_PIDFILE")" 2>/dev/null || true
  rm -f "$SIDECAR_PIDFILE"
fi

pkill -f "streamlit run app.py" 2>/dev/null || true
pkill -f "qvac_server.mjs" 2>/dev/null || true
lsof -ti ":${PORT}" | xargs kill 2>/dev/null || true
lsof -ti ":${SIDECAR_PORT}" | xargs kill 2>/dev/null || true

echo "Dashboard and QVAC sidecar stopped."
