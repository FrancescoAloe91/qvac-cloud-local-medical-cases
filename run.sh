#!/bin/bash
# Avvio rapido senza prompt email di Streamlit al primo run.
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creo ambiente virtuale..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_SHOW_EMAIL_PROMPT=false

echo ""
echo "Avvio QVAC vs Cloud LLMs - Medical Cases..."
echo "Se il browser non si apre da solo, vai su: http://localhost:8501"
echo ""

streamlit run app.py --server.showEmailPrompt=false
