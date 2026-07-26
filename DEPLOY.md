# Free deploy ($0)

## Option A — Streamlit Community Cloud (recommended)

1. Open: **[Deploy on Streamlit Cloud](https://share.streamlit.io/deploy?repository=FrancescoAloe91/qvac-vs-cloud-llms-health-test&branch=main&mainModule=app.py)**
2. Sign in with **GitHub** (free account)
3. Confirm:
   - Repository: `FrancescoAloe91/qvac-vs-cloud-llms-health-test`
   - Branch: `main`
   - Main file: `app.py`
4. Click **Deploy** — no credit card required
5. Live URL: **https://francescoaloe91-qvac-vs-cloud-llms-health-test-app-wihxyd.streamlit.app**

**On the cloud demo:**

- **Automated Benchmark** (`app.py`) with visitor **BYOK** OpenRouter (ChatGPT / Claude / Gemini + DeepSeek R1 judge).
- During judging: **live board** — left = collect-order FIFO queue; right = provisional ranking histogram (see README).
- **BYOK in the UI only:** each visitor pastes their own OpenRouter key and clicks **Save**. The app remembers it for **that visitor’s IP** (prefilled on refresh). Other IPs start with an **empty** field.
- **Do not** add `OPENROUTER_API_KEY` under Streamlit **Settings → Secrets**. That would load *your* key into the server for **every** visitor (they would all bill your account).
- **Private History / KPIs:** Custom Case + Demo runs live under `artifacts/owners/<hash of that visitor’s key>/`. Same key → same history and means; other keys cannot open those files.
- **QVAC MedPsy** (on-device SDK sidecar) is **local-only** — skipped on Streamlit Cloud. For the complete GGUF pack locally, see the README (`./install.sh --full-models`).
- Cloud disk may clear when the app sleeps; durable archives → run locally.

Redeploy: push to `main` (Cloud watches the repo).

---

## Option B — Render.com (free tier)

1. [render.com](https://render.com) → Sign up free (GitHub)
2. **New** → **Blueprint** → connect this repo
3. Render reads `render.yaml` and starts the dashboard
4. URL looks like `https://qvac-health-test.onrender.com`

Same limits: no QVAC sidecar on the free host; use BYOK + per-key History.

---

## Full local setup (live QVAC + private history on disk)

History, KPIs, and saved runs are scoped to **your OpenRouter key** (folder fingerprint). The key is remembered for **your client IP** only.

**macOS / Linux**

```bash
git clone https://github.com/FrancescoAloe91/qvac-vs-cloud-llms-health-test.git
cd qvac-vs-cloud-llms-health-test

# Default (~2.5 GB): MedPsy-4B Q4 only
./install.sh

# Optional — complete on-device pack (~14 GB): 3 MedPsy + Gemma/Llama/Phi
# ./install.sh --full-models
# or: ./scripts/download_all_ggufs.sh

# edit .env → OPENROUTER_API_KEY=sk-or-v1-… (full key)

cd sidecar && npm start           # Terminal A
# Terminal B:
source .venv/bin/activate && streamlit run app.py
```

Or `./launch_dashboard.sh` → `http://localhost:8501`

**Windows**

```powershell
git clone https://github.com/FrancescoAloe91/qvac-vs-cloud-llms-health-test.git
cd qvac-vs-cloud-llms-health-test
powershell -ExecutionPolicy Bypass -File install.ps1
# edit .env with your OpenRouter key
# optional full GGUFs: bash scripts/download_all_ggufs.sh  (Git Bash / WSL)
cd sidecar; npm start
# other terminal:
.\.venv\Scripts\Activate.ps1; streamlit run app.py
```

Artifacts path: `artifacts/owners/<sha256-fingerprint>/` — gitignored. Keep `.env` private; never commit keys or History.

