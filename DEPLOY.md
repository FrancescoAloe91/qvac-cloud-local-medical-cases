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
- **Private History:** each visitor’s Case A / B / C runs (including Case C gold) are stored under a folder named from a **hash of their API key**. Same key → same history. Other keys cannot open those artifacts. The raw key is never saved on disk.
- **QVAC MedPsy** (on-device SDK sidecar) is **local-only** — skipped on Streamlit Cloud.
- Cloud disk may clear when the app sleeps; durable archives → run locally.

Redeploy: push to `main` (Cloud watches the repo). Do **not** put `OPENROUTER_API_KEY` in Streamlit Secrets if you want true BYOK per visitor — let each user paste their own key in the UI.

---

## Option B — Render.com (free tier)

1. [render.com](https://render.com) → Sign up free (GitHub)
2. **New** → **Blueprint** → connect this repo
3. Render reads `render.yaml` and starts the dashboard
4. URL looks like `https://qvac-health-test.onrender.com`

Same limits: no QVAC sidecar on the free host; use BYOK + per-key History.

---

## Full local setup (live QVAC + private history on disk)

**macOS / Linux**

```bash
git clone https://github.com/FrancescoAloe91/qvac-vs-cloud-llms-health-test.git
cd qvac-vs-cloud-llms-health-test
./install.sh
# edit .env → OPENROUTER_API_KEY=sk-or-v1-… (full key)

./scripts/setup_qvac_sidecar.sh   # if install.sh already ran, skip
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
cd sidecar; npm start
# other terminal:
.\.venv\Scripts\Activate.ps1; streamlit run app.py
```

Artifacts path: `artifacts/owners/<sha256-fingerprint>/` — keep `.env` private; never commit keys.
