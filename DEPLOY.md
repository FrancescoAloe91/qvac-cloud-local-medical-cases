# Deployment

## Hosted Streamlit (Community Cloud)

**Target URL (custom subdomain):**
https://qvac-cloud-local-medical-cases.streamlit.app

**Repo:** `FrancescoAloe91/qvac-cloud-local-medical-cases` · branch `main` ·
entrypoint `streamlit_app.py` (Community Cloud default; `app.py` also works)

There is **no** Streamlit Cloud rename/redeploy API usable from this repo’s CLI
(`gh` / `streamlit` cannot manage Community Cloud apps). After a GitHub repo
rename, Community Cloud often loses admin control of the old app (view-only /
stuck waking). Fix by recreating the app in the UI.

### Recreate / rename after GitHub repo rename

Old slug (orphaned / stale):
`https://francescoaloe91-qvac-vs-cloud-llms-health-test-app-wihxyd.streamlit.app`
(tied to former repo name `qvac-vs-cloud-llms-health-test`).

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with the
   GitHub account that owns `FrancescoAloe91/qvac-cloud-local-medical-cases`.
2. If the old app still appears:
   - Overflow (⋮) → **Delete** if Delete is enabled; or
   - If the app is **view-only** (repo rename broke GitHub coordinates): follow
     [Streamlit’s rename docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/rename-your-app)
     — temporarily restore the old GitHub name *or* ask Snowflake/Streamlit
     support to delete the disconnected app — then continue below.
3. **New app** → choose repository
   `FrancescoAloe91/qvac-cloud-local-medical-cases`, branch `main`,
   Main file path `streamlit_app.py` (Cloud default; or `app.py`) → Deploy.
4. After deploy: overflow (⋮) → **Settings** → **General** → set App URL
   subdomain to `qvac-cloud-local-medical-cases` (6–63 chars) → **Save**.
   Public URL becomes
   `https://qvac-cloud-local-medical-cases.streamlit.app`.
5. **Settings → Secrets** (TOML). Do **not** set a shared
   `OPENROUTER_API_KEY` (visitors use BYOK). Optional durable vault:

```toml
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_ANON_KEY = "..."
APP_ENCRYPTION_KEY = "..."
```

6. Reboot the app from the Cloud manager if secrets change. Confirm the new URL
   loads (not the old `…qvac-vs-cloud-llms-health-test…` slug).

Pushing to `main` triggers Streamlit’s normal GitHub auto-redeploy for an
already-connected app. An empty commit alone cannot create a new Cloud app or
rename a subdomain.

### Hosted secrets / behavior

1. Deploy with Main file path `streamlit_app.py` (or `app.py`) from `main`
   (as above).
2. Do not configure a shared `OPENROUTER_API_KEY`. Visitors use BYOK.
   Both Comprehension and Structured strip any process-wide
   `OPENROUTER_API_KEY` on Streamlit Cloud and refuse env fallback when the
   session key is empty (no silent host-key spend).
3. Create a Supabase project and apply
   `supabase/migrations/202607270001_secure_benchmark.sql`.
4. Generate a Fernet key:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

5. Add the Supabase / Fernet values to Streamlit **Secrets** (see TOML above).

The anon key is intended for clients; security comes from authenticated user
sessions and Row Level Security. `APP_ENCRYPTION_KEY` is server-secret and must
never be committed or exposed. Rotating it requires re-encrypting existing rows.

Hosted behavior:

- each visitor signs into a Supabase account;
- OpenRouter keys and complete artifacts are encrypted before persistence;
- no IP-address key vault and no process-global visitor key;
- cloud candidates/judges run through the visitor’s explicit session key;
- QVAC/GGUF models are unavailable unless the host also runs the local sidecar;
- technical failures are N/A and remain visible in artifacts.

If Supabase variables are absent, hosted keys and History stay session-only
(Comprehension and Structured): no plaintext run JSON on the host FS. The UI
shows a quiet caption for this state (no durable account vault).

Honesty fence (product copy, not scoring): cloud slots are OpenRouter API routes
≠ consumer ChatGPT/Claude/Gemini web; scores are gold-relative / not medical
validity; judge is uncalibrated; cost estimate ≠ invoice; teaching pack ≠ EHR.

**Deployability checklist:** `requirements.txt` (hashed lock from
`requirements.in`), entrypoint `streamlit_app.py` (thin wrapper → `app.py`;
both OK as Main file path), no `packages.txt` (no apt deps).
`.streamlit/config.toml` is for local UX; Community Cloud overrides hosting.

## Local

```bash
./install.sh
# or ./install.sh --full-models

# optional local convenience only
cp .env.example .env

# terminal A
cd sidecar && npm start

# terminal B
source .venv/bin/activate
streamlit run app.py
```

Local artifacts are written atomically under the gitignored `artifacts/owners/`
workspace. A session key is passed explicitly to OpenRouter calls; it is not
copied into process-global environment state.

## Release checks

```bash
python -m pip install --require-hashes -r requirements-dev.txt
python -m compileall -q app.py benchmark lib tests
pytest -q
node --check sidecar/qvac_server.mjs
cd sidecar && npm ci --ignore-scripts && npm audit --omit=dev --audit-level=high
```

The same checks run in `.github/workflows/quality.yml`.
