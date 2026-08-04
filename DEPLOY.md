# Deployment

## Hosted Streamlit

1. Deploy `app.py` from `main`.
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

5. Add these values to the hosting secret manager:

```text
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_ANON_KEY=...
APP_ENCRYPTION_KEY=...
```

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
