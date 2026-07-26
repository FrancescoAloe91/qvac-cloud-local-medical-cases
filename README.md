# QVAC vs Cloud LLMs — Health Test

Reproducible clinical benchmark: **3 cloud LLMs** (OpenRouter, **BYOK**) vs **on-device GGUFs** via the **QVAC SDK** sidecar (**Band B peers** + **Tether MedPsy**), scored by a **blind LLM-as-judge** (**DeepSeek R1**).

| | Local (full demo) | Streamlit Cloud |
|---|---|---|
| OpenRouter candidates + judge | ✅ your API key | ✅ visitor BYOK |
| On-device GGUFs (MedPsy + peers) | ✅ QVAC SDK sidecar | ❌ skipped (cloud-only) |
| Automated Benchmark (main app) | ✅ up to **9 models** | ✅ cloud + judge |
| Private History / KPIs / means | ✅ scoped to **your key** (+ key remembered per **IP**) | ✅ same |
| Typical cloud spend (3 cloud + judge) | ~**$0.25–1.50** / case × 1 run | same |

**Live app:** https://francescoaloe91-qvac-vs-cloud-llms-health-test-app-wihxyd.streamlit.app  
**Repo:** https://github.com/FrancescoAloe91/qvac-vs-cloud-llms-health-test

---

## Privacy: IP + API key (BYOK) + private History

Every user’s **scripts, dashboard, KPIs, rankings, Rebuild mean, and saved run outputs** are tied to **that user**, not to a shared global log.

| Mechanism | What it does |
|--|--|
| **API key (BYOK)** | OpenRouter spend + **History folder** `artifacts/owners/<sha256(key)[:24]>/` (fingerprint only; raw key never in paths) |
| **Client IP** | Remembers / prefills **your** key on refresh for **that IP only**. Another IP sees an empty key field |
| **Same key** | Same History, KPIs, multi-run means, Custom Case gold |
| **Different key** | Cannot see another user’s prompts, outputs, scores, or means |

**How to use the key:**

1. Paste your OpenRouter key in the **welcome popup** → **Save / update key** (or sidebar).  
2. Same PC/IP → field comes **pre-filled** (••••).  
3. Different IP → empty field (cannot spend your credits).  
4. Do **not** put `OPENROUTER_API_KEY` in Streamlit Cloud **Secrets** (shared wallet for every visitor). BYOK in the UI is correct.

### What ships on GitHub vs what stays on your machine

| On GitHub (public) | Local only (gitignored · **never** published) |
|--|--|
| App / benchmark code, teaching demo cases, scoring docs | `artifacts/` — **your** prompts, model outputs, judge scores, multi-run means, KPIs |
| Install + **download scripts** for GGUFs | `models/*.gguf` — weights fetched from Hugging Face onto **your** disk |
| | `.env`, API keys, `.case_snapshots.json` |

Cloning the repo does **not** include anyone else’s History or GGUF weights. Each user builds an empty private History after their first runs.

---

## Models (live 9-model grid)

Pinned in [`benchmark/models.yaml`](benchmark/models.yaml) + [`benchmark/qvac_variants.py`](benchmark/qvac_variants.py).

**Band A — cloud (OpenRouter $):**

| Role | OpenRouter ID |
|--|--|
| ChatGPT Instant | `openai/gpt-5.5` |
| Claude Sonnet 5 | `anthropic/claude-sonnet-5` |
| Gemini Flash | `google/gemini-3.5-flash` |

**Band B — open Q4 GGUFs on-device** (same QVAC sidecar · $0 inference · prompt stays local):

| Role | GGUF under `models/` |
|--|--|
| Gemma 2 2B IT | `gemma-2-2b-it-Q4_K_M.gguf` |
| Llama 3.2 3B | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` |
| Phi-3.5 mini | `Phi-3.5-mini-instruct-Q4_K_M.gguf` |

**QVAC MedPsy — on-device** (same sidecar):

| Role | GGUF |
|--|--|
| MedPsy-1.7B Q4 | `medpsy-1.7b-q4_k_m-imat.gguf` |
| MedPsy-4B Q4 (default) | `medpsy-4b-q4_k_m-imat.gguf` |
| MedPsy-4B Q8 | `medpsy-4b-q8_0.gguf` |

| Role | Runtime |
|--|--|
| Judge (blind) | `deepseek/deepseek-r1` (cloud) |

**Live grid:** **3× QVAC** on (default) = **3 cloud + 3 local peers + 3 MedPsy (9)**. Toggle off = only MedPsy-4B Q4 (7).  
**Only local ×N** = 6 on-device GGUFs, no cloud collect (judge $ only).  
**Recommended:** Multi / Only local with **N = 5–30** for stable means (per-model **Runs** column can differ if you repeat local more than cloud).

---

## Optional: download the **complete** local package

GGUF weights are **not** stored in git (GitHub size limits). Anyone can still get a **full offline model pack** from Hugging Face with one command (~**14 GB**: 3 MedPsy + Gemma + Llama + Phi).

```bash
# After clone — full on-device grid
./install.sh --full-models
# or later:
./scripts/download_all_ggufs.sh
```

| Install mode | What you get | Size |
|--|--|--|
| `./install.sh` (default) | Python + sidecar + **MedPsy-4B Q4 only** | ~2.5 GB |
| `./install.sh --full-models` | Above + **all 3 MedPsy + 3 Band B peers** | ~14 GB |
| `./scripts/download_local_peers.sh` | Band B only | ~6 GB |
| `./scripts/download_medpsy_gguf.sh` | One MedPsy quant (see env vars) | varies |

Details: [`models/README.md`](models/README.md).  
Downloading GGUFs does **not** copy or overwrite private History under `artifacts/`.

---

## Scoring (LLM judge)

Full write-up: **[benchmark/SCORING.md](benchmark/SCORING.md)**.

**In short:**

1. DeepSeek R1 scores each section blind (Candidate 1/2/…).  
2. Host computes a **linear** section score (judge’s raw number is ignored):

   Gold: `100 × (0.50·alignment + 0.30·quality + 0.20·stem_spec)`;  
   Rubric: `100 × (0.30·must + 0.20·acceptable + 0.40·quality + 0.10·stem_spec)` → **cap 96.5**

3. **Ranking %** = weighted mean of sections (case weights; diagnosis/safety usually heaviest).  
4. Candidates get **distinct** accuracies (tie-break: safety → quality → stem → diagnosis).

| Parameter (gold) | Weight | Meaning |
|-----------|--------|---------|
| **alignment** | 50% | Semantic closeness to gold thesis |
| **quality** | 30% | Clinical judgment — not writing style |
| **stem_spec** | 20% | Case-specific anchors |

**Rebuild mean across N runs** (UI, $0 API) rescores **your** saved artifacts (3 / 5 / 10 / 20 / 30) with this formula. Tables/charts show only the **current 9 models**, with a **Runs** column (how many scored passes that model/version contributed).

---

## Clone & run (local)

```bash
git clone https://github.com/FrancescoAloe91/qvac-vs-cloud-llms-health-test.git
cd qvac-vs-cloud-llms-health-test

# Lightweight (~2.5 GB) OR full pack (~14 GB)
chmod +x install.sh && ./install.sh
# ./install.sh --full-models

# edit .env → OPENROUTER_API_KEY=sk-or-v1-…  (full key: https://openrouter.ai/keys)

# Terminal A — QVAC sidecar
cd sidecar && npm start

# Terminal B — dashboard
source .venv/bin/activate && streamlit run app.py
# → http://localhost:8501
```

**Needs:** Python 3.10+, Node.js ≥ 22.17, network for Hugging Face downloads.  
**macOS:** OpenSSL 3 via `scripts/setup_qvac_sidecar.sh`.  
**Windows:** `install.ps1`, then sidecar + `streamlit run app.py`.

### Typical session

1. Paste **your** OpenRouter key (welcome / sidebar) — unlocks private History for **this key**  
2. **Custom Case** or **Demo Case 1 / 2** + optional gold  
3. Cost estimate under Single / Multi / QVAC-only / Only local  
4. Confirm spend → live panels + KPIs  
5. Ranking + matrix; files under `artifacts/owners/<your-fingerprint>/`  
6. **Rebuild mean** (offline) · KPI popup  
7. Sidebar **History** → only runs saved with **this** key  

**CLI** (same private folder when `OPENROUTER_API_KEY` is set):

```bash
python -m benchmark list-cases
python -m benchmark dry-run --case caseA --n 1
python -m benchmark run --case caseA --n 3
```

---

## Streamlit Cloud (shared demo URL)

Deploy from `main` / `app.py` — see **[DEPLOY.md](DEPLOY.md)**.

- Visitors bring **their own** OpenRouter key (BYOK).  
- Key remembered per **visitor IP**; History / KPIs per **key fingerprint**.  
- QVAC sidecar **cannot** run on Streamlit Cloud — use a local install for on-device GGUFs.

Live: https://francescoaloe91-qvac-vs-cloud-llms-health-test-app-wihxyd.streamlit.app  

Pushing to `main` redeploys the cloud app.

---

## Project layout

```
app.py                      # Automated Benchmark (Streamlit entry)
benchmark/                  # cases, OpenRouter, judge, runner, CLI
benchmark/workspace.py      # per-API-key private artifact folders
benchmark/models.yaml       # cloud model IDs
benchmark/cases/            # teaching demos + custom case ids
sidecar/                    # QVAC SDK HTTP bridge (npm; no node_modules in git)
models/                     # GGUFs downloaded locally (gitignored)
artifacts/                  # gitignored — owners/<fingerprint>/ per key
scripts/download_all_ggufs.sh
scripts/download_local_peers.sh
scripts/download_medpsy_gguf.sh
install.sh / install.ps1
```
