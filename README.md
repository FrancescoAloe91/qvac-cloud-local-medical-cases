# QVAC vs Cloud LLMs — Health Test

Reproducible clinical benchmark: **six OpenRouter cloud models** (free-web class + light fallbacks, **BYOK**) vs on-device **Tether QVAC MedPsy** (local **QVAC SDK** sidecar), scored by a **blind LLM-as-judge** (**DeepSeek R1**).

| | Local (full demo) | Streamlit Cloud |
|---|---|---|
| OpenRouter candidates + judge | ✅ your API key | ✅ visitor BYOK |
| QVAC MedPsy | ✅ QVAC SDK sidecar | ❌ skipped (cloud-only) |
| Automated Benchmark (main app) | ✅ | ✅ (cloud models + judge) |
| Private History (A / B / C) | ✅ scoped to your key | ✅ same (per key fingerprint) |
| Typical cost (`free_tier_match`, 6 cloud) | ~**$0.40–2.50** / case × 1 run | same |

**Live app:** https://francescoaloe91-qvac-vs-cloud-llms-health-test-app-wihxyd.streamlit.app  
**Repo:** https://github.com/FrancescoAloe91/qvac-vs-cloud-llms-health-test

---

## Privacy: your key (BYOK) + private History

**How the API key works (important):**

1. Paste your OpenRouter key in the **welcome popup** → **Save / update key** (or sidebar).  
2. That key is remembered **for your network IP only**. Refresh on the same PC/IP → field comes **pre-filled** (••••).  
3. A visitor with a **different IP** sees an **empty** field — they cannot spend your credits.  
4. Do **not** put `OPENROUTER_API_KEY` in Streamlit Cloud **Secrets**. A Secret is injected into the server for *every* visitor (shared wallet). BYOK in the UI is the correct path.

**History (Cases A / B / C):**

- Same OpenRouter key → same **History**, **Rebuild mean**, Custom Case gold/answers.  
- Different key → cannot see another visitor’s runs.  
- Artifacts: `artifacts/owners/<sha256(key)[:24]>/…` (key fingerprint only; raw key never in those JSON paths).  
- Cloud disk may clear when the app sleeps; durable archives → local install.

---

## Models (free-tier match)

Pinned in [`benchmark/models.yaml`](benchmark/models.yaml).

**Band A — free-web everyday class** (API proxies of chatgpt.com / claude.ai / Gemini free defaults):

| Role | OpenRouter ID |
|--|--|
| ChatGPT Instant | `openai/gpt-5.5` |
| Claude Sonnet 5 | `anthropic/claude-sonnet-5` |
| Gemini Flash | `google/gemini-3.5-flash` |

**Band B — lighter free-user proxies** (still billed on OpenRouter; not website $0):

| Role | OpenRouter ID |
|--|--|
| GPT Mini | `openai/gpt-5.4-mini` |
| Claude Haiku | `anthropic/claude-haiku-4.5` |
| Qwen Flash | `qwen/qwen3.6-flash` |

| Role | Runtime |
|--|--|
| Judge (blind) | `deepseek/deepseek-r1` |
| QVAC (default) | MedPsy-4B Q4 · QVAC SDK · $0 API |
| QVAC (toggle 3×) | + MedPsy-1.7B Q4 + MedPsy-4B Q8 |

**Live grid:** off = **3+3+1 (7)** · **3× QVAC** on = **3×3 (9)**.  
True OpenRouter **$0** models are only IDs ending in `:free` (e.g. Nemotron / GPT-OSS) — not GPT/Claude proprietary.

**Recommended demo path:** **Multi run ×5**. Single run stays available for a quick check.  
**QVAC only · $0** = local rehearsal (no cloud, no judge, no ranking).

---

## Scoring (LLM judge)

Full write-up: **[benchmark/SCORING.md](benchmark/SCORING.md)**.

**In short:**

1. DeepSeek R1 scores each section blind (Candidate 1/2/…).  
2. Host computes a **linear** section score (judge’s raw number is ignored):

   Gold: `100 × (0.50·alignment + 0.30·quality + 0.20·stem_spec)`;  
   Rubric: `100 × (0.30·must + 0.20·acceptable + 0.40·quality + 0.10·stem_spec)` → **cap 96.5**

3. **Ranking %** = weighted mean of sections (weights fixed in the case; diagnosis/safety usually heaviest).  
4. **100% is not used.** Candidates always get **distinct** accuracies (tie-break: safety → quality → stem → diagnosis).

| Parameter (gold) | Weight | Meaning |
|-----------|--------|---------|
| **alignment** | 50% | Semantic closeness to gold thesis (synonyms / near-equivalents OK) |
| **quality** | 30% | Clinical judgment — not writing style |
| **stem_spec** | 20% | Case-specific anchors (anti–generic paste) |

**Rebuild mean across N runs** (UI, $0 API) rescores **your** saved artifacts with this formula.

---

## Clone the full package (recommended)

One command installs Python deps, downloads the MedPsy **GGUF (~2.5 GB)** from Hugging Face, and sets up the **QVAC SDK** sidecar. The GGUF is **not** in git (GitHub size limit).

```bash
git clone https://github.com/FrancescoAloe91/qvac-vs-cloud-llms-health-test.git
cd qvac-vs-cloud-llms-health-test
chmod +x install.sh && ./install.sh
# edit .env → OPENROUTER_API_KEY=sk-or-v1-…  (FULL key from https://openrouter.ai/keys)

# Terminal A — QVAC MedPsy (on-device)
cd sidecar && npm start

# Terminal B — dashboard
source .venv/bin/activate && streamlit run app.py
# → http://localhost:8501  · Automated Benchmark only
```

**Needs:** Python 3.10+, Node.js ≥ 22.17, network for the GGUF download.  
**macOS:** OpenSSL 3 (handled by `scripts/setup_qvac_sidecar.sh`).  
**Windows:** `install.ps1`, then `cd sidecar; npm start` and `streamlit run app.py`.

See [`models/README.md`](models/README.md) for the GGUF source (`qvac/MedPsy-4B-GGUF`).

### What you see (screen recording)

1. Paste **your** OpenRouter key (welcome / sidebar) — unlocks private History  
2. **Custom Case** (main) or recall **Demo Case 1 / 2** + optional gold / checklist  
3. Cost estimate under Single / Multi / QVAC-only  
4. Confirm spend modal → live panels + KPIs  
5. Ranking + matrix; artifacts under `artifacts/owners/<your-fingerprint>/`  
6. **Rebuild mean across N runs** (3 / 5 / 10) · offline · KPI popup  
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
- History / Custom Case content is **per key**, not a public shared log.  
- QVAC MedPsy sidecar **cannot** run on Streamlit Cloud — use a local install for the full four-model demo.

Live: https://francescoaloe91-qvac-vs-cloud-llms-health-test-app-wihxyd.streamlit.app  

Pushing to `main` redeploys the cloud app.

---

## Project layout

```
app.py                 # Automated Benchmark (only Streamlit entry)
benchmark/             # cases, OpenRouter, judge, runner, CLI
benchmark/workspace.py # per-API-key private artifact folders
benchmark/models.yaml  # free_tier_match model IDs
benchmark/cases/       # Custom (caseC) + Demo 1/2 (caseA/B) — ids kept for History
sidecar/               # QVAC SDK HTTP bridge (npm; no node_modules in git)
artifacts/             # gitignored — owners/<fingerprint>/ per key
OLD/                   # unused legacy dashboard (not launched)
install.sh / install.ps1
```
