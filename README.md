# QVAC vs Cloud LLMs — Health Test

Reproducible clinical benchmark: **ChatGPT / Claude / Gemini** (via [OpenRouter](https://openrouter.ai), **BYOK**) vs on-device **Tether QVAC MedPsy 4B** (local **QVAC SDK** sidecar), scored by a **blind LLM-as-judge** (**DeepSeek R1**).

| | Local (full demo) | Streamlit Cloud |
|---|---|---|
| OpenRouter candidates + judge | ✅ your API key | ✅ visitor BYOK |
| QVAC MedPsy | ✅ QVAC SDK sidecar | ❌ skipped (cloud-only) |
| Automated Benchmark UI | ✅ | ✅ (cloud models + judge) |
| Typical cost (`free_tier_match`) | ~**$0.20–1.50** / case × 1 run | same |

**Live app:** https://francescoaloe91-qvac-vs-cloud-llms-health-test-app-wihxyd.streamlit.app  
**Repo:** https://github.com/FrancescoAloe91/qvac-vs-cloud-llms-health-test

---

## Models (free-tier match)

Pinned in [`benchmark/models.yaml`](benchmark/models.yaml):

| Role | OpenRouter ID | Display |
|--|--|--|
| ChatGPT | `openai/gpt-5.5` | GPT-5.5 Instant (API) |
| Claude | `anthropic/claude-sonnet-4.6` | Sonnet 4.6 Extra-class |
| Gemini | `google/gemini-3.5-flash` | 3.5 Flash |
| Judge (blind) | `deepseek/deepseek-r1` | Independent scorer (not ranked as a vendor) |
| QVAC | local sidecar | MedPsy-4B · QVAC SDK · $0 API |

**Recommended demo path:** **Multi run ×3**. Single run stays available for a quick check.  
**QVAC only · $0** = local rehearsal (no cloud, no judge, no ranking).

---

## Scoring (LLM judge)

- Same structured Q&A for every candidate; judge is **blind** (Candidate 1/2/…).
- **Section weights** are fixed per case (e.g. diagnosis / safety / tests…).
- Per-section score is **continuous linear 0–100** (not step bands):
  - `score ≈ 100 × (0.70 × must_include coverage + 0.30 × acceptable coverage)`
  - synonyms / clinical equivalents count
  - `must_not` / safety traps hard-cap that section
- Final **accuracy** = weighted mean of section scores (no winner-to-100 rescale).

Cases: **A** STEMI teaching · **B** mania+CKD teaching · **C** custom (your stem + gold checklist).

---

## Automated Benchmark (recommended UI)

```bash
cp .env.example .env   # OPENROUTER_API_KEY=sk-or-v1-… (FULL key, no "...")
pip install -r requirements.txt

# QVAC SDK sidecar (separate terminal; macOS needs OpenSSL 3)
./scripts/setup_qvac_sidecar.sh
cd sidecar && npm start

streamlit run app.py
# → sidebar / page menu → “Automated Benchmark”
```

### What you see (screen recording)

1. Case A/B/C + optional gold / checklist  
2. Cost estimate under Single / Multi / QVAC-only  
3. Confirm spend modal (OpenRouter)  
4. Live response panels (parallel stream) + KPI: TTFT, TPS, latency, **words**, **tokens**, $  
5. Sidebar **Run clock**: total / this-run / collect / judge (live seconds)  
6. Ranking + per-question matrix + artifacts under `artifacts/`  
7. Sidebar **History** → popup of a past run (does not open when you only change Case A/B/C)

**CLI:**

```bash
python -m benchmark list-cases
python -m benchmark dry-run --case caseA --n 1
python -m benchmark run --case caseA --n 3
```

---

## Local install (full: cloud + QVAC MedPsy)

The MedPsy **GGUF (~2.5 GB) is not in git** (GitHub size limit).  
`./install.sh` downloads it from Hugging Face and sets up the QVAC SDK sidecar.

```bash
git clone https://github.com/FrancescoAloe91/qvac-vs-cloud-llms-health-test.git
cd qvac-vs-cloud-llms-health-test
chmod +x install.sh && ./install.sh
# edit .env → OPENROUTER_API_KEY=sk-or-v1-… (full key)

# Terminal A
cd sidecar && npm start

# Terminal B
source .venv/bin/activate && streamlit run app.py
# → Automated Benchmark
```

**Needs:** Python 3.10+, Node.js ≥ 22.17, network for the GGUF download.  
**macOS:** OpenSSL 3 (handled by `setup_qvac_sidecar.sh`).

Windows: `install.ps1` then `cd sidecar && npm start` + `streamlit run app.py`.

See [`models/README.md`](models/README.md) for the GGUF source (`qvac/MedPsy-4B-GGUF`).

---

## Streamlit Cloud

Deploy from `main` / `app.py` (see [DEPLOY.md](DEPLOY.md)).  
Cloud hosts **cannot** run the QVAC sidecar — use BYOK OpenRouter for ChatGPT/Claude/Gemini + DeepSeek judge; skip QVAC or paste local QVAC output in the legacy flow.

Pushing to `main` redeploys: https://francescoaloe91-qvac-vs-cloud-llms-health-test-app-wihxyd.streamlit.app

---

## Project layout

```
benchmark/             # cases, OpenRouter, judge, runner, CLI
benchmark/models.yaml  # free_tier_match model IDs
benchmark/cases/       # caseA / caseB / caseC rubrics
sidecar/               # QVAC SDK HTTP bridge (npm; no node_modules in git)
pages/01_Automated_Benchmark.py   # demo studio
artifacts/             # run JSON (gitignored)
app.py                 # entry + legacy paste UI
```

Legacy paste / consensus UI remains in `app.py`; prefer **Automated Benchmark** for demos.
