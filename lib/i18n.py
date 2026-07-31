"""UI strings — default English, optional Italian."""

from typing import Optional

DEFAULT_LANG = "en"
SUPPORTED_LANGS = ("en", "it")

_STRINGS = {
    "en": {
        "page_title": "QVAC vs Cloud LLMs - Health Test",
        "lang_label": "Language",
        "title": "QVAC vs Cloud LLMs - Health Test",
        "subtitle": "Identical prompt for every model · QVAC-only depth setting · Manual cloud paste · Diagnosis comparison",
        "cloud.demo_banner": "☁️ **Public demo (free hosting)** — Cloud LLM paste + rankings work here. **Live QVAC Run benchmark** needs a local Mac/PC with the QVAC sidecar (`./install.sh` + `sidecar`). You can paste a QVAC answer generated locally into the QVAC box.",
        "cloud.demo_link": "Full local setup (free): [GitHub README](https://github.com/FrancescoAloe91/qvac-vs-cloud-llms-health-test) · [Live demo](https://francescoaloe91-qvac-vs-cloud-llms-health-test-app-wihxyd.streamlit.app)",
        "sidebar.reset": "Reset",
        "sidebar.reset_help": "Clear everything including wallet to 0 USDT",
        "sidebar.cloud_tiers": "Cloud model versions",
        "sidebar.cloud_tiers_help": "Shown on cards and charts. Claude shows model + thinking in the chat bar; ChatGPT free often hides the model name — note what you see before pasting.",
        "sidebar.cloud_tiers_howto": "**How to check:** Claude → model name + thinking (e.g. Sonnet · Extra) in the top bar. ChatGPT → free tier often shows no model; optional: ask \"Which model are you?\" once. Gemini → model picker bottom-left if available.",
        "sidebar.cloud_tiers_save": "Save version labels",
        "sidebar.cloud_tiers_saved": "Cloud version labels saved",
        "sidebar.cloud_tiers_qvac_auto": "Auto-detected from your local setup (GGUF quant + QVAC sidecar model).",
        "sidebar.privacy_note": "QVAC generation is on-device; extract/judge still use OpenRouter when scoring.",
        "sidebar.wallet": "Wallet",
        "sidebar.wallet_caption": "Simulated USDT rewards — sell anonymized records below.",
        "sidebar.test_case": "Test case",
        "sidebar.scenario": "Scenario",
        "sidebar.load_case": "Load case",
        "sidebar.vlm": "VLM",
        "sidebar.vlm_upload": "Report / MRI upload",
        "sidebar.vlm_extract": "Extract text",
        "sidebar.vlm_add": "Add to case",
        "sidebar.saved_cases": "Saved case results",
        "sidebar.saved_cases_help": "Load or recall saved benchmark results. Green = ready to recall.",
        "sidebar.save_slot": "Save current benchmark",
        "sidebar.save_slot_help": "Stores charts + table for the active case (overwrites if saved again).",
        "sidebar.save_slot_toast": "**{case}** saved — click the slot anytime to recall it.",
        "sidebar.save_case5_run": "💾 Save run {n}/{max}",
        "sidebar.save_case5_run_rolling": "💾 Save new run (last {max} kept)",
        "sidebar.save_case5_help": "Case 5 keeps the **last {max}** benchmark runs; older saves are dropped. Clinical scores use a rolling average of those runs.",
        "sidebar.save_case5_run_toast": "Run **{n}/{max}** saved — finale uses the rolling average of the last {max} runs.",
        "sidebar.save_case5_run_toast_rolling": "Run saved — oldest dropped; finale averages the last **{max}** runs.",
        "sidebar.case5_runs": "{n}/{max} runs kept · rolling average in finale",
        "sidebar.viewing_case5_avg": "rolling average of last {n} runs",
        "sidebar.slot_badge_empty": "Empty",
        "sidebar.slot_badge_filled": "Saved",
        "sidebar.slot_badge_viewing": "Viewing",
        "sidebar.slot_empty_hint": "No benchmark saved yet",
        "sidebar.slot_saved_at": "Saved at {time}",
        "sidebar.slot_load": "Load",
        "sidebar.slot_recall": "Recall",
        "sidebar.slot_viewing": "Viewing",
        "sidebar.saved_count_inline": "{n}/{total}",
        "sidebar.finale_btn": "FINAL AVERAGED RANKING",
        "sidebar.finale_title": "Final averaged ranking",
        "sidebar.clear_saved": "Reset saved results",
        "sidebar.clear_saved_help": "Deletes all saved case snapshots (cases 1–5). Cannot be undone.",
        "sidebar.clear_saved_title": "Clear all saved case results?",
        "sidebar.clear_saved_body": "Removes every saved snapshot (cases 1–5). The current benchmark and wallet are **not** affected.",
        "sidebar.clear_saved_confirm": "Yes, clear saved",
        "sidebar.viewing_saved": "📂 Viewing saved result — **{case}**",
        "sidebar.return_live": "Back to live",
        "sidebar.saved_count": "**{n}/{total}** case slots filled",
        "case.clinical": "Clinical case",
        "tier.section": "QVAC analysis depth",
        "qvac.instruction": (
            "Provide a balanced, complete clinical assessment: ranked differential diagnosis, "
            "recommended tests, urgency level and red flags, and initial management plan."
        ),
        "badge.local": "On-device",
        "eyebrow.run": "Run benchmark",
        "run.caption": "QVAC runs locally with MedPsy's standard clinical instruction. The prompt below is identical for every cloud site.",
        "prompt.expander": "Clinical prompt (same text for cloud sites; QVAC adds its instruction automatically)",
        "prompt.copy": "Copy prompt to clipboard",
        "prompt.copied": "Prompt copied. Paste it manually on each cloud site.",
        "browser.open": "Open ChatGPT, Claude and Gemini in the browser",
        "browser.open_btn": "Cloud LLMs (ChatGPT, Claude, Gemini)",
        "browser.help": "Opens ChatGPT, Claude and Gemini in 3 tabs. Copy and paste the prompt on each site manually.",
        "caption.cloud": "Cloud sites cannot be embedded in this dashboard (security). Paste responses below. The depth selector only changes how QVAC MedPsy reasons on-device — ChatGPT, Claude and Gemini always receive the exact same plain prompt, so the comparison stays fair. ⚠️ If a cloud site answers in a different language than the others, word-overlap KPIs (Reliability, Accuracy) will honestly read lower even for the same diagnosis — for the fairest comparison, ask each site to answer in the same language as the case (e.g. add \"Please answer in English\").",
        "benchmark.run": "Run benchmark",
        "benchmark.spinner": "Running local QVAC inference...",
        "browser.opened": "Opened",
        "gold.section": "Reference diagnosis (optional)",
        "gold.use": "Use confirmed clinical diagnosis as reference",
        "gold.help": "When enabled, ranking measures deviation from this diagnosis. When disabled, inter-model consensus is used.",
        "gold.caption": "You may use your real pathology in **anonymized** form (no name, dates, hospital, city). Example: «Confirmed diagnosis: relapsing-remitting multiple sclerosis» or paste a de-identified report.",
        "gold.qvac_variance": "QVAC is a 4B on-device model via the **QVAC sidecar** (`@qvac/sdk`). Controlled track uses explicit sampling; native-default keeps model defaults. Save up to **{max} runs** — the finale uses a rolling average for a stable clinical score.",
        "gold.input": "Correct diagnosis / reference report",
        "gold.placeholder": "E.g. Confirmed diagnosis: acute coronary syndrome (NSTEMI). Labs: positive troponin, ECG with ST elevation...",
        "output.section": "Diagnostic output — paste cloud responses here",
        "output.link": "Link ↗",
        "output.placeholder_cloud": "Paste the response from the official site here...",
        "output.placeholder_qvac": "Generated by local QVAC MedPsy when you run the benchmark",
        "output.diagnosis": "Diagnosis",
        "output.stats": "TTFT {ttft}s · TPS {tps} · Latency {latency}s · RAM {ram}GB",
        "recalc": "Recalculate diagnosis comparison",
        "warn.no_output": "Run the QVAC benchmark and paste cloud responses, then recalculate.",
        "kpi.performance": "Performance KPI",
        "kpi.performance_caption": "Only QVAC measures TTFT/TPS locally. Cloud sites are not measurable from the web UI.",
        "kpi.consensus": "Inter-model consensus KPI",
        "kpi.consensus_caption": "Measures how much models agree on the same prompt (including QVAC MedPsy).",
        "kpi.primary_agreement": "Primary diagnosis agreement",
        "kpi.qvac_cloud": "QVAC ↔ Cloud",
        "kpi.models_compared": "Models compared",
        "kpi.consensus_keywords": "Consensus keywords",
        "kpi.matrix_expander": "Pairwise concordance matrix",
        "kpi.gold_section": "KPI vs clinical reference diagnosis",
        "kpi.gold_caption": "Comparison with your confirmed reference (anonymized). Scores reflect **clinical meaning** — diagnosis, plan/next steps, urgency — not identical wording. Uses local semantic embeddings (same engine for all models).",
        "kpi.gold_diagnoses": "Diagnoses extracted from reference",
        "kpi.gold_keywords": "Reference keywords",
        "ranking.final": "Final ranking",
        "ranking.section": "Ranking & score",
        "ranking.dual_explanation": (
            "Two **independent** rankings — they are never averaged. "
            "**Consensus** (#1 = 100%): who agrees most with the other LLMs on this case. "
            "**Clinical** (#1 = best match): who is closest to your **confirmed diagnosis** (100% = perfect match)."
        ),
        "ranking.dual_summary": (
            "🥇 **Best vs confirmed diagnosis:** {clin_name} ({clin_score}%) · "
            "🥇 **Best inter-LLM consensus:** {cons_name} ({cons_score}%)"
        ),
        "ranking.clinical_chart_note": (
            "Clinical chart shows **absolute** match with your confirmed diagnosis — "
            "100% only if the model aligns perfectly with the reference, not by relative scaling."
        ),
        "ranking.score_story_title": "What 0–100% measures (demo)",
        "ranking.score_story_gold_intro": (
            "You enabled a **confirmed reference diagnosis** — the dashboard shows **two independent scores**. "
            "Both use **local semantic embeddings** (meaning, not copy-paste wording)."
        ),
        "ranking.score_story_gold_clinical": (
            "**Ref.% / Clinical chart (absolute 0–100)** — how close each answer is to **your real reference**: "
            "**40%** diagnosis & differential · **30%** plan & next steps · **20%** urgency/triage · **10%** clinical summary. "
            "**100%** = same clinical meaning as the reference; **0%** = unrelated."
        ),
        "ranking.score_story_gold_consensus": (
            "**Cons.% / Consensus chart (relative)** — who agrees most with the **other LLMs** on this case: "
            "**40%** diagnosis · **30%** plan · **20%** urgency · **10%** summary — then **#1 is rescaled to 100%**. "
            "Measures group agreement, not correctness vs ground truth."
        ),
        "ranking.score_story_consensus_intro": (
            "**No confirmed diagnosis this round** — scores measure **inter-model agreement**, not medical correctness."
        ),
        "ranking.score_story_consensus_formula": (
            "**Cons.% (0–100)** = **true weighted average** of four continuous dimension scores (0–100 each, from local embeddings): "
            "**40%** diagnosis · **30%** plan & next steps · **20%** urgency · **10%** clinical summary — "
            "then **#1 in the group = 100%** (relative rescale only on Cons.%). "
            "Enable **confirmed diagnosis** above for the **absolute** clinical match vs your real case. "
            "Use **🔍 See exactly how every score was calculated** to audit the numbers."
        ),
        "chart.privacy_scale_low": "0",
        "chart.privacy_scale_high": "100",
        "chart.privacy_scale_cloud": "cloud",
        "chart.privacy_scale_local": "on-device",
        "chart.privacy_caption": "0% = answer pasted from a cloud site (case left the device) · 100% = QVAC MedPsy generation stayed on-device (extract/judge may still use OpenRouter).",
        "ranking.legend": (
            "**Consensus score %** = weighted blend of diagnosis match (40%), plan & next steps (30%), "
            "urgency agreement (20%), clinical summary (10%) — **rescaled so #1 in the group = 100%**. "
            "When a confirmed diagnosis is set, **Ref.%** and the clinical chart are **absolute** "
            "(100% = perfect match with your reference — never rescaled). "
            "**Cons.%** alone is rescaled (#1 = 100%) because there is no ground truth. "
            "**Privacy** row (0–100) is informational only — not part of any clinical or consensus score."
        ),
        "narrative.section": "Why this ranking?",
        "narrative.section_caption": "Short summary per model.",
        "narrative.compact_leader": "Group leader — **{score}%** consensus.",
        "narrative.compact_leader_clinical": "Best match vs reference — **{score}%**.",
        "narrative.compact_laggard": "Lowest here — **#{rank}** at **{score}%**.",
        "narrative.compact_mid": "Rank **#{rank}** — **{score}%**.",
        "narrative.compact_dual": "Consensus **#{rank_cons}** ({score_cons}%) · Clinical **#{rank_clin}** ({score_clin}%).",
        "narrative.compact_urgency_diff": "Urgency **{label}** (group: **{majority}**).",
        "narrative.verdict_high": "Consensus score **{score}/100** — strong agreement with the rest of the group, both in wording and in clinical meaning{sem_clause}.",
        "narrative.verdict_mid": "Consensus score **{score}/100** — decent agreement with the group, with some differences in phrasing or emphasis{sem_clause}.",
        "narrative.verdict_low": "Consensus score **{score}/100** — this answer diverges more from the group on this case{sem_clause}.",
        "narrative.dual_ranks": "**Rank consensus #{rank_cons}** ({score_cons}%) · **Rank clinical #{rank_clin}** ({score_clin}%) — these are independent.",
        "narrative.clinical_score": "Clinical score vs. confirmed diagnosis: **{score}/100** (rank **#{rank}**). 100% = perfect match with your reference.",
        "narrative.top_consensus": "**Consensus leader:** best inter-LLM agreement this round.",
        "narrative.bottom_consensus": "**Consensus note:** lowest agreement with the other models on this case.",
        "narrative.top_clinical": "**Clinical leader:** closest match to the confirmed diagnosis.",
        "narrative.bottom_clinical": "**Clinical note:** furthest from the confirmed diagnosis on this case.",
        "narrative.sem_clause": " (meaning-level similarity to the others: **{sem}%**)",
        "narrative.concordance": "Agrees with **{rel}%** of the other models' differential diagnoses (reliability) and matches **{acc}%** of the cross-model consensus keywords (accuracy).",
        "narrative.gold_accuracy": "Matches the confirmed reference diagnosis with **{acc}%** keyword accuracy and covers **{cov}%** of its differential list{sem_clause}.",
        "narrative.richness": "Extracted **{n}** differential-diagnosis entries and **{kw}** distinct clinical keywords from its response.",
        "narrative.urgency_match": "Flagged this case as **{label}** urgency, in line with the majority of models.",
        "narrative.urgency_diff": "Flagged this case as **{label}** urgency, while most models leaned **{majority}**.",
        "narrative.urgency_none": "No explicit urgency/triage level could be parsed from its response.",
        "narrative.privacy_cloud": "Required sending the clinical case to {vendor}'s cloud servers to run — 0% on-device privacy.",
        "narrative.privacy_local": "QVAC generation stayed on-device; extract/judge still leave via OpenRouter when used.",
        "narrative.top": "**Strength:** best overall balance of clinical alignment and completeness this round.",
        "narrative.bottom": "**Note:** lower alignment with the group on this case — this can mean a genuinely different (possibly still valid) differential, or simply a less structured pasted response, not necessarily lower medical quality.",
        "charts.section": "Benchmark visualization",
        "charts.tab_radar": "Multidimensional radar",
        "charts.tab_bars": "Bar ranking",
        "charts.tab_dims": "Dimension breakdown",
        "charts.dims_privacy_note": "Dimension breakdown is in this tab — privacy is already shown compactly above the ranking charts.",
        "decision.section": "Decision & compliance",
        "decision.lead": "Now that you've seen the results side by side — which path would you actually pick for this case?",
        "decision.cloud": "Cloud",
        "decision.cloud_err": "HIPAA/GDPR violation. 100% data leak.",
        "decision.cloud_caption": "Data leaves the device — no DPA, no consent trail.",
        "decision.qvac": "Local QVAC",
        "decision.qvac_ok": "Encrypted report. Generation stayed on-device.",
        "decision.qvac_caption": "QVAC generation on-device; extract/judge may still use OpenRouter.",
        "decision.sell": "Sell anonymized",
        "decision.sell_caption": "Simulate a paid, anonymized data sale.",
        "tier.light.desc": "Fast concise answer — top 3 hypotheses, urgency, 2 key tests.",
        "tier.medium.desc": "Balanced workup — full DDx, tests, red flags, initial plan.",
        "tier.premium.desc": "Exhaustive specialist review — scoring, risk stratification, follow-up.",
        "tier.instruction.light": "Reply concisely (max 200 words). List only the top 3 diagnostic hypotheses, 2 key tests, and urgency level.",
        "tier.instruction.medium": "Provide a balanced assessment: ranked differential diagnosis, tests to order, red flags, and initial treatment plan.",
        "tier.instruction.premium": "Provide an exhaustive specialist assessment: full differential with probability scoring, prioritized tests, risk stratification, detailed treatment plan and follow-up.",
        "prompt.template": "You are an expert physician. Analyze the following clinical case and provide:\n1) Differential diagnosis ranked by probability\n2) Imaging and laboratory tests to order\n3) Urgency level and red flags\n4) Initial treatment plan\n\nCLINICAL CASE:\n{case}",
        "cases.case1": "Case 1 — Standard clinical (Chest pain)",
        "cases.case2": "Case 2 — Complex MRI (Neurology)",
        "cases.case3": "Case 3 — Acute pediatric abdomen",
        "cases.case4": "Case 4 — Complex neuropsychiatric",
        "cases.case5": "Case 5 — Real patient (Anonymized template)",
        "vlm.ready": "Image loaded. Local VLM system (LLaVA/Moondream) ready for extraction.",
        "vlm.file_header": "File analyzed: {name} ({size} KB)",
        "browser.note": "Sites opened. Copy the prompt and paste it manually on each site, then paste responses below.",
        "lang.cloud_paste_notice": "Language changed. Cloud responses were not auto-translated — re-paste them if needed.",
        "lang.qvac_rerun_notice": "Language changed. Re-run the benchmark to get a genuine local MedPsy answer in the new language.",
        "matrix.pair": "Pair",
        "matrix.concordance": "Concordance %",
        "cols.model": "Model",
        "cols.version": "Version",
        "cols.tier_requested": "Tier requested",
        "cols.tier": "Tier",
        "cols.local": "Local",
        "cols.kpi": "KPI",
        "cols.kpi_cloud": "Not measurable (web)",
        "cols.kpi_local": "Measured (local)",
        "cols.privacy": "Privacy",
        "cols.reliability": "Reliability %",
        "cols.accuracy_consensus": "Consensus accuracy %",
        "cols.consensus_score": "Consensus score %",
        "cols.primary_accuracy": "Diagnosis (meaning) %",
        "cols.ddx_coverage": "Plan & steps (meaning) %",
        "cols.clinical_score": "Clinical score %",
        "cols.rank": "Rank",
        "cols.final_score": "Final score",
        "cols.consensus_score_short": "Consensus score",
        "cols.clinical_score_short": "Ref.%",
        "cols.clinical_score_rescaled": "Clinical score",
        "cols.diagnosis_match_short": "Dx",
        "cols.plan_next_steps_short": "Plan",
        "cols.clinical_summary_short": "Summary",
        "cols.urgency_agreement_short": "Urgency",
        "cols.consensus_score_rescaled": "Cons.%",
        "cols.consensus_rank": "Rk C",
        "cols.clinical_rank": "Rk Cl",
        "cols.grade_10": "Grade",
        "cols.semantic_gold_short": "Urgency match",
        "cols.clinical_accuracy": "Diagnosis (meaning)",
        "cols.ddx_coverage_short": "Plan & steps",
        "cols.latency": "Latency (s)",
        "cols.token_out": "Tokens out",
        "chart.radar_title": "Multidimensional profile by model",
        "chart.bars_title": "Benchmark ranking",
        "chart.bars_consensus": "Consensus ranking (best = 100%)",
        "chart.bars_clinical": "Clinical match vs reference (absolute %)",
        "chart.bars_final": "Final score",
        "chart.bars_vs": "Consensus vs clinical",
        "chart.dims_title": "KPI dimension breakdown",
        "chart.score_pct": "Score %",
        "chart.acc_cons_short": "Cons. accuracy",
        "chart.privacy_title": "On-device privacy",
        "chart.heatmap_title": "Pairwise diagnostic concordance heatmap",
        "chart.keywords_title": "Shared clinical consensus keywords",
        "chart.keywords_axis": "Models mentioning this term",
        "chart.urgency_title": "Declared triage / urgency level by model",
        "chart.leaderboard_title": "Session leaderboard across cases run",
        # Onboarding
        "onboarding.title": "🩺 How this benchmark works",
        "onboarding.step1_title": "1 · Pick a clinical case",
        "onboarding.step1_body": "Choose one of the preset cases (or write your own) in the card grid below. Each one has a real diagnostic focus — cardio, neuro, pediatric, psych.",
        "onboarding.step2_title": "2 · Choose the analysis depth",
        "onboarding.step2_body": "Light, Medium or Premium — the exact same instruction is sent to ChatGPT, Claude and Gemini so the comparison stays fair.",
        "onboarding.step3_title": "3 · Run QVAC, then paste the cloud answers",
        "onboarding.step3_body": "Tether QVAC MedPsy 4B answers instantly on-device. Open ChatGPT / Claude / Gemini with the links provided, paste the same prompt there manually, then paste each answer back into its card.",
        "onboarding.step4_title": "4 · Read the KPIs & charts",
        "onboarding.step4_body": "Consensus, clinical accuracy vs. a reference diagnosis, triage agreement, privacy score and a live session leaderboard across every case you run.",
        "onboarding.cta": "Got it — let's start 🚀",
        "sidebar.how_it_works": "How it works",
        # Reset confirm
        "reset.confirm_title": "⚠️ Reset everything?",
        "reset.confirm_body": "This clears the current case, pasted answers, KPIs and resets the wallet to 0.00 USDT. **Saved case results in the sidebar are kept.**",
        "reset.confirm_yes": "Yes, reset everything",
        "reset.confirm_cancel": "Cancel",
        # Case section
        "section.case_title": "1 · Choose the clinical case",
        "case.preview_btn": "👁️ Preview",
        "case.load_btn": "Load this case",
        "case.loaded_badge": "✓ Loaded",
        "case.preview_dialog_title": "Case preview",
        "case.focus_label": "Clinical focus",
        "case.specialty.cardio": "Cardiology · Emergency",
        "case.specialty.neuro": "Neurology",
        "case.specialty.pediatric": "Pediatrics · Surgery",
        "case.specialty.psych": "Psychiatry",
        "case.specialty.custom": "Custom / your own case",
        "case.focus.case1": "Recognize STEMI early and catch the sildenafil-nitrate contraindication before treating.",
        "case.focus.case2": "Apply McDonald 2017 criteria and correctly rule out mimics already excluded by labs.",
        "case.focus.case3": "Score classic appendicitis and adapt antibiotic choice to a documented penicillin allergy.",
        "case.focus.case4": "Distinguish a manic switch from substance-induced mania and catch the lithium/renal conflict.",
        "case.focus.case5": "Paste your own anonymized case to benchmark a real scenario.",
        "case.custom_hint": "Custom case — select a slot in the **left sidebar** to load a preset.",
        "case.sidebar_nav_hint": "Select, load or recall cases from the left sidebar.",
        "case.short.case1": "Chest pain",
        "case.short.case2": "MS · MRI",
        "case.short.case3": "Appendicitis",
        "case.short.case4": "Mania",
        "case.short.case5": "Custom",
        # Tier
        "section.tier_title": "2 · Analysis depth (same for every cloud model)",
        # Run
        "section.run_title": "3 · Run the benchmark",
        "run.ready_badge": "Ready to run",
        "status.step_loading": "Loading Tether QVAC MedPsy 4B weights on-device…",
        "status.step_generating": "Generating differential diagnosis locally…",
        "status.step_encrypting": "Computing latency / throughput stats…",
        "status.step_done": "QVAC MedPsy 4B done — 0 bytes left this device.",
        "status.step_error": "Local inference failed",
        "status.live_tokens": "{n} words generated live, on-device",
        "status.live_words": "words",
        # Legacy paste-web path only (OLD/); gold Automated Benchmark uses QVAC sidecar.
        "error.ollama_offline": (
            "⚠️ Legacy helper: local Ollama path unused on the gold Automated Benchmark. "
            "Start the QVAC sidecar (see README) for on-device MedPsy."
        ),
        "error.model_missing": (
            "⚠️ Legacy helper: Ollama medpsy-4b-cpu unused on the gold path. "
            "Download GGUFs via ./install.sh and start the QVAC sidecar."
        ),
        "error.ollama_runtime": "⚠️ Local inference failed: {error}. No fallback answer is shown — this benchmark never fabricates a QVAC result.",
        # Model cards
        "card.instructions_cloud": "Open **{name}** with the Link ↗ above, paste the identical prompt there, then paste its answer into this box.",
        "card.instructions_local": "Generated on-device by QVAC MedPsy 4B; extract/judge still use OpenRouter when scoring.",
        "card.tier_mismatch": "This QVAC answer was generated at **{old}** depth — **{new}** is selected now. Run the benchmark again to regenerate it at the new depth.",
        "card.status_empty": "⏳ Awaiting paste",
        "card.status_filled": "✓ Pasted · {n} words",
        "card.view_full": "🔍 Read full response",
        "card.full_dialog_title": "🔍 Full diagnostic response",
        "card.reasoning_expander": "Real chain-of-thought (as generated by the local model)",
        "card.reasoning_caption": "Unedited reasoning tokens produced by MedPsy-4B before its final answer — shown for transparency, not curated.",
        "card.step1": "1️⃣ Open the link",
        "card.step2": "2️⃣ Paste the prompt there",
        "card.step3": "3️⃣ Paste the answer below",
        "card.steps_inline": "Open the link → paste the prompt → paste the answer below",
        # Compact layout: eyebrows + stepper
        "eyebrow.case": "Clinical case",
        "eyebrow.tier": "QVAC depth & run",
        "eyebrow.run": "Run benchmark",
        "eyebrow.output": "Model responses",
        "eyebrow.results": "Results",
        "stepper.case": "Case",
        "stepper.tier": "Tier",
        "stepper.run": "Run",
        "stepper.results": "Results",
        "case.picker_label": "Pick a clinical case",
        "sidebar.guide_btn": "📖 Guide — how to use this",
        # KPI command center
        "kpi.command_center_title": "📌 Command center — key results at a glance",
        "kpi.best_model": "🏆 Best inter-LLM consensus",
        "kpi.best_clinical": "🎯 Best vs confirmed diagnosis",
        "kpi.best_consensus": "🤝 Best inter-LLM consensus",
        "kpi.privacy_avg": "🔒 Avg. privacy score",
        "kpi.privacy_avg_help": "100% = generation on-device (QVAC). 0% = data sent to a cloud API for every cloud model in the mix. Extract/judge may still use OpenRouter.",
        "kpi.consensus_avg": "🤝 Avg. inter-model agreement",
        "kpi.consensus_avg_help": "Blend of Reliability (whole differential-list overlap with every other model), Accuracy (top-diagnosis-only overlap with every other model) and Semantic similarity (meaning-level match via a local embedding model, so paraphrasing isn't punished). This is an **agreement** score, not a correctness score — see below.",
        "kpi.semantic_unavailable": "ℹ️ Semantic similarity (the meaning-level KPI) needs the local embedding model — start it with the setup script to unlock this third signal; scores above use only Reliability + Accuracy in the meantime.",
        "kpi.glance_title": "📊 Full KPI breakdown — always visible, no clicks needed",
        "kpi.glance_caption": "Consensus tables for this case. Privacy is the compact row under the ranking charts.",
        "kpi.mode_note_consensus": "**No confirmed diagnosis was provided for this round** → scores below are an **agreement score (0-100%)**: how much the models' answers overlap with each other in wording and meaning. Without a ground truth, this measures *consensus*, not *correctness* — 100% would mean word-for-word, meaning-for-meaning identical answers, which is rare even when everyone is clinically right. Paste a confirmed diagnosis in the section above to unlock the 1-10 **clinical grade**.",
        "kpi.mode_note_gold": "**Confirmed reference diagnosis active** → clinical ranking is **absolute** (not rescaled to 100%). Score = semantic match on diagnosis, plan, urgency and summary vs your reference — paraphrases in another language score high; word-for-word copy is **not** required.",
        "kpi.explain_button": "🔍 See exactly how every score was calculated",
        "explain.dialog_title": "🔍 How every score was calculated",
        "explain.dialog_intro": (
            "No hidden numbers: each dimension is a continuous 0–100% from local semantic embeddings "
            "(cosine similarity on extracted clinical text). The displayed score is the **literal weighted average** "
            "shown below — Cons.% is then rescaled so #1 = 100%."
        ),
        "explain.consensus_label": "Consensus score (Cons.%)",
        "explain.consensus_desc": (
            "Four clinical slices compared pairwise vs every other model: diagnosis block, plan/next steps, "
            "clinical summary, and urgency (65% semantic text + 35% triage label). "
            "Each value is a continuous 0–100% — not fixed steps like 100/75/50/0."
        ),
        "explain.consensus_rescale_note": (
            "Cons_raw is the weighted average above. Cons.% in the table rescales vs the leader this round "
            "(#1 → 100%) — Ref.% in gold mode is **not** rescaled."
        ),
        "explain.dimension_pairs": "{dim} — pairwise similarity vs other models",
        "explain.reliability_label": "① Reliability — pairwise agreement",
        "explain.reliability_desc": "Average overlap between this model's differential diagnosis and each other model's, blending list-level overlap (65%) with whole-text keyword overlap (35%).",
        "explain.accuracy_label": "② Accuracy (consensus) — keyword match",
        "explain.accuracy_desc": "Average keyword overlap between this model's primary (top) diagnosis and each other model's primary diagnosis — same idea as Reliability, but focused only on the single top pick instead of the whole differential list.",
        "explain.semantic_label": "③ Semantic similarity — meaning match",
        "explain.semantic_desc": "Cosine similarity between this model's primary diagnosis and each other model's, computed by a local embedding model (all-minilm) — this is what catches \"same diagnosis, different words\".",
        "explain.semantic_unavailable_note": (
            "Local embedding model unreachable this round — consensus falls back to keyword reliability/accuracy blend."
        ),
        "explain.gold_label": "④ Vs. confirmed reference diagnosis",
        "explain.gold_desc": "Measured against your confirmed reference using **meaning**, not word overlap: diagnosis (40%), plan & next steps (30%), urgency (20%), overall clinical summary (10%). Same diagnosis in different words scores high.",
        "explain.grade_label": "Clinical grade",
        "explain.grade_rubric": "Rescaled 0-100% clinical score → familiar 1-10 grade: **1** = unrelated to the reference · **5** ≈ partially right (same category, different specific diagnosis) · **10** = matches the reference almost exactly in wording and meaning, with the full differential covered.",
        "explain.formula_label": "Formula",
        "explain.formula_consensus_dim": "Cons_raw = {formula}\n= {raw}%\nCons_display = {rescaled}% (rescaled vs #1 this round)",
        "explain.formula_gold_dim": "Ref = {formula}\n= {score}% (absolute — not rescaled)",
        "explain.own_keywords": "This model's primary-diagnosis keywords",
        "explain.consensus_keywords": "Majority-support keywords across all models",
        "explain.matched_keywords": "Keywords this model shares with the majority",
        "explain.no_keywords": "No keywords extracted for this comparison.",
        "explain.urgency_label": "Declared urgency/triage",
        "explain.privacy_label": "Privacy",
        "explain.vs": "vs. {name}",
        "explain.select_model": "Model",
        "kpi.triage_agreement_help": "% of models that assigned the same urgency/triage label to this case.",
        "kpi.time_saved": "⏱️ Est. time saved by QVAC",
        "kpi.time_saved_help": "Illustrative estimate only: assumes ~{overhead:.0f}s of manual overhead per cloud consult (open site, copy/paste, wait) vs. QVAC's measured on-device latency.",
        "kpi.models_active": "🧪 Models compared",
        "kpi.triage": "Triage",
        "kpi.triage_caption": "Compares the urgency/priority level each model assigned to the same case — critical for patient safety.",
        "kpi.triage_agreement": "🚦 Triage agreement",
        "kpi.session": "Session report",
        "urgency.critical": "🔴 Critical / emergency",
        "urgency.high": "🟠 High",
        "urgency.moderate": "🟡 Moderate",
        "urgency.low": "🟢 Low",
        "urgency.unknown": "⚪ Not stated",
        "cols.urgency": "Declared urgency",
        "cols.urgency_score": "Urgency score",
        "leaderboard.title": "📈 Session leaderboard",
        "leaderboard.caption": "Add each case you benchmark to build a running leaderboard — perfect for a multi-case demo recording.",
        "leaderboard.empty": "No rounds added yet. Run a case, review it, then click “Add this round”.",
        "leaderboard.add_round": "➕ Add this round to the session leaderboard",
        "leaderboard.added_toast": "Round added to the session leaderboard 📈",
        "leaderboard.wins": "Rounds won",
        "leaderboard.avg_score": "Avg. consensus %",
        "leaderboard.rounds": "Rounds played",
        "leaderboard.clear": "🗑️ Clear session leaderboard",
        "leaderboard.history_expander": "Round-by-round history",
        "session.caption": "Save each reviewed case to include it in the final averaged ranking.",
        "session.save_round": "Save this case ranking",
        "session.save_hint": "**{case}** saved · **{n}** case(s) in session — open **Session report** tab for the finale.",
        "session.save_hint_empty": "Save **{case}** when ready, then move to the next clinical case.",
        "session.saved_cases": "Saved cases this session",
        "session.col_round": "#",
        "session.col_case": "Case",
        "session.col_mode": "Mode",
        "session.col_best_consensus": "Best consensus",
        "session.col_best_clinical": "Best vs reference",
        "session.col_saved_at": "Saved at",
        "session.mode_consensus": "Consensus",
        "session.mode_gold": "Confirmed diagnosis",
        "session.finale_title": "Final averaged ranking",
        "session.finale_caption": "Average across standard cases (1–4) and clinical match on case 5 with confirmed diagnosis.",
        "session.finale_avg": "Average ranking (cases 1–4)",
        "session.finale_avg_help": "Arithmetic mean of consensus scores across **{n}** saved standard case(s). #1 is shown as **100%**.",
        "session.finale_avg_empty": "Save at least one standard case (without confirmed diagnosis) to build the average ranking.",
        "session.finale_gold": "Real diagnosis ranking (case 5)",
        "session.finale_gold_help": "Absolute match vs confirmed diagnosis — **{case}**.",
        "session.finale_gold_avg_help": "Clinical scores are the **rolling average** of the last **{n}** saved runs of case 5 (max {max}).",
        "session.finale_gold_empty": "Save a case with **confirmed diagnosis** enabled to show the clinical finale.",
        "session.slots_empty": "No saved cases yet — run a benchmark and use **Save current benchmark** in the sidebar.",
        "session.save_not_auto": "Saving is **manual** — use the **Save** button in the results area above the ranking chart.",
        "session.save_ready": "Results ready for **{case}** — click **Save current benchmark** below.",
        "session.save_ready_short": "Not saved yet — click **Save** to store this ranking in the sidebar slot for **{case}**.",
        "session.save_case5_hint": "Select **case 5** in the case picker (top bar), then compile the template — saving requires an active case slot.",
        "session.save_case5_ready": "Ready to save **run {n}/{max}** — re-run benchmark between saves.",
        "session.save_case5_ready_rolling": "Ready to save another run — replaces the oldest; finale keeps the last **{max}**.",
        "session.save_case5_ready_short": "Save **run {n}/{max}**, then recall slot 5 to see the rolling average.",
        "session.save_case5_ready_short_rolling": "After **{max}** runs, each new save replaces the oldest. Recall slot 5 for the updated average.",
        "charts.tab_privacy": "Privacy gauges",
        "charts.tab_heatmap": "Concordance heatmap",
        "charts.tab_keywords": "Consensus keywords",
        "charts.tab_urgency": "Triage comparison",
        # Decision dialogs
        "decision.cloud_dialog_title": "🔴 Cloud path selected",
        "decision.cloud_dialog_body": "Sending this clinical case to a public cloud LLM means the data leaves this device and this jurisdiction. Under GDPR/HIPAA this typically requires a signed DPA, patient consent and an audit trail — none of which exist in a copy-pasted chat.",
        "decision.qvac_dialog_title": "🟢 Local QVAC path selected",
        "decision.qvac_dialog_body": "Tether QVAC MedPsy 4B generated on this device. Extract/judge steps may still use OpenRouter when scoring is enabled.",
        "decision.sell_dialog_title": "💠 Simulated anonymized data sale",
        "decision.sell_dialog_body": "In a real deployment, an anonymized, aggregated benchmark record could be sold to research consortia with the patient's consent, rewarding them in USDT. This wallet simulates that reward flow.",
        "decision.sell_question": "Sell your anonymized clinical record to research consortia for **{amount:.2f} USDT**?",
        "decision.sell_confirm": "✅ Yes, sell for {amount:.2f} USDT",
        "decision.sell_cancel": "❌ Cancel",
        "decision.sell_success_title": "Payment received",
        "decision.sell_success_body": "+{amount:.2f} USDT credited to your wallet.",
        "decision.sell_new_balance": "New balance",
        "decision.close": "Close",
        # Automated Benchmark · case-family history resume
        "bench.family_found": (
            "Found {n} prior run(s) across {cohorts} confirm version(s) for this "
            "case+reference — restore a saved confirmed reference to reopen that cohort."
        ),
        "bench.family_select": "Prior confirm versions",
        "bench.family_restore_btn": "Restore last confirmed reference",
        "bench.family_restore_help": (
            "Loads the exact frozen gold contract from History (same claim splits). "
            "Does not merge different Confirm versions. "
            "Re-Prepare or claim-split edits can split cohort_id — restore exact Confirm."
        ),
        "bench.family_restore_ok": (
            "Confirmed reference restored — candidates unlocked; Rebuild uses this cohort."
        ),
        "bench.family_restore_fail": "Restore failed: {err}",
        "bench.family_other_cohorts": (
            "Other confirm versions of this case exist ({n} runs) — restore that "
            "reference to pool them. Different gold contracts are never auto-merged. "
            "Re-Prepare or claim-split edits start a new cohort."
        ),
        "bench.rebuild_sec_label": "Rebuild mean across N runs · $0 API",
        "bench.rebuild_scope_label": "Mean scope",
        "bench.rebuild_scope_same": "Selected case only",
        "bench.rebuild_scope_portfolio": (
            "All cases (portfolio · mixed-case exploratory)"
        ),
        "bench.rebuild_n_label": "Average over ≤N scored / model",
        "bench.rebuild_btn": "Rebuild mean · ≤{n} scored / model · open KPI popup · $0",
        "bench.rebuild_btn_help_same": (
            "Offline mean for the selected Case slot's immutable cohort only "
            "(same clinical stem + confirmed gold contract). N = max **scored** "
            "observations per model (newest first); technical N/A skipped, "
            "older scored used. Zero API cost."
        ),
        "bench.rebuild_btn_help_portfolio": (
            "Offline exploratory mean across all Case slots that have History "
            "for this API key: for each roster model, average its own newest "
            "≤N **scored** observations (same track + scoring_version; N/A "
            "skipped, older scored used). "
            "Mixed-case / mixed-gold — not clinical validation. Zero API cost."
        ),
        "bench.rebuild_portfolio_intro": (
            "**All cases (portfolio)** — exploratory mean across every Case "
            "slot with data for this key. **≤N scored obs/model**; N/A skipped, "
            "older scored used. Same track + scoring_version. Gold contracts are "
            "**not** merged; scores stay reference-relative per run. "
            "**No API calls.**"
        ),
        "bench.rebuild_portfolio_stats": (
            "**≤{n} scored/model · {cases} distinct case stems** (eligible runs: "
            "{avail}) · track `{track}` · mixed-case exploratory · "
            "N/A skipped, older scored used"
        ),
        "bench.rebuild_portfolio_quiet": (
            "Scores stay reference-relative per case; portfolio = exploratory "
            "mixed-case mean (≤N scored obs/model; N/A skipped, older scored "
            "used), not clinical validation. "
            "Mixed rosters OK. Incompatible scoring versions are never auto-merged."
        ),
        "bench.rebuild_portfolio_success": (
            "**All cases (portfolio)** · **≤N/model · {n} run docs · "
            "{cases} case stems** · mixed-case exploratory · "
            "mixed rosters OK · **$0 API**"
        ),
        "bench.rebuild_portfolio_caption": (
            "Each run's Clinical Composite is still relative to that case's "
            "confirmed gold. Averaging across Case 1–5 slots is exploratory "
            "mixed-case analysis — not a clinical validation claim. "
            "≤N scored obs/model; N/A skipped, older scored used. Failed % = "
            "technical failures encountered while filling that scored quota."
        ),
        "bench.rebuild_portfolio_mean_rank": "Mean rank (lower better): {ranks}",
        "bench.rebuild_need_portfolio": (
            "Need at least **1** portfolio-eligible complete/partial run "
            "(found {n}). Same track + scoring_version required; "
            "roster shapes may differ."
        ),
        "bench.rebuild_reopen_portfolio": (
            "Re-open all-cases portfolio mean · ≤N/model · {n} docs · "
            "{cases} stems · $0"
        ),
        "bench.case_slots_label": "Clinical cases",
        "bench.case_slot_btn": "Case {n}",
        "bench.case_slot_empty": "Empty",
        "bench.case_slot_runs": "{n} runs",
        "bench.new_case_btn": "New case",
        "bench.new_case_help": (
            "Open the next empty Case slot, or create Case 6, 7, … when "
            "Case 1–5 are filled. Paste stem + reference, then Prepare / Confirm."
        ),
        "bench.new_case_full": (
            "Reached the soft limit of {n} Case slots for this key. Select an "
            "existing slot to edit or rebuild means."
        ),
        "bench.case_slot_loaded": (
            "Loaded **Case {n}** stem + confirmed reference from History."
        ),
        "bench.case_slot_ready_empty": (
            "**Case {n}** is empty — paste an anonymized case and reference, "
            "then Prepare / Confirm."
        ),
    },
    "it": {
        "page_title": "QVAC vs Cloud LLMs - Health Test",
        "lang_label": "Lingua",
        "title": "QVAC vs Cloud LLMs - Health Test",
        "subtitle": "Prompt identico per tutti · Profondità scelta solo per QVAC · Cloud manuali · Confronto diagnosi",
        "cloud.demo_banner": "☁️ **Demo pubblica (hosting gratuito)** — Incolla cloud + ranking funzionano qui. **Run benchmark QVAC live** richiede Mac/PC locale con sidecar QVAC (`./install.sh` + `sidecar`). Puoi incollare nel riquadro QVAC una risposta generata in locale.",
        "cloud.demo_link": "Setup locale completo (gratis): [README GitHub](https://github.com/FrancescoAloe91/qvac-vs-cloud-llms-health-test) · [Demo live](https://francescoaloe91-qvac-vs-cloud-llms-health-test-app-wihxyd.streamlit.app)",
        "sidebar.reset": "Reset",
        "sidebar.reset_help": "Azzera tutto incluso wallet a 0 USDT",
        "sidebar.cloud_tiers": "Versioni modelli cloud",
        "sidebar.cloud_tiers_help": "Visibili su schede e grafici. Su Claude vedi modello + thinking nella barra in alto; su ChatGPT free spesso il modello non è indicato — annota cosa vedi prima di incollare.",
        "sidebar.cloud_tiers_howto": "**Come verificare:** Claude → nome modello + thinking (es. Sonnet · Extra) nella barra sopra la chat. ChatGPT → tier free spesso senza nome modello; opzionale: chiedi «Quale modello sei?» una volta. Gemini → selettore modello in basso a sinistra se presente.",
        "sidebar.cloud_tiers_save": "Salva etichette versione",
        "sidebar.cloud_tiers_saved": "Etichette versione cloud salvate",
        "sidebar.cloud_tiers_qvac_auto": "Rilevato automaticamente dal setup locale (quant GGUF + modello sidecar QVAC).",
        "sidebar.privacy_note": "Generazione QVAC on-device; extract/judge usano comunque OpenRouter in scoring.",
        "sidebar.wallet": "Wallet",
        "sidebar.wallet_caption": "Reward USDT simulate — vendi record anonimizzati in fondo alla pagina.",
        "sidebar.test_case": "Caso di test",
        "sidebar.scenario": "Scenario",
        "sidebar.load_case": "Carica caso",
        "sidebar.vlm": "VLM",
        "sidebar.vlm_upload": "Referto / RMN",
        "sidebar.vlm_extract": "Estrai testo",
        "sidebar.vlm_add": "Aggiungi al caso",
        "sidebar.saved_cases": "Risultati salvati per caso",
        "sidebar.saved_cases_help": "Carica o richiama i benchmark salvati. Verde = pronto da richiamare.",
        "sidebar.save_slot": "Salva benchmark corrente",
        "sidebar.save_slot_help": "Memorizza grafici + tabella del caso attivo (sovrascrive se salvi di nuovo).",
        "sidebar.save_slot_toast": "**{case}** salvato — clicca lo slot quando vuoi richiamarlo.",
        "sidebar.save_case5_run": "💾 Salva run {n}/{max}",
        "sidebar.save_case5_run_rolling": "💾 Salva nuova run (ultime {max})",
        "sidebar.save_case5_help": "Il caso 5 conserva solo le **ultime {max}** run; le più vecchie vengono scartate. Gli score clinici usano la media rolling su quelle run.",
        "sidebar.save_case5_run_toast": "Run **{n}/{max}** salvata — il finale usa la media rolling delle ultime {max} run.",
        "sidebar.save_case5_run_toast_rolling": "Run salvata — la più vecchia è stata sostituita; media sulle ultime **{max}** run.",
        "sidebar.case5_runs": "{n}/{max} run conservate · media rolling nel finale",
        "sidebar.viewing_case5_avg": "media rolling ultime {n} run",
        "sidebar.slot_badge_empty": "Vuoto",
        "sidebar.slot_badge_filled": "Salvato",
        "sidebar.slot_badge_viewing": "In vista",
        "sidebar.slot_empty_hint": "Nessun benchmark salvato",
        "sidebar.slot_saved_at": "Salvato alle {time}",
        "sidebar.slot_load": "Carica",
        "sidebar.slot_recall": "Richiama",
        "sidebar.slot_viewing": "In vista",
        "sidebar.saved_count_inline": "{n}/{total}",
        "sidebar.finale_btn": "RANKING DEFINITIVO MEDIATO",
        "sidebar.finale_title": "Ranking definitivo mediato",
        "sidebar.clear_saved": "Reset risultati salvati",
        "sidebar.clear_saved_help": "Cancella tutti gli snapshot salvati (casi 1–5). Non si può annullare.",
        "sidebar.clear_saved_title": "Cancellare tutti i risultati salvati?",
        "sidebar.clear_saved_body": "Rimuove ogni snapshot (casi 1–5). Il benchmark corrente e il wallet **non** vengono toccati.",
        "sidebar.clear_saved_confirm": "Sì, cancella salvati",
        "sidebar.viewing_saved": "📂 Risultato salvato — **{case}**",
        "sidebar.return_live": "Torna al live",
        "sidebar.saved_count": "**{n}/{total}** slot caso compilati",
        "case.clinical": "Caso clinico",
        "tier.section": "Profondità di analisi di QVAC",
        "qvac.instruction": (
            "Fornisci una valutazione clinica equilibrata e completa: diagnosi differenziale ordinata, "
            "esami consigliati, livello di urgenza e red flags, e piano di gestione iniziale."
        ),
        "badge.local": "On-device",
        "eyebrow.run": "Esegui benchmark",
        "run.caption": "QVAC gira in locale con l'istruzione clinica standard di MedPsy. Il prompt sotto è identico per ogni sito cloud.",
        "prompt.expander": "Prompt clinico (stesso testo per i siti cloud; QVAC aggiunge l'istruzione in automatico)",
        "prompt.copy": "Copia prompt negli appunti",
        "prompt.copied": "Prompt copiato. Incollalo manualmente sui siti cloud.",
        "browser.open": "Apri ChatGPT, Claude e Gemini nel browser",
        "browser.open_btn": "LLM cloud (ChatGPT, Claude, Gemini)",
        "browser.help": "Apre ChatGPT, Claude e Gemini in 3 tab. Copia e incolla il prompt manualmente su ogni sito.",
        "caption.cloud": "I siti cloud non possono essere incorporati nella dashboard (sicurezza). Incolla le risposte sotto. Il selettore di profondità cambia solo il ragionamento di QVAC MedPsy in locale — ChatGPT, Claude e Gemini ricevono sempre lo stesso identico prompt, così il confronto resta corretto. ⚠️ Se un sito cloud risponde in una lingua diversa dagli altri, i KPI basati su parole condivise (Affidabilità, Accuratezza) risulteranno onestamente più bassi anche a fronte della stessa diagnosi — per il confronto più corretto, chiedi a ciascun sito di rispondere nella stessa lingua del caso (es. aggiungi \"Rispondi in italiano\").",
        "benchmark.run": "Esegui benchmark",
        "benchmark.spinner": "Inferenza QVAC locale in corso...",
        "browser.opened": "Aperti",
        "gold.section": "Diagnosi di riferimento (opzionale)",
        "gold.use": "Usa diagnosi clinica certa come riferimento",
        "gold.help": "Se attivo, il ranking misura lo scostamento da questa diagnosi. Se disattivo, si usa il consenso tra i modelli.",
        "gold.caption": "Puoi usare la tua patologia reale in forma **anonimizzata** (niente nome, date, ospedale, città). Esempio: «Diagnosi confermata: sclerosi multipla recidivante-remittente» oppure incolla un referto depersonalizzato.",
        "gold.qvac_variance": "QVAC è un 4B on-device via **sidecar QVAC** (`@qvac/sdk`). Il track controlled usa sampling esplicito; native-default lascia i default del modello. Salva fino a **{max} run** — il finale usa la media rolling per uno score clinico stabile.",
        "gold.input": "Diagnosi corretta / referto di riferimento",
        "gold.placeholder": "Es: Diagnosi confermata: sindrome coronarica acuta (NSTEMI). Esami: troponina positiva, ECG con sopraslivellamento ST...",
        "output.section": "Output diagnostici — incolla qui le risposte dai siti cloud",
        "output.link": "Link ↗",
        "output.placeholder_cloud": "Incolla qui la risposta dal sito ufficiale...",
        "output.placeholder_qvac": "Generato da QVAC MedPsy locale al benchmark",
        "output.diagnosis": "Diagnosi",
        "output.stats": "TTFT {ttft}s · TPS {tps} · Latenza {latency}s · RAM {ram}GB",
        "recalc": "Ricalcola confronto diagnostico",
        "warn.no_output": "Esegui il benchmark QVAC e incolla le risposte cloud, poi ricalcola.",
        "kpi.performance": "KPI performance",
        "kpi.performance_caption": "Solo QVAC misura TTFT/TPS in locale. I cloud non sono misurabili dai siti web.",
        "kpi.consensus": "KPI consenso tra modelli",
        "kpi.consensus_caption": "Misura quanto i modelli concordano tra loro sullo stesso prompt (incluso QVAC MedPsy).",
        "kpi.primary_agreement": "Accordo diagnosi primaria",
        "kpi.qvac_cloud": "QVAC ↔ Cloud",
        "kpi.models_compared": "Modelli confrontati",
        "kpi.consensus_keywords": "Keyword consenso",
        "kpi.matrix_expander": "Matrice concordanza tra coppie di modelli",
        "kpi.gold_section": "KPI vs diagnosi di riferimento clinica",
        "kpi.gold_caption": "Confronto con la diagnosi certa inserita (anonimizzata). I punteggi riflettono il **significato clinico** — diagnosi, piano/prossimi step, urgenza — non le parole identiche. Usa embedding semantici locali (stesso motore per tutti i modelli).",
        "kpi.gold_diagnoses": "Diagnosi estratte dal riferimento",
        "kpi.gold_keywords": "Keyword riferimento",
        "ranking.final": "Classifica finale",
        "ranking.section": "Classifica e punteggio",
        "ranking.dual_explanation": (
            "Due classifiche **indipendenti** — non vengono mai mediate. "
            "**Consenso** (#1 = 100%): chi concorda di più con gli altri LLM su questo caso. "
            "**Clinico** (#1 = miglior match): chi è più vicino alla **diagnosi certa** (100% = match perfetto)."
        ),
        "ranking.dual_summary": (
            "🥇 **Migliore vs diagnosi certa:** {clin_name} ({clin_score}%) · "
            "🥇 **Migliore consenso tra LLM:** {cons_name} ({cons_score}%)"
        ),
        "ranking.clinical_chart_note": (
            "Il grafico clinico mostra il match **assoluto** con la diagnosi certa — "
            "100% solo se il modello coincide perfettamente col riferimento, non per riscalatura relativa."
        ),
        "ranking.score_story_title": "Cosa misura lo 0–100% (demo)",
        "ranking.score_story_gold_intro": (
            "Hai attivato una **diagnosi di riferimento certa** — la dashboard mostra **due score indipendenti**. "
            "Entrambi usano **embedding semantici locali** (significato clinico, non copia parola per parola)."
        ),
        "ranking.score_story_gold_clinical": (
            "**Ref.% / grafico clinico (0–100 assoluto)** — quanto ogni risposta è vicina al **tuo riferimento reale**: "
            "**40%** diagnosi & DDx · **30%** piano & next steps · **20%** urgenza/triage · **10%** sintesi clinica. "
            "**100%** = stesso significato clinico del riferimento; **0%** = non correlato."
        ),
        "ranking.score_story_gold_consensus": (
            "**Cons.% / grafico consenso (relativo)** — chi concorda di più con gli **altri LLM** su questo caso: "
            "**40%** diagnosi · **30%** piano · **20%** urgenza · **10%** sintesi — poi **#1 riscalato a 100%**. "
            "Misura accordo di gruppo, non correttezza vs verità clinica."
        ),
        "ranking.score_story_consensus_intro": (
            "**Nessuna diagnosi certa in questo round** — gli score misurano l'**accordo tra modelli**, non la correttezza medica."
        ),
        "ranking.score_story_consensus_formula": (
            "**Cons.% (0–100)** = **media pesata reale** di quattro dimensioni continue (0–100 ciascuna, da embedding locali): "
            "**40%** diagnosi · **30%** piano & next steps · **20%** urgenza · **10%** sintesi clinica — "
            "poi **#1 del gruppo = 100%** (riscalatura relativa solo su Cons.%). "
            "Attiva la **diagnosi certa** sopra per il match **assoluto** vs il tuo caso reale. "
            "Usa **🔍 Guarda esattamente come è stato calcolato ogni punteggio** per verificare i numeri."
        ),
        "chart.privacy_scale_low": "0",
        "chart.privacy_scale_high": "100",
        "chart.privacy_scale_cloud": "cloud",
        "chart.privacy_scale_local": "on-device",
        "chart.privacy_caption": "0% = risposta incollata da sito cloud (il caso ha lasciato il device) · 100% = generazione QVAC MedPsy restata on-device (extract/judge possono usare OpenRouter).",
        "ranking.legend": (
            "**Score consenso %** = media pesata di match diagnosi (40%), piano & next steps (30%), "
            "accordo urgenza (20%), sintesi clinica (10%) — **riscalato così #1 del gruppo = 100%**. "
            "Con diagnosi certa, **Ref.%** e il grafico clinico sono **assoluti** "
            "(100% = match perfetto col riferimento — mai riscalati). "
            "**Cons.%** da solo è riscalato (#1 = 100%) perché non c'è ground truth. "
            "La riga **Privacy** (0–100) è solo informativa — non entra in nessuno score clinico o consenso."
        ),
        "narrative.section": "Perché questa classifica?",
        "narrative.section_caption": "Sintesi breve per modello.",
        "narrative.compact_leader": "Leader del gruppo — **{score}%** consenso.",
        "narrative.compact_leader_clinical": "Più vicino al riferimento — **{score}%**.",
        "narrative.compact_laggard": "Più basso qui — **#{rank}** con **{score}%**.",
        "narrative.compact_mid": "Rank **#{rank}** — **{score}%**.",
        "narrative.compact_dual": "Consenso **#{rank_cons}** ({score_cons}%) · Clinico **#{rank_clin}** ({score_clin}%).",
        "narrative.compact_urgency_diff": "Urgenza **{label}** (gruppo: **{majority}**).",
        "narrative.verdict_high": "Score consenso **{score}/100** — forte accordo col resto del gruppo, sia nelle parole che nel significato clinico{sem_clause}.",
        "narrative.verdict_mid": "Score consenso **{score}/100** — accordo discreto col gruppo, con qualche differenza di formulazione o enfasi{sem_clause}.",
        "narrative.verdict_low": "Score consenso **{score}/100** — questa risposta si allontana di più dal gruppo su questo caso{sem_clause}.",
        "narrative.dual_ranks": "**Rank consenso #{rank_cons}** ({score_cons}%) · **Rank clinico #{rank_clin}** ({score_clin}%) — sono indipendenti.",
        "narrative.clinical_score": "Score clinico vs diagnosi certa: **{score}/100** (rank **#{rank}**). 100% = match perfetto col tuo riferimento.",
        "narrative.top_consensus": "**Leader consenso:** miglior accordo inter-LLM in questo round.",
        "narrative.bottom_consensus": "**Nota consenso:** accordo più basso con gli altri modelli su questo caso.",
        "narrative.top_clinical": "**Leader clinico:** match più vicino alla diagnosi certa.",
        "narrative.bottom_clinical": "**Nota clinica:** più lontano dalla diagnosi certa su questo caso.",
        "narrative.sem_clause": " (similarità di significato con gli altri: **{sem}%**)",
        "narrative.concordance": "Coincide con il **{rel}%** delle diagnosi differenziali degli altri modelli (affidabilità) e con il **{acc}%** delle keyword di consenso tra i modelli (accuratezza).",
        "narrative.gold_accuracy": "Coincide con la diagnosi di riferimento confermata con un'accuratezza keyword del **{acc}%** e copre il **{cov}%** della sua diagnosi differenziale{sem_clause}.",
        "narrative.richness": "Ha estratto **{n}** voci di diagnosi differenziale e **{kw}** keyword cliniche distinte dalla sua risposta.",
        "narrative.urgency_match": "Ha classificato il caso come urgenza **{label}**, in linea con la maggioranza dei modelli.",
        "narrative.urgency_diff": "Ha classificato il caso come urgenza **{label}**, mentre la maggioranza dei modelli si è orientata su **{majority}**.",
        "narrative.urgency_none": "Non è stato possibile individuare un livello di urgenza/triage esplicito nella sua risposta.",
        "narrative.privacy_cloud": "Ha richiesto l'invio del caso clinico ai server cloud di {vendor} per funzionare — 0% privacy on-device.",
        "narrative.privacy_local": "Generazione QVAC on-device; extract/judge escono via OpenRouter se usati.",
        "narrative.top": "**Punto forte:** il miglior equilibrio complessivo tra allineamento clinico e completezza in questo round.",
        "narrative.bottom": "**Nota:** allineamento più basso col gruppo su questo caso — può significare una diagnosi differenziale realmente diversa (magari comunque valida), oppure semplicemente una risposta incollata meno strutturata, non necessariamente una qualità clinica inferiore.",
        "charts.section": "Visualizzazione benchmark",
        "charts.tab_radar": "Radar multidimensionale",
        "charts.tab_bars": "Classifica a barre",
        "charts.tab_dims": "Dettaglio dimensioni",
        "charts.dims_privacy_note": "Il breakdown per dimensione è in questa tab — la privacy è già mostrata in modo compatto sopra i grafici di ranking.",
        "decision.section": "Decisione e conformità",
        "decision.lead": "Ora che hai visto i risultati fianco a fianco — quale percorso scegli davvero per questo caso?",
        "decision.cloud": "Cloud",
        "decision.cloud_err": "Violazione HIPAA/GDPR. Leak 100%.",
        "decision.cloud_caption": "Il dato lascia il dispositivo — nessun DPA, nessuna traccia di consenso.",
        "decision.qvac": "QVAC locale",
        "decision.qvac_ok": "Report cifrato. Generazione restata on-device.",
        "decision.qvac_caption": "Generazione QVAC on-device; extract/judge possono ancora usare OpenRouter.",
        "decision.sell": "Vendi anonimizzato",
        "decision.sell_caption": "Simula una vendita a pagamento e anonimizzata dei dati.",
        "tier.light.desc": "Risposta rapida e sintetica — 3 ipotesi principali, urgenza, 2 esami chiave.",
        "tier.medium.desc": "Valutazione equilibrata — DDx completa, esami, red flags, piano iniziale.",
        "tier.premium.desc": "Valutazione specialistica esaustiva — scoring, stratificazione rischio, follow-up.",
        "tier.instruction.light": "Rispondi in modo conciso (max 200 parole). Elenca solo le 3 ipotesi diagnostiche principali, 2 esami chiave e il livello di urgenza.",
        "tier.instruction.medium": "Fornisci una valutazione equilibrata: diagnosi differenziale ordinata, esami da richiedere, red flags e piano terapeutico iniziale.",
        "tier.instruction.premium": "Fornisci una valutazione specialistica esaustiva: diagnosi differenziale completa con scoring di probabilità, esami prioritizzati, stratificazione del rischio, piano terapeutico dettagliato e follow-up.",
        "prompt.template": "Sei un medico esperto. Analizza il seguente caso clinico e fornisci:\n1) Diagnosi differenziale ordinata per probabilità\n2) Esami strumentali e di laboratorio da richiedere\n3) Livello di urgenza e red flags\n4) Piano terapeutico iniziale\n\nCASO CLINICO:\n{case}",
        "cases.case1": "Caso 1 - Clinico standard (Dolore toracico)",
        "cases.case2": "Caso 2 - RMN complesso (Neurologico)",
        "cases.case3": "Caso 3 - Addome acuto pediatrico",
        "cases.case4": "Caso 4 - Neuropsichiatrico complesso",
        "cases.case5": "Caso 5 - Paziente reale (Anonimizzato)",
        "vlm.ready": "Immagine caricata. Sistema VLM (LLaVA/Moondream) pronto per l'estrazione locale.",
        "vlm.file_header": "File analizzato: {name} ({size} KB)",
        "browser.note": "Siti aperti. Copia il prompt e incollalo manualmente su ogni sito, poi incolla le risposte sotto.",
        "lang.cloud_paste_notice": "Lingua cambiata. Le risposte cloud non sono state tradotte automaticamente — incollale di nuovo se necessario.",
        "lang.qvac_rerun_notice": "Lingua cambiata. Riesegui il benchmark per ottenere una risposta locale reale di MedPsy nella nuova lingua.",
        "matrix.pair": "Coppia",
        "matrix.concordance": "Concordanza %",
        "cols.model": "Modello",
        "cols.version": "Versione",
        "cols.tier_requested": "Tier richiesto",
        "cols.tier": "Tier",
        "cols.local": "Local",
        "cols.kpi": "KPI",
        "cols.kpi_cloud": "Non misurabile (sito web)",
        "cols.kpi_local": "Misurato (locale)",
        "cols.privacy": "Privacy",
        "cols.reliability": "Affidabilità %",
        "cols.accuracy_consensus": "Accuratezza consenso %",
        "cols.consensus_score": "Score consenso %",
        "cols.primary_accuracy": "Diagnosi (significato) %",
        "cols.ddx_coverage": "Piano & step (significato) %",
        "cols.clinical_score": "Score clinico %",
        "cols.rank": "Classifica",
        "cols.final_score": "Score finale",
        "cols.consensus_score_short": "Score consenso",
        "cols.clinical_score_short": "Ref.%",
        "cols.clinical_score_rescaled": "Score clinico",
        "cols.diagnosis_match_short": "Dx",
        "cols.plan_next_steps_short": "Piano",
        "cols.clinical_summary_short": "Sintesi",
        "cols.urgency_agreement_short": "Urgenza",
        "cols.consensus_score_rescaled": "Cons.%",
        "cols.consensus_rank": "Rk C",
        "cols.clinical_rank": "Rk Cl",
        "cols.grade_10": "Voto",
        "cols.semantic_gold_short": "Match urgenza",
        "cols.clinical_accuracy": "Diagnosi (significato)",
        "cols.ddx_coverage_short": "Piano & step",
        "cols.latency": "Latenza (s)",
        "cols.token_out": "Token out",
        "chart.radar_title": "Profilo multidimensionale per modello",
        "chart.bars_title": "Classifica benchmark",
        "chart.bars_consensus": "Classifica consenso (migliore = 100%)",
        "chart.bars_clinical": "Match clinico vs riferimento (assoluto %)",
        "chart.bars_final": "Score finale",
        "chart.bars_vs": "Consenso vs clinico",
        "chart.dims_title": "Dettaglio per dimensione KPI",
        "chart.score_pct": "Punteggio %",
        "chart.acc_cons_short": "Acc. consenso",
        "chart.privacy_title": "Privacy on-device",
        "chart.heatmap_title": "Heatmap concordanza diagnostica a coppie",
        "chart.keywords_title": "Keyword cliniche di consenso condivise",
        "chart.keywords_axis": "Modelli che citano il termine",
        "chart.urgency_title": "Livello di urgenza/triage dichiarato per modello",
        "chart.leaderboard_title": "Classifica sessione sui casi eseguiti",
        # Onboarding
        "onboarding.title": "🩺 Come funziona questo benchmark",
        "onboarding.step1_title": "1 · Scegli un caso clinico",
        "onboarding.step1_body": "Scegli uno dei casi preimpostati (o scrivine uno tuo) nella griglia sotto. Ognuno ha un focus diagnostico reale — cardio, neuro, pediatrico, psichiatrico.",
        "onboarding.step2_title": "2 · Scegli la profondità di analisi",
        "onboarding.step2_body": "Light, Medium o Premium — l'istruzione è identica per ChatGPT, Claude e Gemini così il confronto resta equo.",
        "onboarding.step3_title": "3 · Esegui QVAC, poi incolla le risposte cloud",
        "onboarding.step3_body": "Tether QVAC MedPsy 4B risponde subito in locale. Apri ChatGPT / Claude / Gemini con i link forniti, incolla lo stesso prompt manualmente, poi incolla ogni risposta nella relativa scheda.",
        "onboarding.step4_title": "4 · Leggi i KPI e i grafici",
        "onboarding.step4_body": "Consenso, accuratezza clinica vs diagnosi di riferimento, accordo sul triage, punteggio privacy e una classifica di sessione che cresce con ogni caso eseguito.",
        "onboarding.cta": "Ho capito — si parte 🚀",
        "sidebar.how_it_works": "Come funziona",
        # Reset confirm
        "reset.confirm_title": "⚠️ Azzerare tutto?",
        "reset.confirm_body": "Verranno azzerati caso corrente, risposte incollate, KPI e il wallet tornerà a 0.00 USDT. **I risultati salvati in colonna restano intatti.**",
        "reset.confirm_yes": "Sì, azzera tutto",
        "reset.confirm_cancel": "Annulla",
        # Case section
        "section.case_title": "1 · Scegli il caso clinico",
        "case.preview_btn": "👁️ Anteprima",
        "case.load_btn": "Carica questo caso",
        "case.loaded_badge": "✓ Caricato",
        "case.preview_dialog_title": "Anteprima caso",
        "case.focus_label": "Focus clinico",
        "case.specialty.cardio": "Cardiologia · Emergenza",
        "case.specialty.neuro": "Neurologia",
        "case.specialty.pediatric": "Pediatria · Chirurgia",
        "case.specialty.psych": "Psichiatria",
        "case.specialty.custom": "Caso personalizzato",
        "case.focus.case1": "Riconoscere lo STEMI e cogliere la controindicazione nitrati-sildenafil prima di trattare.",
        "case.focus.case2": "Applicare i criteri McDonald 2017 ed escludere correttamente i mimic già smentiti dai referti.",
        "case.focus.case3": "Valutare il quadro appendicitico e adattare la profilassi antibiotica a un'allergia documentata.",
        "case.focus.case4": "Distinguere uno switch maniacale da una mania indotta da sostanze e cogliere il conflitto litio/rene.",
        "case.focus.case5": "Incolla un tuo caso anonimizzato per testare uno scenario reale.",
        "case.custom_hint": "Caso personalizzato — seleziona uno slot nella **colonna sinistra** per caricare un preset.",
        "case.sidebar_nav_hint": "Seleziona, carica o richiama i casi dalla colonna sinistra.",
        "case.short.case1": "Dolore toracico",
        "case.short.case2": "SM · RMN",
        "case.short.case3": "Appendicite",
        "case.short.case4": "Mania",
        "case.short.case5": "Personalizzato",
        # Tier
        "section.tier_title": "2 · Profondità di analisi (uguale per ogni cloud)",
        # Run
        "section.run_title": "3 · Esegui il benchmark",
        "run.ready_badge": "Pronto all'esecuzione",
        "status.step_loading": "Caricamento pesi Tether QVAC MedPsy 4B on-device…",
        "status.step_generating": "Generazione diagnosi differenziale in locale…",
        "status.step_encrypting": "Calcolo statistiche di latenza / throughput…",
        "status.step_done": "QVAC MedPsy 4B completato — 0 byte hanno lasciato questo dispositivo.",
        "status.step_error": "Inferenza locale non riuscita",
        "status.live_tokens": "{n} parole generate dal vivo, on-device",
        "status.live_words": "parole",
        # Solo path legacy paste-web (OLD/); il gold Automated Benchmark usa sidecar QVAC.
        "error.ollama_offline": (
            "⚠️ Helper legacy: path Ollama non usato sul gold Automated Benchmark. "
            "Avvia il sidecar QVAC (vedi README) per MedPsy on-device."
        ),
        "error.model_missing": (
            "⚠️ Helper legacy: Ollama medpsy-4b-cpu non usato sul path gold. "
            "Scarica i GGUF con ./install.sh e avvia il sidecar QVAC."
        ),
        "error.ollama_runtime": "⚠️ Inferenza locale non riuscita: {error}. Nessuna risposta sostitutiva viene mostrata — questo benchmark non inventa mai un risultato per QVAC.",
        # Model cards
        "card.instructions_cloud": "Apri **{name}** dal Link ↗ qui sopra, incolla lì lo stesso prompt, poi incolla qui la sua risposta.",
        "card.instructions_local": "Generato on-device da QVAC MedPsy 4B; extract/judge usano comunque OpenRouter in scoring.",
        "card.tier_mismatch": "Questa risposta di QVAC è stata generata a profondità **{old}** — ora è selezionata **{new}**. Esegui di nuovo il benchmark per rigenerarla alla nuova profondità.",
        "card.status_empty": "⏳ In attesa di incolla",
        "card.status_filled": "✓ Incollato · {n} parole",
        "card.view_full": "🔍 Leggi risposta completa",
        "card.full_dialog_title": "🔍 Risposta diagnostica completa",
        "card.reasoning_expander": "Ragionamento reale (generato dal modello locale)",
        "card.reasoning_caption": "Token di ragionamento non modificati prodotti da MedPsy-4B prima della risposta finale — mostrati per trasparenza, non selezionati a mano.",
        "card.step1": "1️⃣ Apri il link",
        "card.step2": "2️⃣ Incolla lì il prompt",
        "card.step3": "3️⃣ Incolla la risposta sotto",
        "card.steps_inline": "Apri il link → incolla il prompt → incolla qui la risposta",
        # Compact layout: eyebrows + stepper
        "eyebrow.case": "Caso clinico",
        "eyebrow.tier": "Profondità QVAC & esecuzione",
        "eyebrow.run": "Esegui benchmark",
        "eyebrow.output": "Risposte dei modelli",
        "eyebrow.results": "Risultati",
        "stepper.case": "Caso",
        "stepper.tier": "Tier",
        "stepper.run": "Esegui",
        "stepper.results": "Risultati",
        "case.picker_label": "Scegli un caso clinico",
        "sidebar.guide_btn": "📖 Guida — come si usa",
        # KPI command center
        "kpi.command_center_title": "📌 Centro di controllo — risultati chiave a colpo d'occhio",
        "kpi.best_model": "🏆 Miglior consenso tra LLM",
        "kpi.best_clinical": "🎯 Migliore vs diagnosi certa",
        "kpi.best_consensus": "🤝 Miglior consenso tra LLM",
        "kpi.privacy_avg": "🔒 Punteggio privacy medio",
        "kpi.privacy_avg_help": "100% = generazione on-device (QVAC). 0% = dati inviati a un'API cloud per ogni modello cloud presente. Extract/judge possono ancora usare OpenRouter.",
        "kpi.consensus_avg": "🤝 Accordo medio tra modelli",
        "kpi.consensus_avg_help": "Media pesata tra Affidabilità (sovrapposizione con l'intera lista differenziale di ogni altro modello), Accuratezza (sovrapposizione della sola diagnosi principale con ogni altro modello) e Similarità semantica (coincidenza di significato via embedding locale, cosi' le parafrasi non vengono penalizzate). È un punteggio di **accordo**, non di correttezza — vedi sotto.",
        "kpi.semantic_unavailable": "ℹ️ La similarità semantica (il KPI di significato) richiede il modello di embedding locale — avvialo con lo script di setup per abilitare questo terzo segnale; nel frattempo i punteggi sopra usano solo Affidabilità + Accuratezza.",
        "kpi.glance_title": "📊 Dettaglio KPI completo — sempre visibile, senza click",
        "kpi.glance_caption": "Tabelle consenso per questo caso. La privacy è la riga compatta sotto i grafici di ranking.",
        "kpi.mode_note_consensus": "**Nessuna diagnosi confermata è stata inserita per questo round** → i punteggi sotto sono un **punteggio di accordo (0-100%)**: quanto le risposte dei modelli si sovrappongono tra loro nelle parole e nel significato. Senza una verità di riferimento, questo misura il *consenso*, non la *correttezza* — 100% significherebbe risposte identiche parola per parola e significato per significato, cosa rara anche quando tutti hanno clinicamente ragione. Incolla una diagnosi confermata nella sezione sopra per sbloccare il **voto clinico** da 1 a 10.",
        "kpi.mode_note_gold": "**Diagnosi di riferimento confermata attiva** → la classifica clinica è **assoluta** (non riscalata al 100%). Score = match semantico su diagnosi, piano, urgenza e sintesi vs riferimento — parafrasi in altra lingua valgono; **non** serve copiare parola per parola.",
        "kpi.explain_button": "🔍 Guarda esattamente come è stato calcolato ogni punteggio",
        "explain.dialog_title": "🔍 Come è stato calcolato ogni punteggio",
        "explain.dialog_intro": (
            "Nessun numero nascosto: ogni dimensione è un 0–100% continuo da embedding semantici locali "
            "(similarità coseno su testo clinico estratto). Lo score mostrato è la **media pesata letterale** "
            "sotto — Cons.% viene poi riscalato con #1 = 100%."
        ),
        "explain.consensus_label": "Punteggio consenso (Cons.%)",
        "explain.consensus_desc": (
            "Quattro fette cliniche confrontate a coppie con ogni altro modello: blocco diagnosi, piano/next steps, "
            "sintesi clinica e urgenza (65% testo semantico + 35% etichetta triage). "
            "Ogni valore è 0–100 continuo — non gradini fissi tipo 100/75/50/0."
        ),
        "explain.consensus_rescale_note": (
            "Cons_raw è la media pesata sopra. Cons.% in tabella è riscalato vs il leader del round "
            "(#1 → 100%) — Ref.% in modalità gold **non** è riscalato."
        ),
        "explain.dimension_pairs": "{dim} — similarità a coppie vs altri modelli",
        "explain.reliability_label": "① Affidabilità — accordo a coppie",
        "explain.reliability_desc": "Media della sovrapposizione tra la diagnosi differenziale di questo modello e quella di ogni altro modello, mescolando la sovrapposizione a livello di lista (65%) con quella per keyword su tutto il testo (35%).",
        "explain.accuracy_label": "② Accuratezza (consenso) — coincidenza keyword",
        "explain.accuracy_desc": "Sovrapposizione media di keyword tra la diagnosi principale (la prima) di questo modello e la diagnosi principale di ogni altro modello — stessa logica dell'Affidabilità, ma limitata alla sola prima ipotesi invece che a tutta la lista differenziale.",
        "explain.semantic_label": "③ Similarità semantica — coincidenza di significato",
        "explain.semantic_desc": "Similarità coseno tra la diagnosi principale di questo modello e quella di ogni altro modello, calcolata da un modello di embedding locale (all-minilm) — è ciò che cattura \"stessa diagnosi, parole diverse\".",
        "explain.semantic_unavailable_note": (
            "Modello di embedding locale non raggiungibile in questo round — il consenso usa un fallback keyword affidabilità/accuratezza."
        ),
        "explain.gold_label": "④ Vs. diagnosi di riferimento confermata",
        "explain.gold_desc": "Misurato vs la diagnosi certa incollata usando il **significato**, non le parole uguali: diagnosi (40%), piano & prossimi step (30%), urgenza (20%), sintesi clinica (10%). Stessa diagnosi con parole diverse ottiene punteggio alto.",
        "explain.grade_label": "Voto clinico",
        "explain.grade_rubric": "Punteggio clinico 0-100% riscalato in un voto familiare da 1 a 10: **1** = non correlato al riferimento · **5** ≈ parzialmente corretto (stessa categoria, diagnosi specifica diversa) · **10** = coincide col riferimento quasi esattamente nelle parole e nel significato, con diagnosi differenziale completa.",
        "explain.formula_label": "Formula",
        "explain.formula_consensus_dim": "Cons_raw = {formula}\n= {raw}%\nCons_display = {rescaled}% (riscalato vs #1 del round)",
        "explain.formula_gold_dim": "Ref = {formula}\n= {score}% (assoluto — non riscalato)",
        "explain.own_keywords": "Keyword della diagnosi principale di questo modello",
        "explain.consensus_keywords": "Keyword con supporto della maggioranza tra tutti i modelli",
        "explain.matched_keywords": "Keyword che questo modello condivide con la maggioranza",
        "explain.no_keywords": "Nessuna keyword estratta per questo confronto.",
        "explain.urgency_label": "Urgenza/triage dichiarata",
        "explain.privacy_label": "Privacy",
        "explain.vs": "vs. {name}",
        "explain.select_model": "Modello",
        "kpi.triage_agreement_help": "% di modelli che hanno assegnato la stessa etichetta di urgenza/triage a questo caso.",
        "kpi.time_saved": "⏱️ Tempo risparmiato stimato da QVAC",
        "kpi.time_saved_help": "Stima illustrativa: assume ~{overhead:.0f}s di overhead manuale per consulto cloud (apertura sito, copia/incolla, attesa) confrontato con la latenza misurata di QVAC in locale.",
        "kpi.models_active": "🧪 Modelli confrontati",
        "kpi.triage": "Triage",
        "kpi.triage_caption": "Confronta il livello di urgenza/priorità assegnato da ciascun modello allo stesso caso — decisivo per la sicurezza del paziente.",
        "kpi.triage_agreement": "🚦 Accordo sul triage",
        "kpi.session": "Report di sessione",
        "urgency.critical": "🔴 Critico / emergenza",
        "urgency.high": "🟠 Alto",
        "urgency.moderate": "🟡 Moderato",
        "urgency.low": "🟢 Basso",
        "urgency.unknown": "⚪ Non dichiarato",
        "cols.urgency": "Urgenza dichiarata",
        "cols.urgency_score": "Score urgenza",
        "leaderboard.title": "📈 Classifica di sessione",
        "leaderboard.caption": "Aggiungi ogni caso testato per costruire una classifica cumulativa — perfetto per una registrazione con più casi.",
        "leaderboard.empty": "Nessun round aggiunto. Esegui un caso, valutalo, poi clicca “Aggiungi questo round”.",
        "leaderboard.add_round": "➕ Aggiungi questo round alla classifica di sessione",
        "leaderboard.added_toast": "Round aggiunto alla classifica di sessione 📈",
        "leaderboard.wins": "Round vinti",
        "leaderboard.avg_score": "Media consenso %",
        "leaderboard.rounds": "Round giocati",
        "leaderboard.clear": "🗑️ Azzera classifica di sessione",
        "leaderboard.history_expander": "Storico round per round",
        "session.caption": "Salva ogni caso valutato per includerlo nel ranking definitivo mediato.",
        "session.save_round": "Salva ranking di questo caso",
        "session.save_hint": "**{case}** salvato · **{n}** caso/i in sessione — apri tab **Report di sessione** per il finale.",
        "session.save_hint_empty": "Salva **{case}** quando pronto, poi passa al caso clinico successivo.",
        "session.saved_cases": "Casi salvati in questa sessione",
        "session.col_round": "#",
        "session.col_case": "Caso",
        "session.col_mode": "Modalità",
        "session.col_best_consensus": "Miglior consenso",
        "session.col_best_clinical": "Miglior vs riferimento",
        "session.col_saved_at": "Salvato",
        "session.mode_consensus": "Consenso",
        "session.mode_gold": "Diagnosi certa",
        "session.finale_title": "Ranking definitivo mediato",
        "session.finale_caption": "Media sui casi standard (1–4) e match clinico sul caso 5 con diagnosi certa.",
        "session.finale_avg": "Classifica media (casi 1–4)",
        "session.finale_avg_help": "Media aritmetica degli score consenso su **{n}** caso/i standard salvato/i. Il #1 è mostrato come **100%**.",
        "session.finale_avg_empty": "Salva almeno un caso standard (senza diagnosi certa) per la classifica media.",
        "session.finale_gold": "Classifica diagnosi reale (caso 5)",
        "session.finale_gold_help": "Match assoluto vs diagnosi certa — **{case}**.",
        "session.finale_gold_avg_help": "Gli score clinici sono la **media rolling** delle ultime **{n}** run salvate del caso 5 (max {max}).",
        "session.finale_gold_empty": "Salva un caso con **diagnosi certa** attiva per il finale clinico.",
        "session.slots_empty": "Nessun caso salvato — esegui un benchmark e usa **Salva benchmark corrente** in colonna.",
        "session.save_not_auto": "Il salvataggio **non è automatico** — usa il pulsante **Salva** nell'area risultati sopra la classifica.",
        "session.save_ready": "Risultati pronti per **{case}** — clicca **Salva benchmark corrente** sotto.",
        "session.save_ready_short": "Non ancora salvato — clicca **Salva** per memorizzare la classifica nello slot colonna di **{case}**.",
        "session.save_case5_hint": "Seleziona il **caso 5** nel selettore in alto, poi compila il template — serve uno slot caso attivo per salvare.",
        "session.save_case5_ready": "Pronta per **run {n}/{max}** — rilancia il benchmark tra un salvataggio e l'altro.",
        "session.save_case5_ready_rolling": "Pronta per una nuova run — sostituisce la più vecchia; il finale tiene le ultime **{max}**.",
        "session.save_case5_ready_short": "Salva **run {n}/{max}**, poi richiama lo slot 5 per la media rolling.",
        "session.save_case5_ready_short_rolling": "Dopo **{max}** run, ogni nuovo salvataggio sostituisce la più vecchia. Richiama lo slot 5 per la media aggiornata.",
        "charts.tab_privacy": "Gauge privacy",
        "charts.tab_heatmap": "Heatmap concordanza",
        "charts.tab_keywords": "Keyword di consenso",
        "charts.tab_urgency": "Confronto triage",
        # Decision dialogs
        "decision.cloud_dialog_title": "🔴 Percorso Cloud selezionato",
        "decision.cloud_dialog_body": "Inviare questo caso clinico a un LLM cloud pubblico significa che il dato lascia questo dispositivo e questa giurisdizione. Per GDPR/HIPAA servirebbero un DPA firmato, il consenso del paziente e un audit trail — nessuno dei quali esiste in una chat copiata e incollata.",
        "decision.qvac_dialog_title": "🟢 Percorso QVAC locale selezionato",
        "decision.qvac_dialog_body": "Tether QVAC MedPsy 4B ha generato on-device su questo dispositivo. Extract/judge possono ancora usare OpenRouter quando lo scoring è attivo.",
        "decision.sell_dialog_title": "💠 Simulazione vendita dati anonimizzati",
        "decision.sell_dialog_body": "In un deployment reale, un record di benchmark anonimizzato e aggregato potrebbe essere venduto a consorzi di ricerca con il consenso del paziente, remunerandolo in USDT. Questo wallet simula quel flusso di reward.",
        "decision.sell_question": "Vuoi vendere il tuo record clinico anonimizzato ai consorzi di ricerca per **{amount:.2f} USDT**?",
        "decision.sell_confirm": "✅ Sì, vendi per {amount:.2f} USDT",
        "decision.sell_cancel": "❌ Annulla",
        "decision.sell_success_title": "Pagamento ricevuto",
        "decision.sell_success_body": "+{amount:.2f} USDT accreditati sul tuo wallet.",
        "decision.sell_new_balance": "Nuovo saldo",
        "decision.close": "Chiudi",
        # Automated Benchmark · ripresa storia per famiglia di caso
        "bench.family_found": (
            "Trovate {n} run precedenti in {cohorts} versioni di Confirm per questo "
            "caso+riferimento — ripristina un riferimento confermato salvato per "
            "riaprire quel cohort."
        ),
        "bench.family_select": "Versioni Confirm precedenti",
        "bench.family_restore_btn": "Ripristina ultimo riferimento confermato",
        "bench.family_restore_help": (
            "Carica il contratto gold congelato esatto dalla History (stessi split "
            "dei claim). Non unisce versioni Confirm diverse. "
            "Re-Prepare o modifiche agli split dei claim possono spezzare cohort_id — "
            "ripristina il Confirm esatto."
        ),
        "bench.family_restore_ok": (
            "Riferimento confermato ripristinato — candidati sbloccati; Rebuild usa "
            "questo cohort."
        ),
        "bench.family_restore_fail": "Ripristino non riuscito: {err}",
        "bench.family_other_cohorts": (
            "Esistono altre versioni Confirm di questo caso ({n} run) — ripristina "
            "quel riferimento per raggrupparle. Contratti gold diversi non vengono "
            "mai uniti in automatico. Re-Prepare o modifiche agli split dei claim "
            "aprono un nuovo cohort."
        ),
        "bench.rebuild_sec_label": "Ricostruisci media su N run · $0 API",
        "bench.rebuild_scope_label": "Ambito media",
        "bench.rebuild_scope_same": "Solo caso selezionato",
        "bench.rebuild_scope_portfolio": (
            "Tutti i casi (portfolio · esplorativo misti)"
        ),
        "bench.rebuild_n_label": "Media su ≤N scored / modello",
        "bench.rebuild_btn": "Ricostruisci media · ≤{n} scored / modello · apri popup KPI · $0",
        "bench.rebuild_btn_help_same": (
            "Media offline solo per il cohort immutabile dello slot Case "
            "selezionato (stesso stem + gold confermato). N = max osservazioni "
            "**scored** per modello (più recenti prima); N/A tecnici saltati, "
            "si usano scored più vecchi. Costo API zero."
        ),
        "bench.rebuild_btn_help_portfolio": (
            "Media esplorativa offline su tutti gli slot Case con History per "
            "questa chiave: per ogni modello, media delle proprie ≤N "
            "osservazioni **scored** (stesso track + scoring_version; N/A "
            "saltati, scored più vecchi usati). "
            "Casi/gold misti — non validazione clinica. Costo API zero."
        ),
        "bench.rebuild_portfolio_intro": (
            "**Tutti i casi (portfolio)** — media esplorativa su ogni slot Case "
            "con dati per questa chiave. **≤N scored obs/modello**; N/A saltati, "
            "scored più vecchi usati. Stesso track + scoring_version. I contratti "
            "gold **non** vengono uniti. **Nessuna chiamata API.**"
        ),
        "bench.rebuild_portfolio_stats": (
            "**≤{n} scored/modello · {cases} stem di caso distinti** "
            "(run eleggibili: {avail}) · track `{track}` · esplorativo misti · "
            "N/A saltati, scored più vecchi usati"
        ),
        "bench.rebuild_portfolio_quiet": (
            "Gli score restano relativi al riferimento di ciascun caso; "
            "portfolio = media esplorativa misti (≤N scored/modello; N/A "
            "saltati, scored più vecchi usati), "
            "non validazione clinica. Roster misti ok. Versioni di scoring "
            "incompatibili non vengono mai unite."
        ),
        "bench.rebuild_portfolio_success": (
            "**Tutti i casi (portfolio)** · **≤N/modello · {n} doc run · "
            "{cases} stem** · esplorativo misti · roster misti ok · **$0 API**"
        ),
        "bench.rebuild_portfolio_caption": (
            "Il Clinical Composite di ogni run resta relativo al gold confermato "
            "di quel caso. La media tra gli slot Case 1–5 è analisi esplorativa "
            "misti — non una claim di validazione clinica. "
            "≤N scored obs/modello; N/A saltati, scored più vecchi usati. "
            "Failed % = fallimenti tecnici incontrati riempiendo la quota scored."
        ),
        "bench.rebuild_portfolio_mean_rank": "Rank medio (più basso = meglio): {ranks}",
        "bench.rebuild_need_portfolio": (
            "Serve almeno **1** run complete/partial eleggibile per il portfolio "
            "(trovate {n}). Stesso track + scoring_version; roster diversi ok."
        ),
        "bench.rebuild_reopen_portfolio": (
            "Riapri media portfolio tutti i casi · ≤N/modello · {n} doc · "
            "{cases} stem · $0"
        ),
        "bench.case_slots_label": "Casi clinici",
        "bench.case_slot_btn": "Case {n}",
        "bench.case_slot_empty": "Vuoto",
        "bench.case_slot_runs": "{n} run",
        "bench.new_case_btn": "Nuovo caso",
        "bench.new_case_help": (
            "Apre il prossimo slot Case vuoto, oppure crea Case 6, 7, … quando "
            "Case 1–5 sono pieni. Incolla stem + riferimento, poi Prepare / Confirm."
        ),
        "bench.new_case_full": (
            "Raggiunto il limite soft di {n} slot Case per questa chiave. "
            "Seleziona uno slot esistente per modificare o ricalcolare le medie."
        ),
        "bench.case_slot_loaded": (
            "Caricato **Case {n}**: stem + riferimento confermato dalla History."
        ),
        "bench.case_slot_ready_empty": (
            "**Case {n}** è vuoto — incolla un caso anonimizzato e il "
            "riferimento, poi Prepare / Confirm."
        ),
    },
}


def t(key: str, lang: Optional[str] = None, **kwargs) -> str:
    lng = lang or DEFAULT_LANG
    if lng not in _STRINGS:
        lng = DEFAULT_LANG
    text = _STRINGS[lng].get(key) or _STRINGS[DEFAULT_LANG].get(key, key)
    if kwargs:
        return text.format(**kwargs)
    return text


def qvac_instruction(lang: str = "en") -> str:
    return t("qvac.instruction", lang)


def tier_description(tier_key: str, lang: str = "en") -> str:
    return t(f"tier.{tier_key}.desc", lang)


def tier_instruction(tier_key: str, lang: str = "en") -> str:
    return t(f"tier.instruction.{tier_key}", lang)
