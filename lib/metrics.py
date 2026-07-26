"""KPI tables and diagnostic metrics."""

from typing import Optional, Tuple

import pandas as pd

from lib.clinical_scoring import weighted_dimension_composite
from lib.i18n import t
from lib.model_labels import MODEL_LABELS, full_model_label, name_and_version
from lib.tiers import MODEL_CONFIG


def table_version_label(model_key: str, labels=None) -> str:
    """Lazy wrapper so metrics does not import cloud_tiers at module load."""
    from lib.cloud_tiers import table_version_label as _impl

    return _impl(model_key, labels)


# Single source of truth for clinical dimension weights (0–100 each, true weighted mean).
# Unified rule (demo-friendly): 40/30/20/10 for both consensus and gold mode.
CONSENSUS_DIM_WEIGHTS = (
    ("diagnosis", 0.40),
    ("management", 0.30),
    ("urgency", 0.20),
    ("summary", 0.10),
)
GOLD_DIM_WEIGHTS = CONSENSUS_DIM_WEIGHTS

# Stima illustrativa (non misurata) del tempo medio richiesto dal flusso manuale
# cloud: apertura sito, attesa caricamento, copia prompt, incolla, attesa risposta,
# copia risposta, incolla nel riquadro. Usata solo per il KPI "tempo risparmiato".
CLOUD_MANUAL_OVERHEAD_S = 75.0

URGENCY_META = {
    "critical": {"label_key": "urgency.critical", "score": 100, "color": "#ef4444"},
    "high": {"label_key": "urgency.high", "score": 80, "color": "#f97316"},
    "moderate": {"label_key": "urgency.moderate", "score": 50, "color": "#f5c518"},
    "low": {"label_key": "urgency.low", "score": 20, "color": "#22c55e"},
    None: {"label_key": "urgency.unknown", "score": 0, "color": "#64748b"},
}


def privacy_score(is_cloud: bool) -> int:
    return 0 if is_cloud else 100


def clinical_grade_10(score_pct: float) -> float:
    """Rescale a 0-100% clinical accuracy score to a familiar 1-10 school grade.

    Only meaningful in gold-standard mode (a confirmed reference diagnosis was
    provided) — a "10" means the model's primary diagnosis matched the
    reference almost exactly in both wording/keywords and meaning, with full
    differential coverage. Without a reference there is no ground truth to
    grade against, so this is intentionally never shown in consensus-only
    mode (see build_unified_ranking).
    """
    if score_pct is None:
        return None
    return round(max(1.0, min(10.0, 1.0 + (score_pct / 100.0) * 9.0)), 1)


def _blend_score(a: float, b: float, sem, weights=(0.25, 0.25, 0.50)) -> float:
    """Blend keyword metrics with semantic similarity for the final score.

    Semantic similarity is the only signal that stays honest across
    languages (QVAC in Italian vs. cloud in English) and across verbosity
    levels (a terse Gemini line vs. a thorough Claude paragraph saying the
    same thing). Keyword overlap still contributes, but at equal weight to
    each other and below semantic — otherwise thorough, correct answers
    from stronger models get crushed by formatting/length differences.
    """
    if sem is None:
        return round((a + b) / 2, 1)
    wa, wb, ws = weights
    return round(a * wa + b * wb + sem * ws, 1)


def _L(lang: str) -> dict:
    """Column label map for the active UI language."""
    return {
        "model": t("cols.model", lang),
        "version": t("cols.version", lang),
        "tier_req": t("cols.tier_requested", lang),
        "tier": t("cols.tier", lang),
        "local": t("cols.local", lang),
        "kpi": t("cols.kpi", lang),
        "kpi_cloud": t("cols.kpi_cloud", lang),
        "kpi_local": t("cols.kpi_local", lang),
        "privacy": t("cols.privacy", lang),
        "latency": t("cols.latency", lang),
        "token_out": t("cols.token_out", lang),
        "reliability": t("cols.reliability", lang),
        "acc_consensus": t("cols.accuracy_consensus", lang),
        "score_consensus": t("cols.consensus_score", lang),
        "rank_consensus": t("cols.consensus_rank", lang),
        "acc_primary": t("cols.primary_accuracy", lang),
        "ddx_cov": t("cols.ddx_coverage", lang),
        "score_clinical": t("cols.clinical_score", lang),
        "rank_clinical": t("cols.clinical_rank", lang),
        "rank": t("cols.rank", lang),
        "score_final": t("cols.final_score", lang),
        "rel_short": t("cols.clinical_summary_short", lang),
        "acc_cons_short": t("cols.diagnosis_match_short", lang),
        "sem_short": t("cols.plan_next_steps_short", lang),
        "urg_dim_short": t("cols.urgency_agreement_short", lang),
        "score_cons_rescaled": t("cols.consensus_score_rescaled", lang),
        "score_cons_short": t("cols.consensus_score_short", lang),
        "acc_clin_short": t("cols.clinical_accuracy", lang),
        "ddx_short": t("cols.ddx_coverage_short", lang),
        "sem_clin_short": t("cols.semantic_gold_short", lang),
        "score_clin_short": t("cols.clinical_score_short", lang),
        "grade_10": t("cols.grade_10", lang),
        "urgency": t("cols.urgency", lang),
        "urgency_score": t("cols.urgency_score", lang),
    }


# Fields stored as translated column headers inside ranking_records snapshots.
RANKING_COL_FIELDS = (
    "model",
    "acc_cons_short",
    "sem_short",
    "rel_short",
    "urg_dim_short",
    "score_cons_rescaled",
    "privacy",
    "acc_clin_short",
    "ddx_short",
    "sem_clin_short",
    "score_clin_short",
    "grade_10",
    "rank_consensus",
    "rank_clinical",
    "version",
)


def localize_ranking_df(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Map saved ranking table headers (any UI language) to the active language."""
    if df.empty:
        return df
    L = _L(lang)
    rename = {}
    for field in RANKING_COL_FIELDS:
        target = L[field]
        if target in df.columns:
            continue
        for other_lang in ("en", "it"):
            if other_lang == lang:
                continue
            source = _L(other_lang)[field]
            if source in df.columns and source not in rename:
                rename[source] = target
                break
    if rename:
        df = df.rename(columns=rename)
    return df


def build_performance_table(results: dict, lang: str = "en") -> pd.DataFrame:
    L = _L(lang)
    rows = []
    for key, cfg in MODEL_CONFIG.items():
        data = results.get(key, {})
        stats = data.get("stats", {})
        if cfg["cloud"]:
            rows.append(
                {
                    "key": key,
                    "TTFT (s)": "—",
                    "TPS": "—",
                    L["latency"]: "—",
                    L["token_out"]: "—",
                    L["kpi"]: L["kpi_cloud"],
                    L["privacy"]: "0%",
                }
            )
        else:
            rows.append(
                {
                    "key": key,
                    "TTFT (s)": str(stats.get("ttft_s") or "—"),
                    "TPS": str(stats.get("tps") or stats.get("tokens_per_second") or "—"),
                    L["latency"]: str(stats.get("latency_s") or "—"),
                    L["token_out"]: str(stats.get("completion_tokens") or "—"),
                    L["kpi"]: L["kpi_local"],
                    L["privacy"]: "100%",
                }
            )
    return pd.DataFrame(rows)


TABLE_MODEL_SHORT = {k: full_model_label(k) for k in MODEL_LABELS}


def table_model_name(key: str) -> str:
    """Name · Version for tables."""
    if not key:
        return "—"
    if key in MODEL_LABELS:
        return full_model_label(key)
    cfg = MODEL_CONFIG.get(key, {})
    return TABLE_MODEL_SHORT.get(key, cfg.get("name", key))


def table_model_parts(key: str) -> tuple:
    """(Name, Version) columns for ranking tables."""
    if not key:
        return ("—", "—")
    return name_and_version(key)


def _infer_keys_from_model_column(df: pd.DataFrame, model_col: str) -> list:
    """Recover internal keys from legacy snapshots that only stored a model label."""
    name_to_key = {cfg["name"]: k for k, cfg in MODEL_CONFIG.items()}
    short_to_key = dict(TABLE_MODEL_SHORT)
    keys = []
    for raw in df[model_col].fillna(""):
        label = str(raw).strip()
        key = name_to_key.get(label) or short_to_key.get(label)
        if not key and " · " in label:
            base = label.split(" · ", 1)[0].strip()
            key = name_to_key.get(base) or short_to_key.get(base)
        keys.append(key)
    return keys


def prepare_model_table(
    df: pd.DataFrame,
    lang: str = "en",
    tier_labels: Optional[dict] = None,
    columns: Optional[list] = None,
    drop_key: bool = True,
) -> pd.DataFrame:
    """Format any model KPI table with Model + Version as the first two columns."""
    if df.empty:
        return df

    L = _L(lang)
    out = df.copy()

    if "key" not in out.columns:
        if L["model"] in out.columns:
            out["key"] = _infer_keys_from_model_column(out, L["model"])
        else:
            out["key"] = None

    # Saved snapshots may carry an old Version/Versione column — always refresh from tier labels.
    for other_lang in ("en", "it"):
        stale = _L(other_lang)["version"]
        if stale in out.columns:
            out = out.drop(columns=[stale])

    out[L["model"]] = [table_model_name(k) for k in out["key"]]
    out[L["version"]] = [
        table_version_label(k, tier_labels) if k else "—"
        for k in out["key"]
    ]

    if drop_key and "key" in out.columns:
        out = out.drop(columns=["key"])

    front = [L["model"], L["version"]]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]

    if columns:
        metric_cols = []
        for col in columns:
            if col in ("key", L["model"], L["version"]):
                continue
            if col in out.columns and col not in metric_cols:
                metric_cols.append(col)
        out = out[front + metric_cols]

    return out

def build_consensus_table(
    compare: dict, model_keys: list, lang: str = "en", tier_labels: Optional[dict] = None
) -> pd.DataFrame:
    """Full results table with dual ranking columns when gold standard is active."""
    _ = tier_labels  # display formatting applied in ui.show_model_table
    L = _L(lang)
    use_gold = compare.get("mode") == "gold_standard" and compare.get("gold")
    df = build_unified_ranking(compare, model_keys, lang)
    if df.empty:
        return df

    cols = [
        "key",
        L["acc_cons_short"],
        L["sem_short"],
        L["rel_short"],
        L["urg_dim_short"],
        L["score_cons_rescaled"],
        L["rank_consensus"],
    ]
    if use_gold:
        cols += [L["score_clin_short"], L["rank_clinical"], L["grade_10"]]
    return df[cols]


def _gold_dim(gold: dict, field: str, key: str):
    val = (gold.get(field) or {}).get(key)
    return val if val is not None else "—"


def _gold_dim_numeric(gold: dict, field: str, key: str) -> Optional[float]:
    val = (gold.get(field) or {}).get(key)
    return float(val) if val is not None else None


def _consensus_dimensions(compare: dict, key: str) -> dict:
    dims = compare.get("clinical_dimensions") or {}
    return {
        "diagnosis": dims.get("diagnosis", {}).get(key),
        "management": dims.get("management", {}).get(key),
        "summary": dims.get("summary", {}).get(key),
        "urgency": dims.get("urgency", {}).get(key),
    }


def _gold_dimensions(gold: dict, key: str) -> dict:
    return {
        "diagnosis": _gold_dim_numeric(gold, "semantic_diagnosis", key),
        "management": _gold_dim_numeric(gold, "semantic_management", key),
        "urgency": _gold_dim_numeric(gold, "semantic_urgency", key),
        "summary": _gold_dim_numeric(gold, "semantic_summary", key),
    }


def _weighted_from_dimensions(dims: dict, weights: tuple) -> Optional[float]:
    return weighted_dimension_composite([(dims.get(k), w) for k, w in weights])


def _dimension_label(dim_key: str, lang: str) -> str:
    keys = {
        "diagnosis": "cols.diagnosis_match_short",
        "management": "cols.plan_next_steps_short",
        "summary": "cols.clinical_summary_short",
        "urgency": "cols.urgency_agreement_short",
    }
    return t(keys.get(dim_key, dim_key), lang)


def format_weighted_formula(dims: dict, weights: tuple, lang: str = "en") -> str:
    """Human-readable weighted sum, e.g. 35%×82 (Dx) + 25%×71 (Plan) + …"""
    parts = []
    for dim_key, weight in weights:
        val = dims.get(dim_key)
        if val is None:
            continue
        label = _dimension_label(dim_key, lang)
        parts.append(f"{weight * 100:.0f}%×{val} ({label})")
    return " + ".join(parts) if parts else "—"


def consensus_raw_score(compare: dict, key: str) -> float:
    """Consensus raw 0–100: weighted mean of four clinical dimensions."""
    dims = _consensus_dimensions(compare, key)
    weighted = _weighted_from_dimensions(dims, CONSENSUS_DIM_WEIGHTS)
    if weighted is not None:
        return weighted
    rel = compare.get("reliability", {}).get(key, 0.0)
    acc = compare.get("accuracy_consensus", {}).get(key, 0.0)
    sem = compare.get("semantic_similarity", {}).get(key) if compare.get("semantic_available") else None
    return _blend_score(rel, acc, sem)


def build_gold_table(compare: dict, model_keys: list, lang: str = "en") -> pd.DataFrame:
    if not compare.get("gold"):
        return pd.DataFrame()

    L = _L(lang)
    gold = compare["gold"]
    rows = []
    for key in model_keys:
        cfg = MODEL_CONFIG.get(key, {})
        if not cfg:
            continue
        score = _clinical_gold_score(gold, key)
        row = {
            "key": key,
            L["acc_primary"]: _gold_dim(gold, "semantic_diagnosis", key),
            L["ddx_cov"]: _gold_dim(gold, "semantic_management", key),
            L["sem_clin_short"]: _gold_dim(gold, "semantic_urgency", key),
            L["score_clinical"]: score,
            L["grade_10"]: clinical_grade_10(score),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(L["score_clinical"], ascending=False)
        df[L["rank_clinical"]] = range(1, len(df) + 1)
    return df


def rescale_relative(scores: dict) -> dict:
    """Rescale so the best model in the group = 100%, others proportional.

    Used for the *consensus* ranking when there is no ground truth: the
    leader always reads as 100% so the chart is intuitive, and everyone
    else scales down by how far they are from the leader — not by an
    arbitrary absolute ceiling.
    """
    if not scores:
        return {}
    peak = max(scores.values())
    if peak <= 0:
        return {k: 0.0 for k in scores}
    return {k: round(v / peak * 100, 1) for k, v in scores.items()}


def _clinical_gold_score(gold: dict, key: str) -> float:
    """Absolute clinical score vs reference — weighted semantic dimensions."""
    if gold.get("semantic_available"):
        dims = _gold_dimensions(gold, key)
        weighted = _weighted_from_dimensions(dims, GOLD_DIM_WEIGHTS)
        if weighted is not None:
            return weighted

    composite = (gold.get("semantic_composite") or {}).get(key)
    if composite is not None and gold.get("semantic_available"):
        return float(composite)

    sem = gold.get("semantic_accuracy", {}).get(key) if gold.get("semantic_available") else None
    if sem is not None:
        return float(sem)

    acc_p = gold["accuracy_primary"].get(key, 0)
    cov = gold["coverage_ddx"].get(key, 0)
    return round((acc_p + cov) / 2, 1)


def _consensus_score(compare: dict, key: str, rel: float, acc: float, sem) -> float:
    """Alias for consensus_raw_score (legacy call sites)."""
    return consensus_raw_score(compare, key)


def build_unified_ranking(compare: dict, model_keys: list, lang: str = "en") -> pd.DataFrame:
    """Dual ranking: consensus (relative, best=100%) and clinical vs reference (absolute)."""
    L = _L(lang)
    use_gold = compare.get("mode") == "gold_standard" and compare.get("gold")
    rows = []

    raw_cons = {}
    for key in model_keys:
        cfg = MODEL_CONFIG.get(key, {})
        if not cfg:
            continue
        dims = compare.get("clinical_dimensions") or {}
        dx = dims.get("diagnosis", {}).get(key)
        plan = dims.get("management", {}).get(key)
        summary = dims.get("summary", {}).get(key)
        urg = dims.get("urgency", {}).get(key)
        raw_cons[key] = consensus_raw_score(compare, key)

    rescaled_cons = rescale_relative(raw_cons)

    for key in model_keys:
        cfg = MODEL_CONFIG.get(key, {})
        if not cfg:
            continue
        dims = compare.get("clinical_dimensions") or {}
        dx = dims.get("diagnosis", {}).get(key)
        plan = dims.get("management", {}).get(key)
        summary = dims.get("summary", {}).get(key)
        urg = dims.get("urgency", {}).get(key)

        row = {
            "key": key,
            L["model"]: cfg["name"],
            L["acc_cons_short"]: dx if dx is not None else "—",
            L["sem_short"]: plan if plan is not None else "—",
            L["rel_short"]: summary if summary is not None else "—",
            L["urg_dim_short"]: urg if urg is not None else "—",
            L["score_cons_rescaled"]: rescaled_cons.get(key, 0),
            L["privacy"]: privacy_score(cfg["cloud"]),
        }

        if use_gold:
            g = compare["gold"]
            score_gold = _clinical_gold_score(g, key)
            row[L["acc_clin_short"]] = _gold_dim(g, "semantic_diagnosis", key)
            row[L["ddx_short"]] = _gold_dim(g, "semantic_management", key)
            row[L["sem_clin_short"]] = _gold_dim(g, "semantic_urgency", key)
            row[L["score_clin_short"]] = score_gold
            row[L["grade_10"]] = clinical_grade_10(score_gold)
        else:
            row[L["acc_clin_short"]] = None
            row[L["ddx_short"]] = None
            row[L["sem_clin_short"]] = None
            row[L["score_clin_short"]] = None
            row[L["grade_10"]] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Independent ranks — a model can be #1 clinically and #3 on consensus.
    cons_order = df.sort_values(L["score_cons_rescaled"], ascending=False).reset_index(drop=True)
    cons_ranks = {cons_order.loc[i, "key"]: i + 1 for i in range(len(cons_order))}
    df[L["rank_consensus"]] = df["key"].map(cons_ranks)

    if use_gold:
        clin_order = df.sort_values(L["score_clin_short"], ascending=False).reset_index(drop=True)
        clin_ranks = {clin_order.loc[i, "key"]: i + 1 for i in range(len(clin_order))}
        df[L["rank_clinical"]] = df["key"].map(clin_ranks)
    else:
        df[L["rank_clinical"]] = None

    # Default table sort: clinical first when gold is available, else consensus.
    sort_col = L["score_clin_short"] if use_gold else L["score_cons_rescaled"]
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return df


def build_urgency_table(compare: dict, model_keys: list, lang: str = "en") -> pd.DataFrame:
    """Tabella del livello di triage/urgenza dichiarato da ciascun modello."""
    L = _L(lang)
    urgency = compare.get("urgency", {})
    rows = []
    for key in model_keys:
        cfg = MODEL_CONFIG.get(key, {})
        if not cfg:
            continue
        u = urgency.get(key, {"label": None, "score": None})
        meta = URGENCY_META.get(u.get("label"), URGENCY_META[None])
        score = u.get("score")
        rows.append(
            {
                "key": key,
                L["urgency"]: t(meta["label_key"], lang),
                # Cast to str: mixing int scores and "—" placeholders in one object
                # column breaks pandas -> Arrow serialization in st.dataframe.
                L["urgency_score"]: str(score) if score is not None else "—",
            }
        )
    return pd.DataFrame(rows)


def time_saved_seconds(qvac_latency_s) -> float:
    """Stima illustrativa del tempo risparmiato rispetto al flusso manuale cloud."""
    try:
        latency = float(qvac_latency_s)
    except (TypeError, ValueError):
        return 0.0
    return round(max(CLOUD_MANUAL_OVERHEAD_S - latency, 0.0), 1)


def build_session_entry(
    case_label: str,
    case_id: Optional[str],
    ranking_df: pd.DataFrame,
    compare: dict,
    lang: str = "en",
) -> dict:
    """Snapshot of one benchmarked case for the multi-case session story."""
    L = _L(lang)
    if ranking_df.empty:
        return {}
    use_gold = compare.get("mode") == "gold_standard" and compare.get("gold")

    scores_consensus = {}
    ranks_consensus = {}
    scores_clinical = {}
    ranks_clinical = {}
    for _, row in ranking_df.iterrows():
        key = row["key"]
        scores_consensus[key] = float(row[L["score_cons_rescaled"]])
        ranks_consensus[key] = int(row[L["rank_consensus"]])
        if use_gold and row.get(L["score_clin_short"]) is not None:
            scores_clinical[key] = float(row[L["score_clin_short"]])
            ranks_clinical[key] = int(row[L["rank_clinical"]])

    winner_cons = ranking_df[ranking_df[L["rank_consensus"]] == 1].iloc[0]
    entry = {
        "case": case_label,
        "case_id": case_id,
        "mode": "gold_standard" if use_gold else "consensus",
        "scores_consensus": scores_consensus,
        "ranks_consensus": ranks_consensus,
        "winner_consensus_key": winner_cons["key"],
        "winner_consensus_name": winner_cons[L["model"]],
        "winner_consensus_score": float(winner_cons[L["score_cons_rescaled"]]),
    }
    if use_gold and scores_clinical:
        winner_clin = ranking_df[ranking_df[L["rank_clinical"]] == 1].iloc[0]
        entry["scores_clinical"] = scores_clinical
        entry["ranks_clinical"] = ranks_clinical
        entry["winner_clinical_key"] = winner_clin["key"]
        entry["winner_clinical_name"] = winner_clin[L["model"]]
        entry["winner_clinical_score"] = float(winner_clin[L["score_clin_short"]])
        entry["winner_key"] = entry["winner_clinical_key"]
        entry["winner_name"] = entry["winner_clinical_name"]
        entry["winner_score"] = entry["winner_clinical_score"]
        entry["scores"] = scores_clinical
    else:
        entry["winner_key"] = entry["winner_consensus_key"]
        entry["winner_name"] = entry["winner_consensus_name"]
        entry["winner_score"] = entry["winner_consensus_score"]
        entry["scores"] = scores_consensus
    return entry


def _entry_scores(entry: dict, kind: str = "consensus") -> dict:
    """Read scores from new or legacy session entries."""
    if kind == "clinical":
        return entry.get("scores_clinical") or (
            entry.get("scores", {}) if entry.get("mode") == "gold_standard" else {}
        )
    return entry.get("scores_consensus") or entry.get("scores", {})


def build_session_history_table(history: list, lang: str = "en") -> pd.DataFrame:
    """One row per saved case — for the session audit trail."""
    if not history:
        return pd.DataFrame()
    L = _L(lang)
    rows = []
    for i, entry in enumerate(history, 1):
        cons = _entry_scores(entry, "consensus")
        clin = _entry_scores(entry, "clinical")
        best_cons_key = max(cons, key=cons.get) if cons else None
        best_clin_key = max(clin, key=clin.get) if clin else None
        row = {
            t("session.col_round", lang): i,
            t("session.col_case", lang): entry.get("case", "—"),
            t("session.col_mode", lang): t(
                "session.mode_gold" if entry.get("mode") == "gold_standard" else "session.mode_consensus",
                lang,
            ),
            t("session.col_best_consensus", lang): (
                f"{TABLE_MODEL_SHORT.get(best_cons_key, best_cons_key)} ({cons[best_cons_key]:.1f}%)"
                if best_cons_key
                else "—"
            ),
        }
        if clin:
            row[t("session.col_best_clinical", lang)] = (
                f"{TABLE_MODEL_SHORT.get(best_clin_key, best_clin_key)} ({clin[best_clin_key]:.1f}%)"
                if best_clin_key
                else "—"
            )
        saved_at = entry.get("saved_at")
        if saved_at:
            from lib.session_store import format_saved_at

            row[t("session.col_saved_at", lang)] = t(
                "sidebar.slot_saved_at", lang, time=format_saved_at(saved_at, lang)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_final_consensus_average(history: list, lang: str = "en") -> pd.DataFrame:
    """Arithmetic mean of consensus scores across non-gold saved cases (cases 1–4)."""
    L = _L(lang)
    consensus_entries = [e for e in history if e.get("mode") != "gold_standard"]
    if not consensus_entries:
        return pd.DataFrame()

    totals: dict = {}
    counts: dict = {}
    for entry in consensus_entries:
        for key, score in _entry_scores(entry, "consensus").items():
            totals[key] = totals.get(key, 0.0) + float(score or 0)
            counts[key] = counts.get(key, 0) + 1

    avg_scores = {key: round(totals[key] / counts[key], 1) for key in totals}
    rescaled = rescale_relative(avg_scores)

    rows = []
    for key, score in rescaled.items():
        cfg = MODEL_CONFIG.get(key, {})
        rows.append(
            {
                "key": key,
                L["model"]: table_model_name(key),
                L["score_cons_rescaled"]: score,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(L["score_cons_rescaled"], ascending=False).reset_index(drop=True)
    df[L["rank_consensus"]] = range(1, len(df) + 1)
    return df


def build_averaged_entry_from_runs(runs: list) -> dict:
    """Entry sintetica con score mediati (consenso + clinico) per tab sessione / finale."""
    entries = [r.get("entry") for r in runs if r.get("entry")]
    if not entries:
        return {}
    base = dict(entries[0])
    base["mode"] = entries[0].get("mode", "consensus")
    base["run_count"] = len(entries)

    for kind, score_key, rank_key in (
        ("consensus", "scores_consensus", "ranks_consensus"),
        ("clinical", "scores_clinical", "ranks_clinical"),
    ):
        totals: dict = {}
        counts: dict = {}
        for entry in entries:
            for key, score in _entry_scores(entry, kind).items():
                totals[key] = totals.get(key, 0.0) + float(score or 0)
                counts[key] = counts.get(key, 0) + 1
        if totals:
            avg_scores = {k: round(totals[k] / counts[k], 1) for k in totals}
            base[score_key] = avg_scores
            sorted_keys = sorted(avg_scores, key=avg_scores.get, reverse=True)
            base[rank_key] = {k: i + 1 for i, k in enumerate(sorted_keys)}
            if kind == "consensus":
                best = sorted_keys[0]
                base["winner_consensus_key"] = best
                base["winner_consensus_score"] = avg_scores[best]
            elif kind == "clinical":
                best = sorted_keys[0]
                base["winner_clinical_key"] = best
                base["winner_clinical_score"] = avg_scores[best]

    if base.get("scores_clinical"):
        base["scores"] = base["scores_clinical"]
        base["winner_key"] = base.get("winner_clinical_key")
        base["winner_score"] = base.get("winner_clinical_score")
    elif base.get("scores_consensus"):
        base["scores"] = base["scores_consensus"]
        base["winner_key"] = base.get("winner_consensus_key")
        base["winner_score"] = base.get("winner_consensus_score")
    return base


def build_averaged_ranking_from_snapshots(runs: list, lang: str = "en") -> pd.DataFrame:
    """Media per modello delle colonne score su più run dello stesso caso."""
    from lib.session_store import ranking_df_from_snapshot

    dfs = [ranking_df_from_snapshot(r, lang) for r in runs]
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        return pd.DataFrame()
    if len(dfs) == 1:
        return dfs[0].copy()

    L = _L(lang)
    score_cols = [c for c in (L["score_cons_rescaled"], L["score_clin_short"]) if c in dfs[0].columns]
    rows = []
    for key in dfs[0]["key"].tolist():
        base = dfs[0][dfs[0]["key"] == key].iloc[0].to_dict()
        for col in score_cols:
            vals = []
            for df in dfs:
                sub = df[df["key"] == key]
                if sub.empty:
                    continue
                val = sub[col].iloc[0]
                if val is None or val == "—":
                    continue
                try:
                    vals.append(float(val))
                except (TypeError, ValueError):
                    pass
            if vals:
                base[col] = round(sum(vals) / len(vals), 1)
        rows.append(base)

    df = pd.DataFrame(rows)
    df = localize_ranking_df(df, lang)
    df = df.sort_values(L["score_cons_rescaled"], ascending=False).reset_index(drop=True)
    df[L["rank_consensus"]] = range(1, len(df) + 1)
    if L["score_clin_short"] in df.columns:
        clin = df[df[L["score_clin_short"]].notna()].sort_values(
            L["score_clin_short"], ascending=False
        )
        rank_map = {row["key"]: i + 1 for i, (_, row) in enumerate(clin.iterrows())}
        df[L["rank_clinical"]] = df["key"].map(rank_map)
    return df


def build_final_gold_ranking(history: list, lang: str = "en") -> Tuple[pd.DataFrame, str]:
    """Ranking clinico vs diagnosi certa — media se più entry gold."""
    return build_final_gold_ranking_average(
        [e for e in history if e.get("mode") == "gold_standard"], lang
    )


def build_final_gold_ranking_average(entries: list, lang: str = "en") -> Tuple[pd.DataFrame, str]:
    """Media aritmetica score clinici assoluti su N run gold (es. caso 5 × 4)."""
    L = _L(lang)
    if not entries:
        return pd.DataFrame(), ""

    totals: dict = {}
    counts: dict = {}
    for entry in entries:
        for key, score in _entry_scores(entry, "clinical").items():
            totals[key] = totals.get(key, 0.0) + float(score or 0)
            counts[key] = counts.get(key, 0) + 1

    rows = []
    for key, total in totals.items():
        cfg = MODEL_CONFIG.get(key, {})
        rows.append(
            {
                "key": key,
                L["model"]: table_model_name(key),
                L["score_clin_short"]: round(total / counts[key], 1),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df, entries[0].get("case", "")
    df = df.sort_values(L["score_clin_short"], ascending=False).reset_index(drop=True)
    df[L["rank_clinical"]] = range(1, len(df) + 1)
    case_label = entries[0].get("case", "")
    if len(entries) > 1:
        case_label = f"{case_label} (avg ×{len(entries)})"
    return df, case_label


def build_ranking_narrative(
    ranking_df: pd.DataFrame,
    compare: dict,
    model_keys: list,
    lang: str = "en",
    use_gold: bool = False,
) -> list:
    """One short takeaway per model (max 2 lines) — only what explains the rank."""
    if ranking_df.empty:
        return []

    L = _L(lang)
    n = len(ranking_df)
    majority_label = compare.get("urgency_majority_label")
    majority_text = (
        t(URGENCY_META.get(majority_label, URGENCY_META[None])["label_key"], lang)
        if majority_label
        else None
    )

    items = []
    for _, row in ranking_df.iterrows():
        key = row["key"]
        cfg = MODEL_CONFIG.get(key, {})
        rank_cons = int(row[L["rank_consensus"]])
        score_cons = row[L["score_cons_rescaled"]]
        rank_clin = int(row[L["rank_clinical"]]) if use_gold and row.get(L["rank_clinical"]) else None
        score_clin = row.get(L["score_clin_short"])
        bullets = []

        if use_gold and rank_clin is not None and rank_clin != rank_cons:
            bullets.append(
                t(
                    "narrative.compact_dual",
                    lang,
                    rank_cons=rank_cons,
                    score_cons=score_cons,
                    rank_clin=rank_clin,
                    score_clin=score_clin,
                )
            )
        elif use_gold and rank_clin is not None and rank_clin == 1:
            bullets.append(t("narrative.compact_leader_clinical", lang, score=score_clin))
        elif not use_gold and rank_cons == 1:
            bullets.append(t("narrative.compact_leader", lang, score=score_cons))
        elif use_gold and rank_clin is not None and rank_clin == n and n > 1:
            bullets.append(
                t("narrative.compact_laggard", lang, rank=rank_clin, score=score_clin)
            )
        elif not use_gold and rank_cons == n and n > 1:
            bullets.append(t("narrative.compact_laggard", lang, rank=rank_cons, score=score_cons))
        elif use_gold and rank_clin is not None:
            bullets.append(t("narrative.compact_mid", lang, rank=rank_clin, score=score_clin))
        else:
            bullets.append(t("narrative.compact_mid", lang, rank=rank_cons, score=score_cons))

        u = compare.get("urgency", {}).get(key, {})
        u_label = u.get("label")
        if u_label and majority_label and u_label != majority_label:
            u_label_text = t(URGENCY_META.get(u_label, URGENCY_META[None])["label_key"], lang)
            bullets.append(
                t(
                    "narrative.compact_urgency_diff",
                    lang,
                    label=u_label_text,
                    majority=majority_text,
                )
            )

        display_rank = rank_clin if use_gold and rank_clin is not None else rank_cons
        display_score = score_clin if use_gold and score_clin is not None else score_cons
        tone = "strength" if display_rank == 1 else ("watch" if display_rank == n and n > 1 else "neutral")

        items.append(
            {
                "key": key,
                "rank": display_rank,
                "name": table_model_name(key),
                "icon": cfg.get("icon", "🔹"),
                "color": cfg.get("color", "#94a3b8"),
                "score": display_score,
                "bullets": bullets[:2],
                "tone": tone,
            }
        )

    return items


def build_score_explanations(
    compare: dict, model_keys: list, lang: str = "en", use_gold: bool = False
) -> list:
    """Full numeric breakdown: same weighted formulas as the ranking table."""
    names = {k: MODEL_CONFIG[k]["name"] for k in model_keys if k in MODEL_CONFIG}
    sem_available = compare.get("semantic_available", False)
    gold = compare.get("gold") if use_gold else None
    clinical_pairs = compare.get("clinical_dimension_pairs") or {}

    raw_cons = {k: consensus_raw_score(compare, k) for k in model_keys if k in MODEL_CONFIG}
    rescaled_cons = rescale_relative(raw_cons)

    items = []
    for key in model_keys:
        cfg = MODEL_CONFIG.get(key, {})
        if not cfg:
            continue

        cons_dims = _consensus_dimensions(compare, key)
        cons_raw = raw_cons.get(key, 0.0)
        cons_rescaled = rescaled_cons.get(key, 0.0)
        cons_formula = format_weighted_formula(cons_dims, CONSENSUS_DIM_WEIGHTS, lang)

        dim_pairs = {}
        for dim_key in ("diagnosis", "management", "summary"):
            dim_pairs[dim_key] = [
                {"other": names.get(j, j), "value": v}
                for j, v in ((clinical_pairs.get(dim_key) or {}).get(key) or {}).items()
                if v is not None
            ]

        entry = {
            "key": key,
            "name": cfg.get("name", key),
            "icon": cfg.get("icon", "🔹"),
            "color": cfg.get("color", "#94a3b8"),
            "cloud": cfg.get("cloud", False),
            "primary_text": compare.get("primary_diagnosis_text", {}).get(key, ""),
            "consensus": {
                "dimensions": cons_dims,
                "raw": cons_raw,
                "rescaled": cons_rescaled,
                "formula": cons_formula,
                "pairs": dim_pairs,
                "semantic_available": sem_available,
            },
            "privacy": privacy_score(cfg.get("cloud", False)),
            "urgency": compare.get("urgency", {}).get(key, {"label": None, "score": None}),
            "gold": None,
            "final_score": cons_rescaled,
            "consensus_score": cons_rescaled,
        }

        if gold:
            gold_dims = _gold_dimensions(gold, key)
            score_gold = _clinical_gold_score(gold, key)
            gold_formula = format_weighted_formula(gold_dims, GOLD_DIM_WEIGHTS, lang)
            entry["gold"] = {
                "dimensions": gold_dims,
                "formula": gold_formula,
                "semantic_available": gold.get("semantic_available", False),
                "score": score_gold,
                "grade_10": clinical_grade_10(score_gold),
                "gold_keywords": gold.get("gold_keywords", [])[:10],
            }
            entry["final_score"] = score_gold

        items.append(entry)

    items.sort(
        key=lambda e: e.get("gold", {}).get("score", 0) if gold else e["consensus"]["rescaled"],
        reverse=True,
    )
    return items


def build_leaderboard_df(history: list, lang: str = "en") -> pd.DataFrame:
    """Aggrega la cronologia dei round eseguiti in sessione in una classifica cumulativa."""
    L = _L(lang)
    if not history:
        return pd.DataFrame()

    wins: dict = {}
    totals: dict = {}
    counts: dict = {}
    for entry in history:
        wins[entry.get("winner_consensus_key") or entry.get("winner_key")] = (
            wins.get(entry.get("winner_consensus_key") or entry.get("winner_key"), 0) + 1
        )
        for key, score in _entry_scores(entry, "consensus").items():
            totals[key] = totals.get(key, 0.0) + (score or 0.0)
            counts[key] = counts.get(key, 0) + 1

    rows = []
    for key, cfg in MODEL_CONFIG.items():
        if key not in counts:
            continue
        avg_score = round(totals[key] / counts[key], 1) if counts.get(key) else 0.0
        rows.append(
            {
                "key": key,
                L["model"]: table_model_name(key),
                t("leaderboard.wins", lang): wins.get(key, 0),
                t("leaderboard.avg_score", lang): avg_score,
                t("leaderboard.rounds", lang): counts[key],
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            [t("leaderboard.wins", lang), t("leaderboard.avg_score", lang)],
            ascending=[False, False],
        ).reset_index(drop=True)
    return df
