#!/usr/bin/env python3
"""Smoke-test saved snapshots: bars visible, tables compact, margins sane.

NOT authoritative for public Clinical Composite rankings.
Uses the legacy ``lib.metrics`` consensus / rescale path (and
``OLD/legacy_dashboard``-era snapshots). Live Comprehension / Structured
tracks score via the DeepSeek R1 judge — do not regenerate marketing
screenshots from this script and present them as judge-track results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.charts import fig_clinical_ranking_bars, fig_consensus_ranking_bars
from lib.cloud_tiers import effective_tier_labels
from lib.metrics import _L, build_unified_ranking, prepare_model_table
from lib.session_store import ranking_df_from_snapshot


def _snapshot_payload(slot_data: dict) -> dict | None:
    if slot_data.get("ranking_records"):
        return slot_data
    runs = slot_data.get("runs")
    if isinstance(runs, list) and runs:
        return runs[-1]
    return None


def _ranking_for_snapshot(snapshot: dict, lang: str) -> "object":
    compare = snapshot.get("compare") or {}
    model_keys = snapshot.get("model_keys") or list((compare.get("diagnoses") or {}).keys())
    if compare and model_keys:
        return build_unified_ranking(compare, model_keys, lang)
    return ranking_df_from_snapshot(snapshot, lang)


def _check_case(case_id: str, snapshot: dict, lang: str, labels: dict) -> list[str]:
    errors: list[str] = []
    df = _ranking_for_snapshot(snapshot, lang)
    if df.empty:
        return [f"{case_id}/{lang}: empty ranking dataframe"]
    if "key" not in df.columns:
        errors.append(f"{case_id}/{lang}: missing key column")

    L = _L(lang)
    if L["model"] not in df.columns:
        errors.append(f"{case_id}/{lang}: missing model column")
    if L["score_cons_rescaled"] not in df.columns:
        errors.append(f"{case_id}/{lang}: missing score column")

    fig = fig_consensus_ranking_bars(df, lang, tier_labels=labels, height=280)
    if not fig.data:
        errors.append(f"{case_id}/{lang}: consensus chart has no traces")
    else:
        margin_l = int(fig.layout.margin.l or 0)
        margin_r = int(fig.layout.margin.r or 0)
        plot_w = 450 - margin_l - margin_r
        if plot_w < 120:
            errors.append(f"{case_id}/{lang}: plot too narrow ({plot_w}px) margin_l={margin_l}")
        xs = list(fig.data[0].x)
        if not xs or max(float(v) for v in xs) <= 0:
            errors.append(f"{case_id}/{lang}: consensus bars missing or zero")

    display = prepare_model_table(df, lang, labels)
    if len(display) != len(df):
        errors.append(f"{case_id}/{lang}: table row count mismatch")
    if L["version"] not in display.columns:
        errors.append(f"{case_id}/{lang}: table missing version column")

    if snapshot.get("use_gold") and L["score_clin_short"] in df.columns:
        fig2 = fig_clinical_ranking_bars(df, lang, tier_labels=labels, height=280)
        if fig2.data:
            xs2 = list(fig2.data[0].x)
            if not xs2:
                errors.append(f"{case_id}/{lang}: clinical bars missing")

    return errors


def main() -> int:
    path = ROOT / ".case_snapshots.json"
    if not path.is_file():
        print("SKIP: no .case_snapshots.json")
        return 0

    slots = json.loads(path.read_text(encoding="utf-8"))
    labels = effective_tier_labels({})
    failures: list[str] = []

    for case_id, slot_data in slots.items():
        snapshot = _snapshot_payload(slot_data) if isinstance(slot_data, dict) else None
        if not snapshot:
            continue
        for lang in ("en", "it"):
            failures.extend(_check_case(case_id, snapshot, lang, labels))

    if failures:
        print("FAILED:")
        for f in failures:
            print(" -", f)
        return 1

    print(f"OK: verified saved cases in EN + IT ({len(slots)} slot(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
