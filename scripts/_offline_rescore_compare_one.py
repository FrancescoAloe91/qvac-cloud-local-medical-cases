"""Non-destructive OLD vs NEW offline rescore for one artifact."""
from __future__ import annotations

import json
from pathlib import Path

from benchmark.report import load_artifact, rescore_artifact_current_formula
from benchmark.scoring import graded_clinical_score

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "artifacts/owners/893e6a29cf690fbef4d6aee2/caseC-3f5bb3a7ef.json"
)
REPORT = ROOT / ".cursor" / "offline_rescore_caseC-3f5bb3a7ef.json"


def _rank_map(rows):
    out = {}
    for r in rows or []:
        k = r.get("key")
        if not k:
            continue
        out[k] = {
            "accuracy": r.get("accuracy"),
            "rank": r.get("rank"),
            "status": r.get("status")
            or ("ok" if r.get("accuracy") is not None else "n/a"),
            "coverage": r.get("coverage"),
            "quality": r.get("quality"),
            "discipline": r.get("discipline"),
        }
    return out


def _recompute_from_claims(art) -> dict:
    """Recompute composite from stored claim fields per question, then mean."""
    out = {}
    for j in art.judgments or []:
        secs = []
        for qs in j.question_scores or []:
            if qs.claim_coverage and qs.recall is not None:
                secs.append(
                    graded_clinical_score(
                        coverage=qs.recall,
                        quality=qs.quality if qs.quality is not None else 0.5,
                        discipline=qs.precision if qs.precision is not None else 1.0,
                    )
                )
        if secs:
            out[j.candidate_key] = round(sum(secs) / len(secs), 2)
        else:
            out[j.candidate_key] = None
    return out


def main() -> None:
    art = load_artifact(ARTIFACT)
    old_map = _rank_map(art.ranking)
    old_version = str(art.scoring_version or "")
    claim_composites = _recompute_from_claims(art)

    claim_deltas = {}
    for k, stored in ((k, old_map[k]["accuracy"]) for k in old_map):
        recomputed = claim_composites.get(k)
        if stored is not None and recomputed is not None:
            claim_deltas[k] = round(float(recomputed) - float(stored), 4)
        else:
            claim_deltas[k] = None

    scored = rescore_artifact_current_formula(art)
    new_rows = scored.get("ranking") or []
    new_map = _rank_map(new_rows)
    new_version = str(
        scored.get("scoring_version_stamp") or scored.get("formula") or ""
    )
    recovered = list(scored.get("recovered_keys") or [])
    unrecovered = list(scored.get("unrecovered_na") or [])

    per_model = []
    for k in sorted(set(old_map) | set(new_map)):
        oa = old_map.get(k, {}).get("accuracy")
        na = new_map.get(k, {}).get("accuracy")
        delta = None
        if oa is not None and na is not None:
            delta = round(float(na) - float(oa), 4)
        oc = {
            "coverage": old_map.get(k, {}).get("coverage"),
            "quality": old_map.get(k, {}).get("quality"),
            "discipline": old_map.get(k, {}).get("discipline"),
        }
        nc = {
            "coverage": new_map.get(k, {}).get("coverage"),
            "quality": new_map.get(k, {}).get("quality"),
            "discipline": new_map.get(k, {}).get("discipline"),
        }
        comp_deltas = {}
        for c in ("coverage", "quality", "discipline"):
            if oc.get(c) is not None and nc.get(c) is not None:
                comp_deltas[c] = round(float(nc[c]) - float(oc[c]), 4)
            else:
                comp_deltas[c] = None
        row = {
            "key": k,
            "old_accuracy": oa,
            "new_accuracy": na,
            "delta": delta,
            "old_rank": old_map.get(k, {}).get("rank"),
            "new_rank": new_map.get(k, {}).get("rank"),
            "old_components": oc,
            "new_components": nc,
            "component_deltas": comp_deltas,
            "claim_recomputed": claim_composites.get(k),
            "claim_minus_stored": claim_deltas.get(k),
        }
        per_model.append(row)

    report = {
        "artifact": str(ARTIFACT),
        "run_id": art.run_id,
        "run_status": art.run_status,
        "n_candidates": len(art.candidates or []),
        "scoring_version_before": old_version,
        "scoring_version_after": new_version,
        "preserved_stored_ranking": bool(scored.get("preserved_stored_ranking")),
        "formula": scored.get("formula"),
        "recovered_keys": recovered,
        "unrecovered_na": unrecovered,
        "artifact_rewritten": False,
        "per_model": per_model,
        "max_abs_delta": max(
            (abs(r["delta"]) for r in per_model if r["delta"] is not None),
            default=0.0,
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote comparison report: {REPORT}")


if __name__ == "__main__":
    main()
