"""Full offline rescore + OLD vs NEW + last-5 NEW means. No API calls."""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from benchmark.report import (
    _atomic_write,
    list_run_artifacts,
    load_artifact,
    rescore_artifact_current_formula,
    write_artifact,
)

# Require explicit owner — never default to a personal fingerprint.
_OWNER_ENV = (os.environ.get("OFFLINE_RESCORE_OWNER") or "").strip()
if not _OWNER_ENV:
    print(
        "Set OFFLINE_RESCORE_OWNER=artifacts/owners/<fingerprint> "
        "(no hardcoded default).",
        file=sys.stderr,
    )
    raise SystemExit(2)
OWNER = Path(_OWNER_ENV)
FOCUS_KEYS = [
    "chatgpt",
    "claude",
    "gemini",
    "qvac",
    "qvac_1_7b",
    "qvac_4b_q8",
    "local_phi",
    "local_gemma",
    "local_llama",
]
LABEL = {
    "chatgpt": "ChatGPT",
    "claude": "Claude",
    "gemini": "Gemini",
    "qvac": "MedPsy/QVAC",
    "qvac_1_7b": "QVAC 1.7B",
    "qvac_4b_q8": "QVAC 4B Q8",
    "local_phi": "Local Phi",
    "local_gemma": "Local Gemma",
    "local_llama": "Local Llama",
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def rank_map(rows):
    out = {}
    for r in rows or []:
        k = r.get("key")
        out[k] = {
            "accuracy": r.get("accuracy"),
            "rank": r.get("rank"),
            "status": r.get("status")
            or ("ok" if r.get("accuracy") is not None else "n/a"),
        }
    return out


def original_old_ranking(art):
    """Preserve true pre-rescore ranking across repeated offline passes."""
    off = (art.reproducibility or {}).get("offline_rescore") or {}
    stored = off.get("stored_ranking")
    if stored:
        return stored
    return list(art.ranking or [])


def fmt(v):
    if v is None:
        return "N/A"
    return f"{float(v):.2f}"


def main() -> None:
    comparisons = []
    written = []
    unrecovered_all = []
    recovered_all = []

    for path in list_run_artifacts(OWNER):
        try:
            art = load_artifact(path)
        except Exception:
            continue
        if not art.judgments and not art.ranking:
            continue
        old = original_old_ranking(art)
        scored = rescore_artifact_current_formula(art)
        new = scored["ranking"]
        clone = art.model_copy(deep=True)
        clone.ranking = new
        if scored.get("effective_judgments"):
            clone.judgments = list(scored["effective_judgments"])
        repro = dict(clone.reproducibility or {})
        repro["offline_rescore"] = {
            "formula": scored.get("formula") or "graded-clinical-v4",
            "preserved_stored_ranking": bool(scored.get("preserved_stored_ranking")),
            "recovered_keys": list(scored.get("recovered_keys") or []),
            "unrecovered_na": list(scored.get("unrecovered_na") or []),
            "stored_ranking": old,
        }
        clone.reproducibility = repro
        if scored.get("scoring_version_stamp") and not scored.get(
            "preserved_stored_ranking"
        ):
            clone.scoring_version = str(scored["scoring_version_stamp"])
        if scored.get("recovered_keys"):
            note = (clone.notes or "").strip()
            tag = "offline-recovered: " + ", ".join(scored["recovered_keys"])
            if tag not in note:
                clone.notes = (note + " | " + tag).strip(" |")
        write_artifact(clone, OWNER)
        if path.name != f"{clone.run_id}.json":
            _atomic_write(path, clone.model_dump_json(indent=2))
        written.append(clone.run_id)
        recovered_all.extend(scored.get("recovered_keys") or [])
        for u in scored.get("unrecovered_na") or []:
            unrecovered_all.append({"run_id": clone.run_id, **u})
        comparisons.append(
            {
                "run_id": clone.run_id,
                "case_id": clone.case_id,
                "finished_at": clone.finished_at,
                "cohort_id": clone.cohort_id or "",
                "batch_id": clone.batch_id or "",
                "mode": (clone.models_config or {}).get("mode") or "cloud",
                "stem_n": norm((clone.models_config or {}).get("case_stem")),
                "gold_n": norm(
                    str((clone.models_config or {}).get("gold_reference") or "")
                ),
                "old": old,
                "new": new,
                "recovered_keys": scored.get("recovered_keys") or [],
                "unrecovered_na": scored.get("unrecovered_na") or [],
                "keys_new": [r.get("key") for r in new],
            }
        )

    print(
        f"Persisted {len(written)} artifacts; "
        f"recovered slots={len(recovered_all)}; unrecovered={len(unrecovered_all)}"
    )

    c = [
        x
        for x in comparisons
        if x["case_id"] == "caseC" and x["mode"] != "local_only"
    ]

    def has_cloud_local(x):
        ks = set(x["keys_new"])
        return {"chatgpt", "claude", "gemini", "qvac"} <= ks and any(
            k.startswith("local_") for k in ks
        )

    c_full = [x for x in c if has_cloud_local(x)]
    c_full.sort(key=lambda x: x["finished_at"] or "", reverse=True)

    # Prefer the most recent homogeneous stem+gold group with N≥5
    # (largest among ties). Absolute older groups are not preferred.
    by_pair: dict = defaultdict(list)
    for x in c_full:
        by_pair[(x["stem_n"], x["gold_n"])].append(x)
    candidates = []
    for pair, items in by_pair.items():
        items = sorted(items, key=lambda z: z["finished_at"] or "", reverse=True)
        if len(items) >= 5:
            candidates.append((items[0]["finished_at"] or "", len(items), pair, items))
    if not candidates:
        # fallback: largest group of any size
        pair, items = max(by_pair.items(), key=lambda kv: len(kv[1]))
        items = sorted(items, key=lambda z: z["finished_at"] or "", reverse=True)
        candidates = [(items[0]["finished_at"] or "", len(items), pair, items)]
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    _, best_n, best_pair, cohort = candidates[0]
    print(
        f"Most recent homogeneous cloud+local gold cohort with N≥5: N={best_n}"
    )
    print(
        "runs:",
        [
            (x["run_id"], x["finished_at"], x["cohort_id"][:16] or "(legacy)")
            for x in cohort
        ],
    )

    last5 = cohort[:5]
    last5_ids = {r["run_id"] for r in last5}
    print("LAST5:", [x["run_id"] for x in last5])

    recent_any = [
        x for x in c_full if x["finished_at"] and x["finished_at"] >= "2026-07-27"
    ]
    print(
        "Since Jul27 cloud+local:",
        [
            (
                x["run_id"],
                x["finished_at"][:19],
                "cohort=" + (x["cohort_id"][:12] or "none"),
            )
            for x in recent_any
        ],
    )

    report_runs = []
    seen = set()
    for x in (
        recent_any
        + last5
        + [
            x
            for x in c_full
            if x["finished_at"] and x["finished_at"] >= "2026-07-26T20:00:00"
        ]
    ):
        if x["run_id"] in seen:
            continue
        seen.add(x["run_id"])
        report_runs.append(x)
    report_runs.sort(key=lambda x: x["finished_at"] or "", reverse=True)

    print("\n========== PER-RUN OLD vs NEW (caseC gold cloud+local, recent) ==========")
    for x in report_runs:
        om, nm = rank_map(x["old"]), rank_map(x["new"])
        in_l5 = x["run_id"] in last5_ids
        finished = (x["finished_at"] or "?")[:19]
        cohort_s = x["cohort_id"][:20] or "(legacy)"
        print(
            f"\n### {x['run_id']}  {finished}  cohort={cohort_s}  in_last5={in_l5}"
        )
        if x["recovered_keys"]:
            print("  recovered:", x["recovered_keys"])
        if x["unrecovered_na"]:
            print("  unrecovered:", x["unrecovered_na"])
        print(
            f"{'model':16} {'old%':>8} {'oR':>3} {'new%':>8} {'nR':>3} {'Δ':>7}"
        )
        for k in FOCUS_KEYS:
            if k not in om and k not in nm:
                continue
            oa = om.get(k, {}).get("accuracy")
            na = nm.get(k, {}).get("accuracy")
            orank = om.get(k, {}).get("rank")
            nrank = nm.get(k, {}).get("rank")
            delta = None
            if oa is not None and na is not None:
                delta = round(float(na) - float(oa), 2)
            d_s = "" if delta is None else f"{delta:+.2f}"
            print(
                f"{LABEL.get(k, k):16} {fmt(oa):>8} {str(orank or '—'):>3} "
                f"{fmt(na):>8} {str(nrank or '—'):>3} {d_s:>7}"
            )

    print("\n========== LAST 5 NEW MEANS (homogeneous stem+gold) ==========")
    print("N runs:", len(last5), "ids:", [x["run_id"] for x in last5])
    vals: dict = defaultdict(list)
    fail: dict = defaultdict(int)
    req: dict = defaultdict(int)
    for x in last5:
        nm = rank_map(x["new"])
        for k in FOCUS_KEYS:
            if k not in nm:
                continue
            req[k] += 1
            r = nm[k]
            if r["status"] == "ok" and r["accuracy"] is not None:
                vals[k].append(float(r["accuracy"]))
            else:
                fail[k] += 1

    print(
        f"{'model':16} {'mean_NEW':>9} {'n_valid':>8} {'n_fail':>7} "
        f"{'min':>7} {'max':>7}"
    )
    means = []
    for k in FOCUS_KEYS:
        if req[k] == 0:
            continue
        v = vals[k]
        mean = statistics.mean(v) if v else None
        means.append(
            (k, mean, len(v), fail[k], min(v) if v else None, max(v) if v else None)
        )
    means.sort(key=lambda t: (-1 if t[1] is None else -t[1], t[0]))
    for i, (k, mean, nv, nf, mn, mx) in enumerate(means, 1):
        print(
            f"{i:2}. {LABEL.get(k, k):14} {fmt(mean):>9} {nv:>8} {nf:>7} "
            f"{fmt(mn):>7} {fmt(mx):>7}"
        )

    batches = {x["batch_id"] for x in last5 if x["batch_id"]}
    print("batch_ids in last5:", batches or "(none — paired sensitivity N/A)")

    print(f"\n========== ALL N={len(cohort)} HOMOGENEOUS NEW MEANS ==========")
    vals = defaultdict(list)
    fail = defaultdict(int)
    req = defaultdict(int)
    for x in cohort:
        nm = rank_map(x["new"])
        for k in FOCUS_KEYS:
            if k not in nm:
                continue
            req[k] += 1
            r = nm[k]
            if r["status"] == "ok" and r["accuracy"] is not None:
                vals[k].append(float(r["accuracy"]))
            else:
                fail[k] += 1
    means_all = []
    for k in FOCUS_KEYS:
        if req[k] == 0:
            continue
        v = vals[k]
        mean = statistics.mean(v) if v else None
        means_all.append(
            (k, mean, len(v), fail[k], min(v) if v else None, max(v) if v else None)
        )
    means_all.sort(key=lambda t: (-1 if t[1] is None else -t[1], t[0]))
    for i, (k, mean, nv, nf, mn, mx) in enumerate(means_all, 1):
        print(
            f"{i:2}. {LABEL.get(k, k):14} {fmt(mean):>9} {nv:>8} {nf:>7} "
            f"{fmt(mn):>7} {fmt(mx):>7}"
        )

    print("\n========== UNRECOVERED N/A (offline) ==========")
    if not unrecovered_all:
        print("(none)")
    else:
        by_run = defaultdict(list)
        for u in unrecovered_all:
            by_run[u["run_id"]].append(u)
        for run_id, items in sorted(by_run.items(), reverse=True)[:30]:
            keys = [i.get("key") for i in items]
            reasons = sorted({(i.get("reason") or "")[:80] for i in items})
            print(f"{run_id}: {keys} :: {reasons}")
        print("total unrecovered slots", len(unrecovered_all))

    print("\n========== caseA/B written ==========")
    for case in ("caseA", "caseB"):
        xs = [x for x in comparisons if x["case_id"] == case]
        print(case, "n=", len(xs))
        for x in sorted(xs, key=lambda z: z["finished_at"] or "", reverse=True)[:5]:
            om, nm = rank_map(x["old"]), rank_map(x["new"])
            deltas = []
            for k in ("chatgpt", "claude", "gemini", "qvac"):
                if k in om or k in nm:
                    oa = om.get(k, {}).get("accuracy")
                    na = nm.get(k, {}).get("accuracy")
                    if oa is not None and na is not None:
                        deltas.append(f"{k}:{oa}->{na} ({na - oa:+.2f})")
            print(" ", x["run_id"], "; ".join(deltas))

    l5 = []
    vals = defaultdict(list)
    fail = defaultdict(int)
    req = defaultdict(int)
    for x in last5:
        nm = rank_map(x["new"])
        for k in FOCUS_KEYS:
            if k not in nm:
                continue
            req[k] += 1
            r = nm[k]
            if r["status"] == "ok" and r["accuracy"] is not None:
                vals[k].append(float(r["accuracy"]))
            else:
                fail[k] += 1
    for k in FOCUS_KEYS:
        if req[k] == 0:
            continue
        v = vals[k]
        mean = statistics.mean(v) if v else None
        l5.append(
            {
                "key": k,
                "label": LABEL.get(k, k),
                "mean": None if mean is None else round(mean, 2),
                "n_valid": len(v),
                "n_fail": fail[k],
                "min": None if not v else round(min(v), 2),
                "max": None if not v else round(max(v), 2),
            }
        )
    l5.sort(key=lambda r: (-1e9 if r["mean"] is None else -r["mean"], r["key"]))

    out = {
        "formula_new": (
            "graded-clinical-v4 (50% coverage + 35% quality + 15% discipline); "
            "v3 artifacts keep stored ranking when unclamped quality is unavailable"
        ),
        "formula_change_commits": [
            "0214d4c Replace demo scoring with graded clinical benchmark",
            "753973e Harden recovery and reference-relative scoring",
        ],
        "written": len(written),
        "last5_run_ids": [x["run_id"] for x in last5],
        "homogeneous_cohort_n": len(cohort),
        "homogeneous_cohort_ids": [x["run_id"] for x in cohort],
        "note": (
            "Largest recent homogeneous caseC custom gold cohort = same normalized "
            "stem+gold; official cohort_id empty on these runs (legacy). Last-5 means "
            "are exploratory under the current protocol. Jul27–28 cohort-stamped runs "
            "exist but N<5 per cohort_id."
        ),
        "last5_means_new": l5,
        "unrecovered_na": unrecovered_all,
        "comparisons": [
            {
                "run_id": x["run_id"],
                "case_id": x["case_id"],
                "finished_at": x["finished_at"],
                "cohort_id": x["cohort_id"],
                "old": x["old"],
                "new": x["new"],
                "recovered_keys": x["recovered_keys"],
                "unrecovered_na": x["unrecovered_na"],
            }
            for x in comparisons
        ],
    }
    (OWNER / "_offline_rescore_report.json").write_text(json.dumps(out, indent=2))
    print("\nWrote", OWNER / "_offline_rescore_report.json")


if __name__ == "__main__":
    main()
