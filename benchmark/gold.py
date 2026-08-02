"""Gold-only reference extraction, validation, and cohort identity."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from benchmark.schema import ConfirmedGold, GoldClaim, GoldSection, ModelCallMeta

SECTION_IDS = ("diagnosis", "tests", "urgency", "safety", "plan")
EXTRACTION_PROMPT_VERSION = "gold-extract-v2"
SCORING_VERSION = "graded-clinical-v4"
LOCAL_QNA_EXTRACTOR_MODEL = "local/qna-sections-v1"

# Default-pack / New-case Q&A gold: Q1 [diagnosis] … A1: … through Q5 [plan] / A5:.
_QNA_NUM_TO_SECTION = {
    1: "diagnosis",
    2: "tests",
    3: "urgency",
    4: "safety",
    5: "plan",
}
_QNA_ANSWER_RE = re.compile(
    r"(?P<head>Q(?P<num>[1-5])\s*\[(?P<label>[^\]]+)\]\s*:?[^\n]*\n)"
    r"A(?P=num)\s*:\s*(?P<answer>.+?)(?=\n\nQ[1-5]\s*\[|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _normalized(text: str) -> str:
    """NFC + collapse whitespace + casefold — matching only; never invents text."""
    collapsed = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "").strip())
    return collapsed.casefold()


def source_quote_is_verbatim(raw_text: str, quote: str) -> bool:
    """True when quote is a contiguous substring of raw after matching normalize."""
    q = _normalized(quote)
    return bool(q) and q in _normalized(raw_text)


def _quotes_overlap_substantially(a: str, b: str) -> bool:
    """True when one non-trivial quote contains the other as a contiguous span."""
    if not a or not b or a == b:
        return False
    # Ignore tiny fragments — overlap checks target substantial clinical quotes.
    if min(len(a), len(b)) < 12:
        return False
    return a in b or b in a


def _quote_reject_message(claim_id: str, quote: str) -> str:
    preview = (quote or "").strip().replace("\n", " ")
    if len(preview) > 140:
        preview = preview[:137] + "..."
    return (
        f"Claim {claim_id} has no verbatim source quote; extractor output rejected. "
        f"Rejected quote (paraphrase not allowed): {preview!r}. "
        f"Fix: set source_quote to an exact contiguous substring of the raw reference "
        f"(matching ignores case/NFC/whitespace only — wording must appear in the text)."
    )


def format_prepare_error(exc: BaseException) -> str:
    """User-facing Prepare error with an actionable next step."""
    msg = str(exc).strip() or f"{type(exc).__name__} during Prepare"
    lower = msg.casefold()
    if "openrouter" in lower and ("key" in lower or "api" in lower or "auth" in lower):
        return (
            f"{msg} — paste a full OpenRouter key (sk-or-v1-…) in the sidebar, "
            f"then click Prepare reference again."
        )
    if "too short" in lower:
        return f"{msg} — paste a longer clinical reference (≥40 characters), then retry Prepare."
    if "complete json" in lower or "empty response" in lower or "json object" in lower:
        return (
            f"{msg} — the extractor returned incomplete JSON (often a timeout/truncation). "
            f"Click Prepare reference again; if it persists, shorten the reference slightly "
            f"or check the OpenRouter key/credits."
        )
    if "overlapping source quote" in lower or "duplicate source quote" in lower:
        return (
            f"{msg} — click Prepare reference again (auto-dedupe retries nested quotes). "
            f"If it persists, use Q1[diagnosis]/A1 … Q5[plan]/A5 form, or delete "
            f"overlapping claims after a successful Prepare."
        )
    if "verbatim source quote" in lower or "paraphrase" in lower:
        return (
            f"{msg} — click Prepare reference again. Prefer Q1–A5 labeled answers so "
            f"Prepare can extract locally without paraphrases."
        )
    if "missing extracted section" in lower or "no claims with verbatim" in lower:
        return (
            f"{msg} — ensure the reference covers diagnosis, tests, urgency, safety, "
            f"and plan (Q1–A5 form works best), then retry Prepare."
        )
    return (
        f"{msg} — click Prepare reference to retry. "
        f"Check the OpenRouter key if the error mentions API/auth/network."
    )


def _validate_source_quotes(
    raw_text: str,
    sections: Mapping[str, GoldSection],
) -> None:
    """Reject empty, non-verbatim, duplicate, or overlapping substantial quotes."""
    raw_norm = _normalized(raw_text)
    seen: list[str] = []
    for section in sections.values():
        for claim in section.claims:
            quote = _normalized(claim.source_quote)
            if not quote:
                raise ValueError(f"Claim {claim.id} has an empty source quote")
            if quote not in raw_norm:
                raise ValueError(_quote_reject_message(claim.id, claim.source_quote))
            if quote in seen:
                raise ValueError(
                    f"Duplicate source quote across scoring claims: {claim.source_quote}"
                )
            for prior in seen:
                if _quotes_overlap_substantially(quote, prior):
                    raise ValueError(
                        f"Overlapping source quote across scoring claims: {claim.source_quote}"
                    )
            seen.append(quote)


def _drop_nested_overlapping_claims(
    sections: Mapping[str, GoldSection],
) -> Dict[str, GoldSection]:
    """Keep earlier claims; drop later duplicates / nested-overlapping quotes."""
    kept_quotes: list[str] = []
    out: Dict[str, GoldSection] = {}
    for section_id in SECTION_IDS:
        section = sections[section_id]
        kept: list[GoldClaim] = []
        for claim in section.claims:
            quote = _normalized(claim.source_quote)
            if not quote:
                continue
            if quote in kept_quotes:
                continue
            if any(_quotes_overlap_substantially(quote, prior) for prior in kept_quotes):
                continue
            kept_quotes.append(quote)
            kept.append(claim)
        out[section_id] = section.model_copy(update={"claims": kept})
    return out


def try_extract_qna_sections(raw_text: str) -> Optional[Dict[str, GoldSection]]:
    """Deterministic Prepare for Q1[section]/A1 … Q5[section]/A5 references.

    Returns None when the reference is not a complete five-answer Q&A pack so
    callers can fall back to the LLM extractor. All source_quote values are
    exact contiguous substrings of ``raw_text`` (no paraphrase risk).
    """
    text = raw_text or ""
    if len(text.strip()) < 40:
        return None
    found: Dict[str, str] = {}
    for match in _QNA_ANSWER_RE.finditer(text):
        num = int(match.group("num"))
        section_id = _QNA_NUM_TO_SECTION.get(num)
        if section_id is None:
            continue
        answer = (match.group("answer") or "").strip()
        if not answer:
            continue
        # Prefer labeled mapping when present, but number wins for pack format.
        found[section_id] = answer
    if any(section_id not in found for section_id in SECTION_IDS):
        return None
    raw_norm = _normalized(text)
    sections: Dict[str, GoldSection] = {}
    for section_id in SECTION_IDS:
        answer = found[section_id]
        if _normalized(answer) not in raw_norm:
            return None
        summary = answer if len(answer) <= 240 else answer[:237].rstrip() + "..."
        sections[section_id] = GoldSection(
            summary=summary,
            claims=[
                GoldClaim(
                    id=f"{section_id}-1",
                    text=answer,
                    source_quote=answer,
                    critical=False,
                )
            ],
        )
    try:
        _validate_source_quotes(text, sections)
    except ValueError:
        return None
    return sections


def looks_like_qna_reference(raw_text: str) -> bool:
    """True when Prepare can run the local Q1–A5 path (no OpenRouter call)."""
    return try_extract_qna_sections(raw_text) is not None


def extract_json_object(raw: str) -> Mapping[str, Any]:
    """Parse JSON even when a provider wraps it in Markdown or commentary."""
    text = (raw or "").strip().lstrip("\ufeff")
    if not text:
        raise ValueError("Gold extractor returned an empty response")

    try:
        payload = json.loads(text)
        if isinstance(payload, Mapping):
            return payload
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                payload, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                return payload
    raise ValueError("Gold extractor did not return a complete JSON object")


def extraction_messages(raw_text: str) -> list[dict[str, str]]:
    schema = {
        "sections": {
            section: {
                "summary": "faithful concise reorganization",
                "claims": [
                    {
                        "id": f"{section}-1",
                        "text": "one atomic clinical claim",
                        "source_quote": "exact verbatim quote from REFERENCE",
                        "critical": "boolean; true only for explicit safety-critical facts",
                    }
                ],
            }
            for section in SECTION_IDS
        }
    }
    return [
        {
            "role": "system",
            "content": (
                "You reorganize a user-supplied clinical reference; you do not diagnose. "
                "Extract only facts explicitly present in the source. Never infer, complete, "
                "correct, or add medical information. Every claim requires an exact verbatim "
                "source_quote copied character-for-character from REFERENCE (a contiguous "
                "substring). Do not paraphrase source_quote. Prefer fewer claims with real "
                "quotes over extra invented claims. Claims must not duplicate or nest inside "
                "each other (no overlapping source_quote spans). If a section is absent, "
                "return an empty summary and claims. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"PROMPT VERSION: {EXTRACTION_PROMPT_VERSION}\n"
                f"REFERENCE:\n{raw_text.strip()}\n\n"
                f"Required JSON shape:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def quote_repair_messages(
    raw_text: str,
    prior_payload: Mapping[str, Any],
    failures: List[str],
) -> list[dict[str, str]]:
    """Ask the extractor to fix only source_quote fields that failed verbatim checks."""
    return [
        {
            "role": "system",
            "content": (
                "You repair gold extraction JSON. Keep section structure and claim ids. "
                "For each failed claim, replace source_quote with an exact contiguous "
                "substring copied from REFERENCE (never paraphrase). If no verbatim quote "
                "exists for a claim, remove that claim. Remove duplicate or nested-"
                "overlapping source_quote claims (keep the most specific non-nested "
                "quote). Keep each section nonempty when the reference supports it. "
                "Return JSON only with the same shape."
            ),
        },
        {
            "role": "user",
            "content": (
                f"PROMPT VERSION: {EXTRACTION_PROMPT_VERSION}-quote-repair\n"
                f"REFERENCE:\n{raw_text.strip()}\n\n"
                f"VALIDATION FAILURES:\n"
                + "\n".join(f"- {item}" for item in failures)
                + "\n\nPRIOR JSON:\n"
                + json.dumps(prior_payload, ensure_ascii=False, indent=2)
            ),
        },
    ]


def _resolve_claim_quote(
    raw_norm: str,
    claim: GoldClaim,
) -> Optional[str]:
    """Return a strip()'d candidate that matches raw after normalize, else None."""
    for candidate in (claim.source_quote, claim.text):
        q = _normalized(candidate)
        if q and q in raw_norm:
            return (candidate or "").strip()
    return None


def _sections_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_sections = payload.get("sections")
    # Some providers honor the requested content but omit the outer
    # {"sections": ...} wrapper. Accept that harmless schema variation.
    if not isinstance(raw_sections, Mapping) and all(
        section_id in payload for section_id in SECTION_IDS
    ):
        raw_sections = payload
    if not isinstance(raw_sections, Mapping):
        raise ValueError("Extractor response has no sections object")
    return raw_sections


def parse_extraction(
    raw_text: str,
    payload: Mapping[str, Any],
    *,
    drop_invalid_claims: bool = False,
) -> Dict[str, GoldSection]:
    """Strictly validate source-linked sections proposed by the extractor."""
    raw_norm = _normalized(raw_text)
    raw_sections = _sections_mapping(payload)

    sections: Dict[str, GoldSection] = {}
    dropped: list[str] = []
    for section_id in SECTION_IDS:
        item = raw_sections.get(section_id)
        if not isinstance(item, Mapping):
            raise ValueError(f"Missing extracted section: {section_id}")
        claims_raw = item.get("claims")
        if not isinstance(claims_raw, list):
            raise ValueError(f"Invalid claims list: {section_id}")
        claims: list[GoldClaim] = []
        seen: set[str] = set()
        for index, claim_raw in enumerate(claims_raw, 1):
            if not isinstance(claim_raw, Mapping):
                raise ValueError(f"Invalid claim in {section_id}")
            claim = GoldClaim.model_validate(claim_raw)
            if not claim.id:
                claim.id = f"{section_id}-{index}"
            if claim.id in seen:
                raise ValueError(f"Duplicate claim id: {claim.id}")
            seen.add(claim.id)
            resolved = _resolve_claim_quote(raw_norm, claim)
            if resolved is None:
                if drop_invalid_claims:
                    dropped.append(claim.id)
                    continue
                raise ValueError(_quote_reject_message(claim.id, claim.source_quote))
            claim.source_quote = resolved
            if not claim.text.strip():
                if drop_invalid_claims:
                    dropped.append(claim.id)
                    continue
                raise ValueError(f"Claim {claim.id} is empty")
            # The extractor may segment but cannot rewrite scored meaning or assign weight.
            claim.text = claim.source_quote.strip()
            claim.critical = False
            claims.append(claim)
        if drop_invalid_claims and not claims and claims_raw:
            raise ValueError(
                f"Section {section_id} has no claims with verbatim source_quote after "
                f"dropping invalid quotes ({', '.join(dropped) or 'none'}). "
                f"Add exact substrings from the raw reference for this section."
            )
        sections[section_id] = GoldSection(
            summary=str(item.get("summary") or "").strip(),
            claims=claims,
        )
    if drop_invalid_claims:
        sections = _drop_nested_overlapping_claims(sections)
        emptied = [
            sid
            for sid in SECTION_IDS
            if not sections[sid].claims
            and isinstance(raw_sections.get(sid), Mapping)
            and isinstance((raw_sections.get(sid) or {}).get("claims"), list)
            and (raw_sections.get(sid) or {}).get("claims")
        ]
        if emptied:
            raise ValueError(
                f"Section {emptied[0]} has no claims with verbatim source_quote after "
                f"dropping invalid/overlapping quotes. "
                f"Add exact substrings from the raw reference for this section."
            )
    _validate_source_quotes(raw_text, sections)
    return sections


def collect_quote_failures(
    raw_text: str, payload: Mapping[str, Any]
) -> List[str]:
    """List per-claim quote failures without raising (for repair prompts / UI)."""
    failures: list[str] = []
    try:
        raw_sections = _sections_mapping(payload)
    except ValueError as exc:
        return [str(exc)]
    raw_norm = _normalized(raw_text)
    seen: list[str] = []
    for section_id in SECTION_IDS:
        item = raw_sections.get(section_id)
        if not isinstance(item, Mapping):
            failures.append(f"Missing extracted section: {section_id}")
            continue
        claims_raw = item.get("claims")
        if not isinstance(claims_raw, list):
            failures.append(f"Invalid claims list: {section_id}")
            continue
        for index, claim_raw in enumerate(claims_raw, 1):
            if not isinstance(claim_raw, Mapping):
                failures.append(f"Invalid claim in {section_id}")
                continue
            claim = GoldClaim.model_validate(claim_raw)
            if not claim.id:
                claim.id = f"{section_id}-{index}"
            resolved = _resolve_claim_quote(raw_norm, claim)
            if resolved is None:
                failures.append(_quote_reject_message(claim.id, claim.source_quote))
                continue
            quote = _normalized(resolved)
            if quote in seen:
                failures.append(
                    f"Duplicate source quote across scoring claims: {claim.source_quote}"
                )
                continue
            if any(_quotes_overlap_substantially(quote, prior) for prior in seen):
                failures.append(
                    f"Overlapping source quote across scoring claims: {claim.source_quote}"
                )
                continue
            seen.append(quote)
    return failures


def confirmed_gold(
    *,
    raw_text: str,
    sections: Mapping[str, GoldSection | Mapping[str, Any]],
    extraction_model: str,
    extraction_cost_usd: float = 0.0,
) -> ConfirmedGold:
    """Create the frozen contract; all five sections must be assessable."""
    for section_id in SECTION_IDS:
        if sections.get(section_id) is None:
            raise ValueError(f"Complete and confirm section: {section_id}")
    parsed = assign_deterministic_claim_ids(sections)
    for section_id, section in parsed.items():
        if not section.summary.strip() or not section.claims:
            raise ValueError(f"Complete and confirm section: {section_id}")
    _validate_source_quotes(raw_text, parsed)
    return ConfirmedGold(
        raw_text=raw_text.strip(),
        sections=parsed,
        confirmed_at=datetime.now(timezone.utc).isoformat(),
        extraction_model=extraction_model,
        extraction_prompt_version=EXTRACTION_PROMPT_VERSION,
        extraction_cost_usd=round(float(extraction_cost_usd or 0.0), 8),
    )


def gold_json(gold: ConfirmedGold) -> str:
    return gold.model_dump_json(indent=2)


def load_confirmed_gold(value: str | Mapping[str, Any] | ConfirmedGold) -> ConfirmedGold:
    try:
        if isinstance(value, ConfirmedGold):
            gold = value.model_copy(deep=True)
        elif isinstance(value, Mapping):
            gold = ConfirmedGold.model_validate(value)
        else:
            text = (value or "").strip()
            if not text:
                raise ValueError("A confirmed five-section reference is required")
            gold = ConfirmedGold.model_validate_json(text)
    except Exception as exc:
        raise ValueError(
            "Gold must be the confirmed five-section JSON contract, not unreviewed prose"
        ) from exc
    for section in gold.sections.values():
        for claim in section.claims:
            claim.text = claim.source_quote.strip()
            claim.critical = False
    try:
        _validate_source_quotes(gold.raw_text, gold.sections)
    except ValueError as exc:
        raise ValueError(
            f"Gold contains an empty, non-verbatim, duplicate, or overlapping source quote: {exc}"
        ) from exc
    return gold


def case_family_key(*, case_stem: str, reference_raw: str) -> str:
    """Stable key for clinical case + free-form reference (before claim splits).

    Does **not** include claim IDs, section splits, extractor metadata, or
    ``confirmed_at``. Same pasted case+reference → same family; different
    Prepare/Confirm contracts remain separate cohorts under that family.
    """
    canonical = json.dumps(
        {
            "case_stem": _normalized(case_stem),
            "reference_raw": _normalized(reference_raw),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def scoring_contract_dump(gold: ConfirmedGold) -> Dict[str, Any]:
    """Gold fields that affect scoring / cohort identity.

    Section ``summary`` is display-only (never sent to the judge) and is
    excluded so editing summaries cannot silently split cohorts.
    ``extraction_cost_usd`` is also excluded — cost metadata must not change
    scoring identity.
    """
    sections: Dict[str, Any] = {}
    for section_id in SECTION_IDS:
        section = gold.sections[section_id]
        # Sort claims by id so reorder-without-id-change is identity-stable;
        # confirm assigns deterministic ids from quote order.
        claims = sorted(
            (
                {
                    "id": c.id,
                    "text": (c.source_quote or c.text or "").strip(),
                    "source_quote": (c.source_quote or "").strip(),
                    "critical": False,
                }
                for c in section.claims
            ),
            key=lambda row: row["id"],
        )
        sections[section_id] = {"claims": claims}
    return {
        "raw_text": _normalized(gold.raw_text),
        "sections": sections,
        "extraction_model": gold.extraction_model,
        "extraction_prompt_version": gold.extraction_prompt_version,
    }


def assign_deterministic_claim_ids(
    sections: Mapping[str, GoldSection | Mapping[str, Any]],
) -> Dict[str, GoldSection]:
    """Assign stable ``{section}-{n}`` ids at Confirm (final contract only)."""
    out: Dict[str, GoldSection] = {}
    for section_id in SECTION_IDS:
        item = sections[section_id]
        section = item if isinstance(item, GoldSection) else GoldSection.model_validate(item)
        renumbered = []
        for index, claim in enumerate(section.claims, 1):
            renumbered.append(
                claim.model_copy(
                    update={
                        "id": f"{section_id}-{index}",
                        "text": claim.source_quote.strip(),
                        "critical": False,
                    }
                )
            )
        out[section_id] = section.model_copy(update={"claims": renumbered})
    return out


# Pack revisions below this threshold are omitted from cohort_id so stamped
# rev-3 artifacts keep the same hash as pre-stamp History (missing field).
# Future pack bumps (≥4) enter the hash and split cohorts honestly.
COHORT_HASH_PACK_REVISION_FROM = 4


def cohort_id(
    *,
    case_stem: str,
    gold: ConfirmedGold,
    prompt_version: str,
    model_config: Mapping[str, Any],
    benchmark_track: str,
    scoring_version: str = SCORING_VERSION,
    pack_revision: int | None = None,
) -> str:
    payload: Dict[str, Any] = {
        "case_stem": _normalized(case_stem),
        "gold": scoring_contract_dump(gold),
        "scoring_version": str(scoring_version or SCORING_VERSION),
        "prompt_version": prompt_version,
        "models": model_config,
        "track": benchmark_track,
    }
    try:
        pr = int(pack_revision) if pack_revision is not None else None
    except (TypeError, ValueError):
        pr = None
    if pr is not None and pr >= COHORT_HASH_PACK_REVISION_FROM:
        payload["pack_revision"] = pr
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def execution_cohort_id(
    *,
    case_stem: str,
    gold: ConfirmedGold,
    prompt_version: str,
    benchmark_track: str,
    candidates: Sequence[Any],
    judgments: Sequence[Any],
    scoring_version: str = SCORING_VERSION,
) -> str:
    """Hash result-affecting execution facts (actual routes + digests)."""
    cand_rows = []
    for c in candidates:
        meta = getattr(c, "meta", None)
        attempts = list(getattr(meta, "paid_attempts", None) or [])
        cand_rows.append(
            {
                "key": getattr(c, "candidate_key", ""),
                "requested_model": getattr(meta, "requested_model", None)
                or getattr(meta, "model", ""),
                "routed_model": getattr(meta, "routed_model", None)
                or getattr(meta, "model", ""),
                "routed_provider": getattr(meta, "routed_provider", None)
                or getattr(meta, "provider", ""),
                "temperature": getattr(meta, "temperature", None),
                "top_k": getattr(meta, "top_k", None),
                "top_p": getattr(meta, "top_p", None),
                "seed": getattr(meta, "seed", None),
                "gguf_sha256": getattr(meta, "gguf_sha256", "") or "",
                "device": getattr(meta, "device", "") or "",
                "gpu_layers": getattr(meta, "gpu_layers", None),
                "ctx_size": getattr(meta, "ctx_size", None),
                "predict": getattr(meta, "predict", None),
                "configuration_deviation": bool(
                    getattr(meta, "configuration_deviation", False)
                ),
                "paid_attempts": [
                    {
                        "model": a.get("model"),
                        "provider": a.get("provider") or a.get("routed_provider"),
                        "routed_model": a.get("routed_model"),
                    }
                    for a in attempts
                    if isinstance(a, dict)
                ],
            }
        )
    cand_rows.sort(key=lambda r: str(r.get("key") or ""))
    judge_rows = []
    for j in judgments:
        meta = getattr(j, "judge_meta", None)
        judge_rows.append(
            {
                "key": getattr(j, "candidate_key", ""),
                "judge_model": getattr(j, "judge_model", ""),
                "primary_judge_model": getattr(j, "primary_judge_model", ""),
                "routed_model": getattr(meta, "routed_model", None)
                or getattr(meta, "model", ""),
                "routed_provider": getattr(meta, "routed_provider", None)
                or getattr(meta, "provider", ""),
            }
        )
    judge_rows.sort(key=lambda r: str(r.get("key") or ""))
    canonical = json.dumps(
        {
            "case_stem": _normalized(case_stem),
            "gold": scoring_contract_dump(gold),
            "scoring_version": scoring_version,
            "prompt_version": prompt_version,
            "track": benchmark_track,
            "candidates": cand_rows,
            "judges": judge_rows,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_strict_track(benchmark_track: str) -> bool:
    return str(benchmark_track or "").strip() == "strict_controlled"


def uses_controlled_sampling(benchmark_track: str) -> bool:
    return str(benchmark_track or "").strip() in {
        "controlled",
        "strict_controlled",
        "best_effort",
    }


def track_ui_routing_blurb(benchmark_track: str) -> str:
    """Roster caption fragment: fallbacks + sampling by track (no Streamlit)."""
    track = str(benchmark_track or "").strip() or "controlled"
    if is_strict_track(track):
        return (
            "pinned prefer-order; OpenRouter fallbacks OFF "
            "(strict · route miss → N/A) · not bit-identical backends. "
            f"Track **{track}** · temp 0.2 (strict_controlled sampling)."
        )
    if uses_controlled_sampling(track):
        return (
            "pinned prefer-order; OpenRouter fallbacks remain on · "
            "not bit-identical backends. "
            f"Track **{track}** · temp 0.2 "
            "(controlled ≠ stock web/API defaults)."
        )
    return (
        "pinned prefer-order; native provider defaults · "
        "not bit-identical backends. "
        f"Track **{track}** · native_defaults "
        "(separate cohort; never pooled with controlled)."
    )


def _accumulate_extract_meta(
    prior: ModelCallMeta, nxt: ModelCallMeta, *, role: str
) -> ModelCallMeta:
    attempts = list(prior.paid_attempts or [])
    attempts.append(
        {
            "role": role,
            "model": nxt.model,
            "cost_usd": float(nxt.cost_usd or 0.0),
            "prompt_tokens": int(nxt.prompt_tokens or 0),
            "completion_tokens": int(nxt.completion_tokens or 0),
            "error": nxt.error or "",
        }
    )
    return prior.model_copy(
        update={
            "cost_usd": round(
                float(prior.cost_usd or 0.0) + float(nxt.cost_usd or 0.0), 8
            ),
            "prompt_tokens": int(prior.prompt_tokens or 0)
            + int(nxt.prompt_tokens or 0),
            "completion_tokens": int(prior.completion_tokens or 0)
            + int(nxt.completion_tokens or 0),
            "paid_attempts": attempts,
            "retry_count": max(0, int(prior.retry_count or 0) + 1),
        }
    )


def _local_qna_meta() -> ModelCallMeta:
    return ModelCallMeta(
        model=LOCAL_QNA_EXTRACTOR_MODEL,
        provider="local",
        requested_model=LOCAL_QNA_EXTRACTOR_MODEL,
        routed_model=LOCAL_QNA_EXTRACTOR_MODEL,
        cost_usd=0.0,
        prompt_tokens=0,
        completion_tokens=0,
    )


def extract_with_chat(
    raw_text: str,
    *,
    model: str,
    chat: Callable[..., Tuple[str, Any]],
    api_key: Optional[str] = None,
) -> Tuple[Dict[str, GoldSection], Any]:
    """Extract source-linked sections via local Q1–A5 parse or pinned chat.

    Order:
    1. Deterministic Q1[diagnosis]/A1 … Q5[plan]/A5 local parse (no API).
    2. Pinned extractor chat.
    3. One quote-repair chat on verbatim/overlap failures.
    4. Drop remaining invalid / nested-overlapping claims.
    5. Local Q1–A5 fallback if the reference matches that shape.
    """
    if len(raw_text.strip()) < 40:
        raise ValueError("Reference is too short to extract safely")

    local = try_extract_qna_sections(raw_text)
    if local is not None:
        return local, _local_qna_meta()

    # Long clinical refs (Case 6/7 style) need headroom; truncation → incomplete JSON.
    max_tokens = 8000 if len(raw_text.strip()) >= 1200 else 4000
    kwargs: Dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "timeout": 180.0,
    }
    if api_key is not None:
        kwargs["api_key"] = api_key

    raw, meta = chat(model, extraction_messages(raw_text), **kwargs)
    if getattr(meta, "error", None):
        raise RuntimeError(f"Gold extraction failed: {meta.error}")

    try:
        payload = extract_json_object(raw)
    except ValueError as json_exc:
        # One retry when the model truncated / wrapped JSON incompletely.
        retry_raw, retry_meta = chat(
            model,
            extraction_messages(raw_text)
            + [
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not a complete JSON object. "
                        "Return ONLY the full JSON object with all five sections."
                    ),
                }
            ],
            **kwargs,
        )
        if getattr(retry_meta, "error", None):
            raise RuntimeError(
                f"Gold extraction failed after JSON retry: {retry_meta.error}. "
                f"Primary issue: {json_exc}"
            ) from json_exc
        if isinstance(meta, ModelCallMeta) and isinstance(retry_meta, ModelCallMeta):
            meta = _accumulate_extract_meta(meta, retry_meta, role="gold_json_retry")
        else:
            meta = retry_meta
        payload = extract_json_object(retry_raw)

    def _qna_or_raise(exc: BaseException) -> Tuple[Dict[str, GoldSection], Any]:
        qna_fallback = try_extract_qna_sections(raw_text)
        if qna_fallback is not None:
            return qna_fallback, meta
        raise exc

    try:
        return parse_extraction(raw_text, payload), meta
    except ValueError as first_exc:
        failures = collect_quote_failures(raw_text, payload) or [str(first_exc)]
        quote_related = any(
            "source quote" in f.casefold()
            or "verbatim" in f.casefold()
            or "overlapping" in f.casefold()
            or "duplicate" in f.casefold()
            for f in failures
        ) or any(
            token in str(first_exc).casefold()
            for token in (
                "source quote",
                "verbatim",
                "overlapping",
                "duplicate",
            )
        )
        if not quote_related:
            # Non-quote schema errors: try drop path, then local QnA if shape matches.
            try:
                return parse_extraction(
                    raw_text, payload, drop_invalid_claims=True
                ), meta
            except ValueError:
                return _qna_or_raise(first_exc)

        repair_raw, repair_meta = chat(
            model, quote_repair_messages(raw_text, payload, failures), **kwargs
        )
        if getattr(repair_meta, "error", None):
            try:
                return parse_extraction(
                    raw_text, payload, drop_invalid_claims=True
                ), meta
            except ValueError:
                return _qna_or_raise(
                    RuntimeError(
                        f"Gold quote repair failed: {repair_meta.error}. "
                        f"Primary issue: {first_exc}"
                    )
                )
        if isinstance(meta, ModelCallMeta) and isinstance(repair_meta, ModelCallMeta):
            meta = _accumulate_extract_meta(
                meta, repair_meta, role="gold_quote_repair"
            )
        else:
            # Preserve whatever the transport returned when not ModelCallMeta.
            meta = repair_meta

        try:
            repair_payload = extract_json_object(repair_raw)
        except ValueError as repair_json_exc:
            try:
                return parse_extraction(
                    raw_text, payload, drop_invalid_claims=True
                ), meta
            except ValueError:
                return _qna_or_raise(
                    ValueError(
                        f"{repair_json_exc} (after quote-repair JSON parse failed; "
                        f"primary issue: {first_exc})"
                    )
                )

        try:
            return parse_extraction(raw_text, repair_payload), meta
        except ValueError as repair_exc:
            for candidate in (repair_payload, payload):
                try:
                    return parse_extraction(
                        raw_text, candidate, drop_invalid_claims=True
                    ), meta
                except ValueError:
                    continue
            return _qna_or_raise(
                ValueError(
                    f"{repair_exc} "
                    f"(after quote-repair; paraphrases are not allowed — copy "
                    f"exact phrases from the raw reference into source_quote, "
                    f"or use Q1[diagnosis]/A1 … Q5[plan]/A5 form)"
                )
            )
