"""Gold-only reference extraction, validation, and cohort identity."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from benchmark.schema import ConfirmedGold, GoldClaim, GoldSection, ModelCallMeta

SECTION_IDS = ("diagnosis", "tests", "urgency", "safety", "plan")
EXTRACTION_PROMPT_VERSION = "gold-extract-v2"
SCORING_VERSION = "graded-clinical-v4"


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
                "quotes over extra invented claims. If a section is absent, return an empty "
                "summary and claims. Return JSON only."
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
                "exists for a claim, remove that claim. Keep each section nonempty when "
                "the reference supports it. Return JSON only with the same shape."
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
            if _resolve_claim_quote(raw_norm, claim) is None:
                failures.append(_quote_reject_message(claim.id, claim.source_quote))
    return failures


def confirmed_gold(
    *,
    raw_text: str,
    sections: Mapping[str, GoldSection | Mapping[str, Any]],
    extraction_model: str,
    extraction_cost_usd: float = 0.0,
) -> ConfirmedGold:
    """Create the frozen contract; all five sections must be assessable."""
    parsed: Dict[str, GoldSection] = {}
    for section_id in SECTION_IDS:
        item = sections.get(section_id)
        if item is None:
            raise ValueError(f"Complete and confirm section: {section_id}")
        section = item if isinstance(item, GoldSection) else GoldSection.model_validate(item)
        if not section.summary.strip() or not section.claims:
            raise ValueError(f"Complete and confirm section: {section_id}")
        for claim in section.claims:
            claim.text = claim.source_quote.strip()
            claim.critical = False
        parsed[section_id] = section
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


def cohort_id(
    *,
    case_stem: str,
    gold: ConfirmedGold,
    prompt_version: str,
    model_config: Mapping[str, Any],
    benchmark_track: str,
) -> str:
    canonical = json.dumps(
        {
            "case_stem": _normalized(case_stem),
            "gold": gold.model_dump(mode="json", exclude={"confirmed_at"}),
            "scoring_version": SCORING_VERSION,
            "prompt_version": prompt_version,
            "models": model_config,
            "track": benchmark_track,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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


def extract_with_chat(
    raw_text: str,
    *,
    model: str,
    chat: Callable[..., Tuple[str, Any]],
    api_key: Optional[str] = None,
) -> Tuple[Dict[str, GoldSection], Any]:
    """Call a pinned extractor through an injected chat transport.

    On non-verbatim quotes: one quote-repair chat call, then drop remaining
    invalid claims if each section still has ≥1 verifiable claim.
    """
    if len(raw_text.strip()) < 40:
        raise ValueError("Reference is too short to extract safely")
    kwargs: Dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": 4000,
        "response_format": {"type": "json_object"},
        "timeout": 180.0,
    }
    if api_key is not None:
        kwargs["api_key"] = api_key

    raw, meta = chat(model, extraction_messages(raw_text), **kwargs)
    if getattr(meta, "error", None):
        raise RuntimeError(f"Gold extraction failed: {meta.error}")

    payload = extract_json_object(raw)
    try:
        return parse_extraction(raw_text, payload), meta
    except ValueError as first_exc:
        failures = collect_quote_failures(raw_text, payload) or [str(first_exc)]
        quote_related = any(
            "source quote" in f.casefold() or "verbatim" in f.casefold()
            for f in failures
        ) or ("source quote" in str(first_exc).casefold())
        if not quote_related:
            raise

        repair_raw, repair_meta = chat(
            model, quote_repair_messages(raw_text, payload, failures), **kwargs
        )
        if getattr(repair_meta, "error", None):
            raise RuntimeError(
                f"Gold quote repair failed: {repair_meta.error}. "
                f"Primary issue: {first_exc}"
            ) from first_exc
        if isinstance(meta, ModelCallMeta) and isinstance(repair_meta, ModelCallMeta):
            meta = _accumulate_extract_meta(
                meta, repair_meta, role="gold_quote_repair"
            )
        else:
            # Preserve whatever the transport returned when not ModelCallMeta.
            meta = repair_meta

        repair_payload = extract_json_object(repair_raw)
        try:
            sections = parse_extraction(raw_text, repair_payload)
            return sections, meta
        except ValueError as repair_exc:
            try:
                sections = parse_extraction(
                    raw_text, repair_payload, drop_invalid_claims=True
                )
                return sections, meta
            except ValueError:
                # Last resort: drop from primary payload
                try:
                    sections = parse_extraction(
                        raw_text, payload, drop_invalid_claims=True
                    )
                    return sections, meta
                except ValueError:
                    raise ValueError(
                        f"{repair_exc} "
                        f"(after quote-repair; paraphrases are not allowed — copy "
                        f"exact phrases from the raw reference into source_quote)"
                    ) from repair_exc
