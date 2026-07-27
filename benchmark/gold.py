"""Gold-only reference extraction, validation, and cohort identity."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from benchmark.schema import ConfirmedGold, GoldClaim, GoldSection

SECTION_IDS = ("diagnosis", "tests", "urgency", "safety", "plan")
EXTRACTION_PROMPT_VERSION = "gold-extract-v1"
SCORING_VERSION = "graded-clinical-v3"


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


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
                "source_quote. If a section is absent, return an empty summary and claims. "
                "Return JSON only."
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


def parse_extraction(raw_text: str, payload: Mapping[str, Any]) -> Dict[str, GoldSection]:
    """Strictly validate source-linked sections proposed by the extractor."""
    raw_norm = _normalized(raw_text)
    raw_sections = payload.get("sections")
    # Some providers honor the requested content but omit the outer
    # {"sections": ...} wrapper. Accept that harmless schema variation.
    if not isinstance(raw_sections, Mapping) and all(
        section_id in payload for section_id in SECTION_IDS
    ):
        raw_sections = payload
    if not isinstance(raw_sections, Mapping):
        raise ValueError("Extractor response has no sections object")

    sections: Dict[str, GoldSection] = {}
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
            quote = _normalized(claim.source_quote)
            if not quote or quote not in raw_norm:
                raise ValueError(
                    f"Claim {claim.id} has no verbatim source quote; extractor output rejected"
                )
            if not claim.text.strip():
                raise ValueError(f"Claim {claim.id} is empty")
            claims.append(claim)
        sections[section_id] = GoldSection(
            summary=str(item.get("summary") or "").strip(),
            claims=claims,
        )
    return sections


def confirmed_gold(
    *,
    raw_text: str,
    sections: Mapping[str, GoldSection | Mapping[str, Any]],
    extraction_model: str,
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
        parsed[section_id] = section
    return ConfirmedGold(
        raw_text=raw_text.strip(),
        sections=parsed,
        confirmed_at=datetime.now(timezone.utc).isoformat(),
        extraction_model=extraction_model,
        extraction_prompt_version=EXTRACTION_PROMPT_VERSION,
    )


def gold_json(gold: ConfirmedGold) -> str:
    return gold.model_dump_json(indent=2)


def load_confirmed_gold(value: str | Mapping[str, Any] | ConfirmedGold) -> ConfirmedGold:
    if isinstance(value, ConfirmedGold):
        return value
    if isinstance(value, Mapping):
        return ConfirmedGold.model_validate(value)
    text = (value or "").strip()
    if not text:
        raise ValueError("A confirmed five-section reference is required")
    try:
        return ConfirmedGold.model_validate_json(text)
    except Exception as exc:
        raise ValueError(
            "Gold must be the confirmed five-section JSON contract, not unreviewed prose"
        ) from exc


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


def extract_with_chat(
    raw_text: str,
    *,
    model: str,
    chat: Callable[..., Tuple[str, Any]],
    api_key: Optional[str] = None,
) -> Tuple[Dict[str, GoldSection], Any]:
    """Call a pinned extractor through an injected chat transport."""
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
    return parse_extraction(raw_text, payload), meta
