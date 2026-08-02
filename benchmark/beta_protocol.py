"""Comprehension track wire ids — isolated from graded A1–A5 History / Rebuild.

Primary stamps are ``comprehension-v1`` / ``comprehension``.
Legacy ``beta-*`` ids remain in alias sets so older History artifacts still pool.
"""

from __future__ import annotations

PROTOCOL_ID = "comprehension-v1"
# Artifacts stamp scoring_version with this id so graded Rebuild
# (graded-clinical-v4) never pools Comprehension runs into the clean mean.
SCORING_VERSION = PROTOCOL_ID
PROMPT_VERSION = PROTOCOL_ID
CASE_ID = "comprehension"
PACK_FILENAME = "comprehension.json"
# Dual-read (and transitional dual-write of slot keys) for pre-rename History.
LEGACY_PROTOCOL_ID = "beta-comprehension-v1"
LEGACY_CASE_ID = "beta_comprehension"
LEGACY_PACK_FILENAME = "beta_comprehension.json"
SCORING_VERSION_ALIASES = frozenset({PROTOCOL_ID, LEGACY_PROTOCOL_ID})
CASE_ID_ALIASES = frozenset({CASE_ID, LEGACY_CASE_ID})
