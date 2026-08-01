"""Beta comprehension track — isolated from graded A1–A5 History / Rebuild."""

from __future__ import annotations

PROTOCOL_ID = "beta-comprehension-v1"
# Artifacts stamp scoring_version with this id so graded Rebuild
# (graded-clinical-v4) never pools Beta runs into the clean mean.
SCORING_VERSION = PROTOCOL_ID
PROMPT_VERSION = "beta-comprehension-v1"
CASE_ID = "beta_comprehension"
PACK_FILENAME = "beta_comprehension.json"
