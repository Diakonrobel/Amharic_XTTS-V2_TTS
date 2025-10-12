"""
Language normalization utilities for the XTTS Amharic pipeline.

Goal: ensure we consistently use 'amh' (ISO 639-3) as the canonical code
when interfacing with Coqui XTTS tokenizer/trainer and when writing dataset
artifacts (e.g., lang.txt), while still accepting user inputs like 'am',
'am-ET', 'AM', or 'Amharic'.

We also provide helpers for contexts like YouTube transcripts where
ISO 639-1 ('am') may be required.
"""
from __future__ import annotations

from typing import Optional


_AMHARIC_SYNONYMS = {
    "am", "amh", "am-et", "am_et", "amharic", "አማርኛ",
}


def _safe_lower(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    try:
        return s.strip().lower()
    except Exception:
        return s


def is_amharic(code: Optional[str]) -> bool:
    """Return True if the given language code/name refers to Amharic."""
    s = _safe_lower(code)
    if not s:
        return False
    s = s.replace("_", "-")
    return s in _AMHARIC_SYNONYMS


def canonical_lang(code: Optional[str], purpose: str = "coqui") -> Optional[str]:
    """
    Canonicalize a language code for a given purpose.

    - For Coqui XTTS/training/inference/dataset writing, we use 'amh'.
    - For Chinese, Coqui commonly expects 'zh-cn' in some paths.
    - Returns None unchanged.

    Args:
        code: Incoming language code (can be 'am', 'amh', 'am-ET', etc.)
        purpose: One of {"coqui", "dataset", "ui"}. Currently "coqui" and "dataset"
                 behave the same (return 'amh' for Amharic). "ui" returns a nicer
                 display code but functionally identical here.
    """
    s = _safe_lower(code)
    if s is None:
        return None

    # Normalize separators
    s = s.replace("_", "-")

    # Amharic handling
    if is_amharic(s):
        # Always use ISO 639-3 for Coqui/train/dataset to avoid NotImplementedError
        return "amh"

    # Chinese handling for Coqui
    if s == "zh":
        return "zh-cn" if purpose in {"coqui", "dataset"} else "zh"

    # Strip region if any (e.g., xx-YY -> xx)
    if "-" in s:
        base = s.split("-")[0]
        # Keep only base for our current needs
        s = base

    return s


def to_transcript_code(code: Optional[str]) -> Optional[str]:
    """
    Convert a canonical code back to a typical transcript/YouTube code.
    For YouTube transcripts, 'am' is the expected code, not 'amh'.
    """
    s = _safe_lower(code)
    if s is None:
        return None
    if is_amharic(s):
        return "am"
    if s == "zh-cn":
        return "zh"
    return s
