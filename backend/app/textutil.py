"""Small text helpers shared across the app."""
from __future__ import annotations

import re


def no_em_dashes(s: str | None) -> str:
    """Remove em/en dashes from model output (the user wants none).

    Em dash -> comma; en dash -> hyphen (keeps numeric ranges like 5-10 readable).
    """
    if not s:
        return s or ""
    s = s.replace("—", ", ").replace("–", "-")
    s = re.sub(r"\s*,\s*,\s*", ", ", s)   # collapse ", ,"
    s = re.sub(r"\s+,", ",", s)           # "word ," -> "word,"
    s = re.sub(r",\s*([.!?])", r"\1", s)  # ", ." -> "."
    return s.strip()


def trim_to_sentence(s: str, limit: int = 600) -> str:
    """Trim to the last complete sentence within `limit` chars (no mid-word cuts)."""
    if not s:
        return ""
    if len(s) <= limit:
        return s.strip()
    cut = s[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return (cut[: end + 1] if end > 60 else cut.rsplit(" ", 1)[0] + "…").strip()
