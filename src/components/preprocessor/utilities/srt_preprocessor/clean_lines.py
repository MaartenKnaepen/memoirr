"""Cleaning and normalization for SRT caption lines."""
from __future__ import annotations

import html
import re
from typing import List

_TAG_RE = re.compile(r"</?\w+[^>]*>")
_SSA_RE = re.compile(r"\{\\.*?\}")
_SFX_BRACKET_RE = re.compile(r"\[(?:[^\]]+)\]")
_SFX_PAREN_RE = re.compile(r"\((?:[^\)]+)\)")
_MUSIC_RE = re.compile(r"[♪♫]+.*?[♪♫]+")
_DASH_PREFIX_RE = re.compile(r"^\s*[-–—]\s*")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_caption_lines(lines: List[str]) -> str:
    """Clean a list of raw caption lines into a single normalized string.

    Steps:
    - Strip HTML-like tags (<i>, <b>, <font...>)
    - Remove SSA/ASS style braces (e.g., {\an8})
    - Remove hearing-impaired cues (e.g., [applause], (SIREN), ♪lyrics♪)
    - Unescape HTML entities
    - Normalize whitespace
    - Normalize dialogue markers by stripping leading dashes
    - Collapse multiple lines into one sentence-like string
    """
    cleaned: List[str] = []
    for raw in lines:
        if not raw:
            continue
        txt = html.unescape(raw)
        txt = _TAG_RE.sub(" ", txt)
        txt = _SSA_RE.sub(" ", txt)
        txt = _SFX_BRACKET_RE.sub(" ", txt)
        txt = _SFX_PAREN_RE.sub(" ", txt)
        txt = _MUSIC_RE.sub(" ", txt)
        txt = _DASH_PREFIX_RE.sub("", txt)
        txt = txt.strip()
        if txt:
            cleaned.append(txt)

    if not cleaned:
        return ""

    joined = " ".join(cleaned)
    joined = _MULTI_SPACE_RE.sub(" ", joined).strip()
    return joined
