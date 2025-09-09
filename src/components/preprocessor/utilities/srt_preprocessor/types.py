"""Shared types and data models for SRT preprocessing.

Adheres to the project's coding standards: type hints, Google-style docstrings,
SRP, and testability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class CaptionUnit:
    """A single SRT caption block with timing and raw text lines.

    Attributes:
        caption_index: Original SRT index (1-based typical).
        start_ms: Start time in milliseconds.
        end_ms: End time in milliseconds.
        lines: Raw text lines as extracted from SRT (without timecodes).
    """

    caption_index: int
    start_ms: int
    end_ms: int
    lines: List[str]


@dataclass(frozen=True)
class JsonRecord:
    """Minimal JSONL record for a cleaned caption unit.

    Attributes:
        text: Cleaned caption text (English only; no tags/timecodes/cues).
        start_ms: Start time in milliseconds.
        end_ms: End time in milliseconds.
        caption_index: Original SRT index (for traceability).
    """

    text: str
    start_ms: int
    end_ms: int
    caption_index: int


@dataclass(frozen=True)
class CleanStats:
    """Summary statistics produced by the SRT preprocessing run.

    Attributes:
        total_captions: Number of caption blocks parsed from the input SRT.
        kept: Number of captions emitted to JSONL.
        dropped_empty: Captions dropped because they became empty/noise.
        dropped_non_english: Captions (or lines) dropped because not English.
        deduped: Count of captions removed as near-duplicates.
    """

    total_captions: int
    kept: int
    dropped_empty: int
    dropped_non_english: int
    deduped: int
