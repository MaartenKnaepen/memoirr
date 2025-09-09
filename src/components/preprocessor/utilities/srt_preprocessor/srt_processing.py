"""SRT preprocessor: produce a minimal JSONL artifact from raw SRT.

Follows Memoirr coding standards: type hints, Google-style docstrings, SRP.
This module composes small utilities to create a clean, embedding-ready file.

Note: This preprocessor focuses on cleaning and normalization only. Chunking
and embedding are separate concerns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from components.preprocessor.utilities.srt_preprocessor.apply_cleaning import clean_and_collapse_captions
from components.preprocessor.utilities.srt_preprocessor.deduplicate import deduplicate_nearby
from components.preprocessor.utilities.srt_preprocessor.drop_empty import drop_empty_or_noise
from components.preprocessor.utilities.srt_preprocessor.language_filter import filter_english_captions
from components.preprocessor.utilities.srt_preprocessor.parse_srt import parse_srt_text
from components.preprocessor.utilities.srt_preprocessor.to_jsonl import to_jsonl_lines
from components.preprocessor.utilities.srt_preprocessor.types import CaptionUnit, CleanStats


def srt_preprocess_text(
    srt_text: str,
    *,
    min_len: int = 1,
    dedupe_window_ms: int = 1000,
) -> tuple[List[CaptionUnit], CleanStats]:
    """Clean an SRT text and return cleaned caption units plus stats.

    This is the core routine designed for testing. It returns clean caption
    units (one string per caption in ``lines[0]``) and summary statistics.

    Args:
        srt_text: Raw SRT content as a single string.
        min_len: Minimum characters required to keep a caption.
        dedupe_window_ms: Time window for nearby-duplicate removal.

    Returns:
        A tuple of (cleaned_captions, stats).
    """
    parsed = parse_srt_text(srt_text)
    total = len(parsed)

    english_only = filter_english_captions(parsed)

    cleaned = clean_and_collapse_captions(english_only)

    non_empty = drop_empty_or_noise(cleaned, min_len=min_len)
    dropped_empty = len(cleaned) - len(non_empty)

    deduped = deduplicate_nearby(non_empty, window_ms=dedupe_window_ms)
    deduped_count = len(non_empty) - len(deduped)

    stats = CleanStats(
        total_captions=total,
        kept=len(deduped),
        dropped_empty=dropped_empty,
        dropped_non_english=total - len(english_only),
        deduped=deduped_count,
    )
    return deduped, stats


def srt_preprocess_to_jsonl(
    srt_text: str,
) -> Iterator[str]:
    """Yield JSONL lines representing cleaned caption units.

    This streams JSON records for efficient file writing.
    """
    cleaned, _ = srt_preprocess_text(srt_text)
    yield from to_jsonl_lines(cleaned)


def srt_preprocess_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    min_len: int = 1,
    dedupe_window_ms: int = 1000,
) -> CleanStats:
    """Read an SRT file, write a JSONL file, and return summary stats.

    Args:
        input_path: Path to the input .srt file.
        output_path: Destination path for .jsonl file (overwritten if exists).
        min_len: Minimum characters required to keep a caption.
        dedupe_window_ms: Time window for nearby-duplicate removal.

    Returns:
        CleanStats containing summary counters.
    """
    input_p = Path(input_path)
    output_p = Path(output_path)

    text = input_p.read_text(encoding="utf-8", errors="replace")

    cleaned_captions, stats = srt_preprocess_text(
        text, min_len=min_len, dedupe_window_ms=dedupe_window_ms
    )

    with output_p.open("w", encoding="utf-8") as f:
        for line in to_jsonl_lines(cleaned_captions):
            f.write(line + "\n")

    return stats
