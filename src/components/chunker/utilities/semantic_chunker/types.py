"""Shared types and data models for the SemanticChunker utilities.

Follows Memoirr coding standards: type hints, docstrings, and SRP.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class CaptionJson:
    """Parsed caption record from preprocessor JSONL.

    Attributes:
        text: Cleaned caption text.
        start_ms: Start time in milliseconds.
        end_ms: End time in milliseconds.
        caption_index: Original SRT index.
    """

    text: str
    start_ms: int
    end_ms: int
    caption_index: int


@dataclass(frozen=True)
class Span:
    """Character span for a caption within the concatenated text.

    Attributes:
        start: Inclusive start index.
        end: Exclusive end index.
        start_ms: Start time of the caption.
        end_ms: End time of the caption.
        caption_index: Original SRT index.
    """

    start: int
    end: int
    start_ms: int
    end_ms: int
    caption_index: int


@dataclass(frozen=True)
class ChunkerParams:
    """Configuration for semantic chunking."""

    threshold: float | int | str
    chunk_size: int
    similarity_window: int
    min_sentences: int
    min_characters_per_sentence: int
    delim: List[str] | str
    include_delim: Optional[str]
    skip_window: int


@dataclass(frozen=True)
class ChunkSpan:
    """Chunk span produced by chonkie with positional metadata."""

    text: str
    start_index: int
    end_index: int
    token_count: int


@dataclass(frozen=True)
class ChunkWithTime:
    """Chunk span mapped back to time, optionally with caption indices."""

    text: str
    start_ms: int
    end_ms: int
    token_count: int
    caption_indices: Optional[List[int]] = None


@dataclass(frozen=True)
class ChunkRecord:
    """JSONL-ready record for a chunk."""

    text: str
    start_ms: int
    end_ms: int
    token_count: int
    caption_indices: Optional[List[int]]
    chunker_params: Optional[dict]
