# SRT Preprocessor Utilities

This package contains small, single-responsibility utilities used to clean and normalize SRT subtitle files into a minimal, embedding-ready representation.

It is designed to be composed by the orchestrator `srt_processing.py` and wrapped by the Haystack component `components/preprocessor/srt_preprocessor.py`.

---

## What this does

- Parse SRT into caption units (index, start_ms, end_ms, lines)
- Keep only English lines (pragmatic 95% ASCII heuristic)
- Clean lines: remove HTML-like tags, SSA/ASS braces, hearing‑impaired cues (e.g., `[applause]`, `♪…♪`), leading dialogue dashes, normalize whitespace
- Drop empty/too-short results
- Deduplicate exact repeated texts within a short time window (default 1000 ms)
- Emit a minimal JSONL record per caption unit

This does not chunk into embedding-sized segments; it preserves one record per caption to keep precise time alignment. Chunking can be a later step.

---

## Output format (JSONL)

One line per cleaned caption unit with a minimal schema:

```json
{
  "text": "Cleaned caption text",
  "start_ms": 1000,
  "end_ms": 2000,
  "caption_index": 1
}
```

Notes:
- `text`: English-only, no timecodes/indices/formatting/tags/HI cues
- `start_ms`, `end_ms`: integers, for deep-linking playback
- `caption_index`: original SRT caption index (traceability)

---

## Modules

- `types.py`
  - `CaptionUnit`: (caption_index, start_ms, end_ms, lines)
  - `JsonRecord`: (text, start_ms, end_ms, caption_index)
  - `CleanStats`: counts for QA (total, kept, dropped_empty, dropped_non_english, deduped)

- `exceptions.py`
  - `SRTParseError`, `LanguageFilterError`

- `parse_srt.py`
  - `parse_srt_text(srt_text: str) -> list[CaptionUnit]`
  - Uses the `srt` library; extracts per-caption timings and lines

- `language_filter.py`
  - `is_english_text_heuristic(text: str) -> bool` (≥ 95% ASCII)
  - `filter_english_captions(captions) -> list[CaptionUnit]` (line-level filtering)

- `clean_lines.py`
  - `clean_caption_lines(lines: list[str]) -> str`
  - Strips HTML-ish tags (`<i>`, `<b>`), SSA/ASS braces (`{\an8}`), hearing‑impaired cues (`[applause]`, `(SIREN)`, `♪…♪`), leading dashes; unescapes entities; normalizes whitespace

- `apply_cleaning.py`
  - `clean_and_collapse_captions(captions) -> list[CaptionUnit]`
  - Applies `clean_caption_lines` and collapses into a single string per caption

- `drop_empty.py`
  - `drop_empty_or_noise(captions, *, min_len=1) -> list[CaptionUnit]`

- `deduplicate.py`
  - `deduplicate_nearby(captions, *, window_ms=1000) -> list[CaptionUnit]`
  - Drops exact duplicate texts appearing within `window_ms`

- `to_jsonl.py`
  - `to_json_records(captions) -> Iterator[JsonRecord]`
  - `to_jsonl_lines(captions) -> Iterator[str]`

- `srt_processing.py`
  - Orchestrates the full preprocessing pipeline for SRT text
  - `srt_preprocess_text(srt_text, *, min_len=1, dedupe_window_ms=1000) -> (list[CaptionUnit], CleanStats)`

---

## Typical usage

Direct (utilities orchestrator):

```python
from components.preprocessor.utilities.srt_preprocessor.srt_processing import srt_preprocess_text
from components.preprocessor.utilities.srt_preprocessor.to_jsonl import to_jsonl_lines

srt_text = """1\n00:00:01,000 --> 00:00:02,000\n- Hello!\n\n2\n00:00:02,500 --> 00:00:04,000\nHi again\n"""
cleaned_captions, stats = srt_preprocess_text(srt_text)
jsonl = list(to_jsonl_lines(cleaned_captions))
```

Inside a Haystack pipeline, use the wrapper component: `components/preprocessor/srt_preprocessor.py`.

---

## Design decisions

- Keep caption granularity for precise time alignment
- Filter non‑English aggressively with a simple threshold (≥ 95% ASCII)
- Remove hearing‑impaired cues from `text` (can be revisited later if needed)
- Minimal schema for portability and future evolution

---

## Extensibility

- Swap language heuristic for a detector (fastText, langdetect) without changing public APIs
- Adjust dedupe window, minimum length thresholds
- Add optional metadata fields or a scene/windowing chunker as a separate stage
- Add a file-writer component for pipeline-driven output

---

## Tests

Tests mirror the `src` structure under `test/` and cover all modules:
- parsing, filtering, cleaning, deduplication, JSONL conversion, and end‑to‑end orchestration

Run tests:

```bash
pytest test
```
