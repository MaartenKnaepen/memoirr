# Chunker

This folder contains the design and implementation notes for text chunking used in Memoirr. It summarizes our decision, trade-offs, and a practical plan to implement time-aware chunking for subtitles using the chonkie library.

## Decision Summary

- Primary method: Semantic chunking at ingestion using `chonkie.SemanticChunker`.
  - Why: Dialog-heavy subtitles require merging short lines into coherent units; we need fast retrieval and precise jump-to-time. SemanticChunker gives coherent, deterministic, and time-accurate chunks.
- Optional fallback (later): Late chunking at query-time using `chonkie.LateChunker` on larger retrieved windows for edge cases.
  - Why: Enables experimentation without re-indexing and can inject document-level context into chunk embeddings. Kept off by default to preserve latency.
- Neural chunker: Not planned initially (heavier inference, overkill for SRTs).

## When to Consider Alternatives

- Consider `LateChunker` if you need rapid iteration on chunk parameters without re-indexing, or if you want broader context baked into chunk embeddings.
- Consider `NeuralChunker` only if you see consistent mis-boundaries that semantic similarity + filtering can’t handle.

## Parameters (initial defaults for subtitles)

- `threshold`: 0.75 (0.70–0.80 acceptable)
- `chunk_size`: 512 tokens
- `similarity_window`: 3
- `min_sentences_per_chunk`: 2
- `min_characters_per_sentence`: 24
- `delim`: default or `[". ", "! ", "? ", "\n"]`
- `include_delim`: "prev"
- `skip_window`: 0 (use 1 only if non-consecutive similarity clearly helps)
- `filter_window`: 5, `filter_polyorder`: 3, `filter_tolerance`: 0.2

Record the chosen parameter set alongside outputs for reproducibility.

## Implementation Plan (time-aware)

Source: cleaned JSONL from `components/preprocessor/utilities/srt_preprocessor` (one record per caption with `text`, `start_ms`, `end_ms`, `caption_index`).

High-level steps:

1) Build a single text string and span map
- Concatenate each caption `text` with a consistent delimiter (e.g., single space).
- Track per-caption character spans: for caption i, store `(start_char, end_char)` in the concatenated string.

2) Chunk with SemanticChunker
- Initialize `SemanticChunker` with the defaults above.
- Call `chunker.chunk(text)` to produce chunks with `(text, start_index, end_index, token_count)`.

3) Map chunks back to time
- For each chunk’s `(start_index, end_index)`, find the set of caption spans that overlap this range.
- Derive chunk time range: `start_ms = min(start_ms of covered captions)`, `end_ms = max(end_ms of covered captions)`.
- Optionally retain `caption_indices` for traceability.

4) Emit chunk records
- Prepare chunk objects (or JSONL) with fields like: `text`, `start_ms`, `end_ms`, `token_count`, `caption_indices` (optional), plus `chunker_params` (optional versioning).
- These records are now ready to embed and index.

5) Embedding and storage (later)
- Use your chosen embedder (e.g., sentence-transformers) and document store.
- Keep `start_ms`/`end_ms` in metadata to support jump-to-time in the app.

### Data Flow

```
SRT (.srt)
  → SRT Preprocessor (cleaning, filtering, dedupe)
  → Clean JSONL (one record per caption)
  → Semantic Chunker (build text + span map → chunk)
  → Time-aware Chunks (ready for embedding/indexing)
  → Retrieval → Jump-to-time
```

## Testing & Validation

- Mapping correctness: For several fixtures, verify that chunk time ranges map to intended portions of the episode.
- Chunk coherence: Spot-check chunks read as a single idea/dialogue.
- Retrieval quality: Evaluate representative queries; confirm relevant chunks rank high.
- Latency: Ingestion-time cost is acceptable; query-time remains fast.

## Optional: Late Chunking Fallback

- Strategy: For low-confidence cases, retrieve a larger window (scene/full transcript), then apply `LateChunker` at query-time to produce refined segments.
- Trade-offs: Higher query-time CPU/latency; keep behind a feature flag.

## Notes & References

- chonkie SemanticChunker provides similarity-based grouping with smoothing and optional skip-window merging.
- chonkie LateChunker builds chunk embeddings from a document-level embedding via recursive rules.
- Retaining a clean JSONL as the canonical source enables re-chunking without re-parsing SRT.

## Next Steps

- Implement a `semantic_chunker` component that:
  - Accepts cleaned JSONL per SRT input
  - Produces time-aware chunks using `SemanticChunker`
  - Includes tests for span mapping and parameterized behavior
- Optionally prototype a `late_chunker` path for evaluation on a small subset.
