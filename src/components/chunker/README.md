# Semantic Chunker

This package provides time‑aware semantic chunking of cleaned captions. It consumes the JSONL output from the SRT preprocessor and produces chunk JSONL lines with accurate start/end times, ready for embedding and indexing.

It is composed by the utilities in `utilities/semantic_chunker/` (pure functions and types), orchestrated by `orchestrate_chunking.py`, and wrapped by a Haystack component `semantic_chunker.py`.

---

## What this does

- Parse cleaned caption JSONL lines into typed records
- Concatenate texts and keep a span map for each original caption
- Use Chonkie `SemanticChunker` to split the text into coherent, similarity‑based chunks
- Map chunk character ranges back to time using the span map (preserves deep‑link playback)
- Emit JSONL chunk records compatible with downstream embedding / vector DBs

This stage merges short dialog lines into coherent units while preserving accurate time ranges.

---

## Inputs

Cleaned caption JSONL (one line per caption) as produced by the preprocessor. Example:

```json
{"text": "Hello!", "start_ms": 1000, "end_ms": 2000, "caption_index": 1}
{"text": "How are you?", "start_ms": 2100, "end_ms": 3000, "caption_index": 2}
```

---

## Outputs (JSONL)

One line per semantic chunk:

```json
{
  "text": "Hello! How are you?",
  "start_ms": 1000,
  "end_ms": 3000,
  "token_count": 27,
  "caption_indices": [1, 2],
  "chunker_params": {
    "threshold": 0.75,
    "chunk_size": 512,
    "similarity_window": 3,
    "min_sentences": 2,
    "min_characters_per_sentence": 24,
    "delim": [". ", "! ", "? ", "\n"],
    "include_delim": "prev",
    "skip_window": 0
  }
}
```

Notes:
- `token_count` is reported by Chonkie for each chunk.
- `caption_indices` (optional) preserve traceability back to original captions.
- `chunker_params` (optional) record the parameter set used to produce the chunk.

---

## Configuration

Settings are read from `.env` via `src/core/config.py` and can be overridden per component constructor.

Required/important settings:
- `EMBEDDING_MODEL_NAME` (default: `qwen3-embedding-0.6B`)
  - Name of the local SentenceTransformers model folder. Resolved under `models/<name>` or `models/chunker/<name>`.
- `EMBEDDING_DEVICE` (optional; e.g., `cuda:0` or `cpu`)
- `EMBEDDING_DIMENSION` (optional)
  - If set, forwarded to Chonkie for Pooling modules that require it.

Chunker parameters (env → defaults):
- `CHUNK_THRESHOLD` (default: `auto`)
  - Accepts: float in (0,1), percentile (1–100], or `auto`. The runtime coerces to a valid (0,1) float.
- `CHUNK_SIZE` (default: `512`)
- `CHUNK_SIMILARITY_WINDOW` (default: `3`)
- `CHUNK_MIN_SENTENCES` (default: `2`)
- `CHUNK_MIN_CHARACTERS_PER_SENTENCE` (default: `24`)
- `CHUNK_DELIM` (default: `[". ", "! ", "? ", "\n"]`)
  - JSON array string or a plain string; if JSON parsing fails, the raw string is used.
- `CHUNK_INCLUDE_DELIM` (default: `prev`) — `prev` | `next` | `none`
- `CHUNK_SKIP_WINDOW` (default: `0`)
- `CHUNK_INCLUDE_PARAMS` (default: `true`)
- `CHUNK_INCLUDE_CAPTION_INDICES` (default: `true`)
- `CHUNK_FAIL_FAST` (default: `true`)

---

## Local model layout (offline)

A local SentenceTransformers folder is expected under `models/chunker/<name>` (or `models/<name>`):
- `config.json`, tokenizer files (`tokenizer.json` or `vocab.txt`/`merges.txt`, etc.)
- `model.safetensors`
- `modules.json` describing the ST pipeline modules
- `1_Pooling/config.json` with at least `{"word_embedding_dimension": <int>}`
- `2_Normalize/config.json` (if referenced by `modules.json`)

Chonkie will instantiate embeddings from this path via its `AutoEmbeddings` and `SentenceTransformerEmbeddings`. No network access is required when all files are present.

---

## Modules

- `semantic_chunker.py`
  - Haystack component that reads configuration, calls the orchestrator, and outputs chunk JSONL lines + stats.
- `utilities/semantic_chunker/types.py`
  - `CaptionJson`, `Span`, `ChunkerParams`, `ChunkSpan`, `ChunkWithTime`, `ChunkRecord` (data models)
- `utilities/semantic_chunker/parse_jsonl.py`
  - Parse cleaned caption JSONL lines into `CaptionJson` records
- `utilities/semantic_chunker/build_text_and_spans.py`
  - Concatenate texts and compute per‑caption character spans
- `utilities/semantic_chunker/run_semantic_chunker.py`
  - Resolve local model path, coerce threshold, instantiate Chonkie `SemanticChunker`, return `ChunkSpan` list
  - Cross‑version compatibility: supports `threshold` and `similarity_threshold`; falls back to `chunk()` or `__call__()`
- `utilities/semantic_chunker/map_chunk_to_time.py`
  - Map chunk span back to `[start_ms, end_ms]` and caption indices using span overlaps
- `utilities/semantic_chunker/emit_chunk_records.py`
  - Emit final JSONL lines with optional `chunker_params` and `caption_indices`
- `utilities/semantic_chunker/orchestrate_chunking.py`
  - Orchestrate: parse → build spans → chunk → map time → emit and compute summary stats

---

## Typical usage

Direct (utilities orchestrator):

```python
from src.components.chunker.utilities.semantic_chunker.orchestrate_chunking import orchestrate_semantic_chunking

lines = [
    '{"text": "Hello!", "start_ms": 1000, "end_ms": 2000, "caption_index": 1}',
    '{"text": "How are you?", "start_ms": 2100, "end_ms": 3000, "caption_index": 2}',
]
chunk_lines, stats = orchestrate_semantic_chunking(lines)
```

Inside a Haystack pipeline:

```python
from haystack import Pipeline
from src.components.preprocessor.srt_preprocessor import SRTPreprocessor
from src.components.chunker.semantic_chunker import SemanticChunker

pipe = Pipeline()
pipe.add_component("pre", SRTPreprocessor())
pipe.add_component("chunk", SemanticChunker())
pipe.connect("pre.jsonl_lines", "chunk.jsonl_lines")

result = pipe.run({"pre": {"srt_text": "..."}})
chunks = result["chunk"]["chunk_lines"]
stats = result["chunk"]["stats"]
```

---

## Design decisions

- Time awareness: maintain precise jump‑to‑time by mapping chunk spans to caption time ranges
- Deterministic, semantic grouping with Chonkie; minimal defaults tailored for subtitle dialogue
- Keep JSONL schema small and traceable (caption_indices + optional chunker_params)
- Readable, testable utilities; Haystack component remains thin

---

## Extensibility

- Parameterize thresholds, windows, and delimiters via env/constructor
- Swap out or add chunkers (e.g., SDPM, sentence/window variants) following the same orchestration pattern
- Add refineries (e.g., overlap/embeddings) post‑chunking if needed for retrieval

---

## Tests

Tests mirror the `src` structure under `test/` and include:
- Utilities: parsing, spans, time mapping (including no‑overlap), emitting JSONL
- Runtime: threshold coercion, Chonkie param fallback (`threshold` → `similarity_threshold`), `chunk()` vs `__call__()`
- Orchestration: end‑to‑end parse → chunk → map → emit
- Component: env propagation and delimiter parsing

Run tests:

```bash
pytest test
```

Optional integration tests can be added to exercise a real local model when available (kept opt‑in to avoid network).
