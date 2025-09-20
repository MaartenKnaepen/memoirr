# Semantic Chunker Utilities

This directory contains the pure utility functions that implement the semantic chunking pipeline. These functions are orchestrated by the SemanticChunker component.

## Modules

### orchestrate_chunking.py
The main orchestration function that composes all the utilities into a complete chunking pipeline:
- Parses cleaned JSONL lines into records
- Builds concatenated text and spans
- Runs chonkie SemanticChunker with a local embeddings model
- Maps chunk spans back to time ranges
- Emits JSONL chunk records and summary stats

### parse_jsonl.py
Parses cleaned JSONL lines into structured caption records with validation and error handling.

### build_text_and_spans.py
Builds concatenated text and span mappings for semantic chunking while preserving time information.

### run_semantic_chunker.py
Executes the chonkie SemanticChunker with proper model loading and parameter handling.

### map_chunk_to_time.py
Maps chunk character spans back to original time ranges from the source captions.

### emit_chunk_records.py
Converts chunked data back to JSONL format with metadata and statistics.

### types.py
Type definitions and dataclasses for the chunking pipeline.