import json
import sys
import types
from pathlib import Path

import pytest

from src.components.chunker.utilities.semantic_chunker.types import ChunkSpan
from src.components.chunker.utilities.semantic_chunker.orchestrate_chunking import orchestrate_semantic_chunking
from src.components.chunker.utilities.semantic_chunker.run_semantic_chunker import (
    _coerce_threshold,
    _resolve_model_dir,
    run_semantic_chunker,
)


def test_coerce_threshold_valid_inputs():
    assert _coerce_threshold("auto") == 0.75
    assert _coerce_threshold(0.8) == 0.8
    assert _coerce_threshold("0.8") == 0.8
    assert _coerce_threshold(75) == 0.75
    assert _coerce_threshold("75") == 0.75


@pytest.mark.parametrize("bad", [0, 1, 101, "abc"])
def test_coerce_threshold_invalid_inputs(bad):
    with pytest.raises(ValueError):
        _coerce_threshold(bad)


def test_resolve_model_dir_prefers_existing_under_models_chunker(tmp_path, monkeypatch):
    # Change into tmp directory and create expected structure
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models" / "chunker" / "foo").mkdir(parents=True)
    # Only models/chunker/foo exists
    resolved = _resolve_model_dir("foo")
    assert Path(resolved).as_posix().endswith("models/chunker/foo")


def test_orchestrate_passes_settings_and_maps_time(monkeypatch):
    # Prepare minimal JSONL lines
    lines = [
        json.dumps({"text": "Hello", "start_ms": 0, "end_ms": 1000, "caption_index": 1}),
        json.dumps({"text": "World", "start_ms": 1100, "end_ms": 2000, "caption_index": 2}),
    ]

    # Capture arguments passed to run_semantic_chunker and return a fake chunk span
    captured = {}

    def fake_run_semantic_chunker(text, params, model_name, device=None):
        captured["text"] = text
        captured["params"] = params
        captured["model_name"] = model_name
        captured["device"] = device
        # One chunk covering both captions and the separator
        start = 0
        end = len("Hello World")
        return [ChunkSpan(text=text[start:end], start_index=start, end_index=end, token_count=5)]

    # Patch the symbol used inside orchestrate_semantic_chunking
    monkeypatch.setattr(
        "src.components.chunker.utilities.semantic_chunker.orchestrate_chunking.run_semantic_chunker",
        fake_run_semantic_chunker,
        raising=False,
    )

    # Also ensure settings embedding model is read (use default from config)
    out_lines, stats = orchestrate_semantic_chunking(lines)

    # Verify outputs
    assert len(out_lines) == 1
    obj = json.loads(out_lines[0])
    assert obj["start_ms"] == 0
    assert obj["end_ms"] == 2000
    assert stats["input_captions"] == 2
    assert stats["output_chunks"] == 1
    # Ensure params propagated
    assert captured["model_name"]  # non-empty


def test_run_semantic_chunker_calls_chunk_when_available(monkeypatch, tmp_path):
    # Build a fake chonkie module with a SemanticChunker exposing .chunk
    class DummyChunk:
        def __init__(self, text):
            self.text = text
            self.start_index = 0
            self.end_index = len(text)
            self.token_count = 3

    class DummySemanticChunker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def chunk(self, text):
            return [DummyChunk(text)]

    # Patch the symbol used by the runtime directly
    monkeypatch.setattr(
        "src.components.chunker.utilities.semantic_chunker.run_semantic_chunker.ChonkieSemanticChunker",
        DummySemanticChunker,
        raising=False,
    )

    text = "Hello world"
    from src.components.chunker.utilities.semantic_chunker.types import ChunkerParams
    params = ChunkerParams(
        threshold=0.8,
        chunk_size=16,
        similarity_window=1,
        min_sentences=1,
        min_characters_per_sentence=1,
        delim=[". "],
        include_delim="prev",
        skip_window=0,
    )
    chunks = run_semantic_chunker(text, params, model_name="whatever")
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_run_semantic_chunker_calls_dunder_call_when_chunk_missing(monkeypatch, tmp_path):
    # Build a fake chonkie module with a SemanticChunker exposing __call__ only
    class DummyChunk:
        def __init__(self, text):
            self.text = text
            self.start_index = 0
            self.end_index = len(text)
            self.token_count = 3

    class DummySemanticChunker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def __call__(self, text):
            return [DummyChunk(text)]

    # Patch the symbol used by the runtime directly
    monkeypatch.setattr(
        "src.components.chunker.utilities.semantic_chunker.run_semantic_chunker.ChonkieSemanticChunker",
        DummySemanticChunker,
        raising=False,
    )

    text = "Hello world"
    from src.components.chunker.utilities.semantic_chunker.types import ChunkerParams
    params = ChunkerParams(
        threshold=0.8,
        chunk_size=16,
        similarity_window=1,
        min_sentences=1,
        min_characters_per_sentence=1,
        delim=[". "],
        include_delim="prev",
        skip_window=0,
    )
    chunks = run_semantic_chunker(text, params, model_name="whatever")
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_run_semantic_chunker_falls_back_to_similarity_threshold_keyword(monkeypatch):
    # Fake chonkie SemanticChunker that errors when 'threshold' is used but accepts 'similarity_threshold'
    class DummyChunk:
        def __init__(self, text):
            self.text = text
            self.start_index = 0
            self.end_index = len(text)
            self.token_count = 3

    class DummySemanticChunker:
        def __init__(self, **kwargs):
            if "threshold" in kwargs:
                raise TypeError("unexpected keyword 'threshold'")
            self.kwargs = kwargs
        def chunk(self, text):
            return [DummyChunk(text)]

    # Patch the symbol used by the runtime and raise on threshold kw
    monkeypatch.setattr(
        "src.components.chunker.utilities.semantic_chunker.run_semantic_chunker.ChonkieSemanticChunker",
        DummySemanticChunker,
        raising=False,
    )

    text = "Hello world"
    from src.components.chunker.utilities.semantic_chunker.types import ChunkerParams
    params = ChunkerParams(
        threshold=0.8,
        chunk_size=16,
        similarity_window=1,
        min_sentences=1,
        min_characters_per_sentence=1,
        delim=[". "],
        include_delim="prev",
        skip_window=0,
    )

    chunks = run_semantic_chunker(text, params, model_name="whatever")
    # Ensure the constructor received similarity_threshold instead
    assert len(chunks) == 1
