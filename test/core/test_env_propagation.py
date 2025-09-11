import json
import os
import pytest

from src.core.config import get_settings
from src.components.preprocessor.srt_preprocessor import SRTPreprocessor
from src.components.chunker.semantic_chunker import SemanticChunker


@pytest.fixture(autouse=True)
def clear_settings_cache():
    # Ensure each test sees fresh env values
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "my-model")
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")

    monkeypatch.setenv("CHUNK_THRESHOLD", "0.75")
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("CHUNK_SIMILARITY_WINDOW", "5")
    monkeypatch.setenv("CHUNK_MIN_SENTENCES", "3")
    monkeypatch.setenv("CHUNK_MIN_CHARACTERS_PER_SENTENCE", "42")
    monkeypatch.setenv("CHUNK_DELIM", '[". ", "? "]')
    monkeypatch.setenv("CHUNK_INCLUDE_DELIM", "next")
    monkeypatch.setenv("CHUNK_SKIP_WINDOW", "1")
    monkeypatch.setenv("CHUNK_INCLUDE_PARAMS", "false")
    monkeypatch.setenv("CHUNK_INCLUDE_CAPTION_INDICES", "false")
    monkeypatch.setenv("CHUNK_FAIL_FAST", "false")

    monkeypatch.setenv("PRE_MIN_LEN", "7")
    monkeypatch.setenv("PRE_DEDUPE_WINDOW_MS", "2222")

    # Refresh settings after env changes
    get_settings.cache_clear()
    s = get_settings()

    assert s.embedding_model_name == "my-model"
    assert s.device == "cpu"

    assert s.chunk_threshold == "0.75"
    assert s.chunk_size == 256
    assert s.chunk_similarity_window == 5
    assert s.chunk_min_sentences == 3
    assert s.chunk_min_characters_per_sentence == 42
    # Stored as string in settings; parsing happens in component
    assert s.chunk_delim == '[". ", "? "]'
    assert s.chunk_include_delim == "next"
    assert s.chunk_skip_window == 1
    assert s.chunk_include_params is False
    assert s.chunk_include_caption_indices is False
    assert s.chunk_fail_fast is False

    assert s.pre_min_len == 7
    assert s.pre_dedupe_window_ms == 2222


def test_components_pick_env_defaults(monkeypatch):
    # Set env values
    monkeypatch.setenv("CHUNK_THRESHOLD", "0.75")
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("CHUNK_SIMILARITY_WINDOW", "5")
    monkeypatch.setenv("CHUNK_MIN_SENTENCES", "3")
    monkeypatch.setenv("CHUNK_MIN_CHARACTERS_PER_SENTENCE", "42")
    monkeypatch.setenv("CHUNK_DELIM", '[". ", "? "]')
    monkeypatch.setenv("CHUNK_INCLUDE_DELIM", "next")
    monkeypatch.setenv("CHUNK_SKIP_WINDOW", "1")
    monkeypatch.setenv("CHUNK_INCLUDE_PARAMS", "false")
    monkeypatch.setenv("CHUNK_INCLUDE_CAPTION_INDICES", "false")
    monkeypatch.setenv("CHUNK_FAIL_FAST", "false")

    monkeypatch.setenv("PRE_MIN_LEN", "7")
    monkeypatch.setenv("PRE_DEDUPE_WINDOW_MS", "2222")

    # Refresh settings after env changes
    get_settings.cache_clear()

    pre = SRTPreprocessor()
    assert pre.min_len == 7
    assert pre.dedupe_window_ms == 2222

    chunk = SemanticChunker()
    # threshold stays a string as per Settings (component allows str|int|float)
    assert chunk.threshold == "0.75"
    assert chunk.chunk_size == 256
    assert chunk.similarity_window == 5
    assert chunk.min_sentences == 3
    assert chunk.min_characters_per_sentence == 42
    assert chunk.delim == [". ", "? "]
    assert chunk.include_delim == "next"
    assert chunk.skip_window == 1
    assert chunk.include_params is False
    assert chunk.include_caption_indices is False
    assert chunk.fail_fast is False

    # Explicit constructor args override env defaults
    chunk2 = SemanticChunker(chunk_size=1024, min_sentences=4, include_params=True)
    assert chunk2.chunk_size == 1024
    assert chunk2.min_sentences == 4
    assert chunk2.include_params is True
