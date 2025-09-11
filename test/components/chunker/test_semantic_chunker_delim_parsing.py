import json
from src.components.chunker.semantic_chunker import SemanticChunker
from src.core.config import get_settings


def test_semantic_chunker_uses_plain_string_delim_when_json_invalid(monkeypatch):
    # Force CHUNK_DELIM to a non-JSON string; component should fall back to using it as-is
    monkeypatch.setenv("CHUNK_DELIM", "||")
    get_settings.cache_clear()

    comp = SemanticChunker()
    assert comp.delim == "||"


def test_semantic_chunker_uses_json_list_delim_when_valid(monkeypatch):
    monkeypatch.setenv("CHUNK_DELIM", '[". ", "? "]')
    get_settings.cache_clear()

    comp = SemanticChunker()
    assert comp.delim == [". ", "? "]
