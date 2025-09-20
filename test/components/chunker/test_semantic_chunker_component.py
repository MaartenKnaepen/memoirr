from src.components.chunker.semantic_chunker import SemanticChunker


def test_component_instantiation_and_run_smoke(monkeypatch):
    # Minimal smoke test: provide tiny input and stub orchestrator to avoid heavy deps
    # We'll monkeypatch the orchestrate function used inside the component via the module alias it imports.
    from src.components.chunker.utilities.semantic_chunker import orchestrate_chunking as orchestrate_module

    def fake_orchestrate(jsonl_lines, **kwargs):
        return [
            '{"text":"hi","start_ms":0,"end_ms":1,"token_count":1,"caption_indices":[1]}'
        ], {"input_captions": 1, "output_chunks": 1, "avg_tokens_per_chunk": 1.0}

    monkeypatch.setattr(orchestrate_module, "orchestrate_semantic_chunking", fake_orchestrate)

    comp = SemanticChunker()
    out = comp.run(['{"text":"hi","start_ms":0,"end_ms":1,"caption_index":1}'])
    assert "chunk_lines" in out and "stats" in out
    assert len(out["chunk_lines"]) == 1
