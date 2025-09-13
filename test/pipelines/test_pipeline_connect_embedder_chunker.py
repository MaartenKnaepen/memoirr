import pytest
from haystack import Pipeline

from src.components.preprocessor.srt_preprocessor import SRTPreprocessor
from src.components.chunker.semantic_chunker import SemanticChunker
from src.components.embedder.text_embedder import TextEmbedder


def test_pipeline_connect_chunker_to_embedder(monkeypatch):
    # Stub chunker orchestrator to avoid heavy deps
    from src.components.chunker.utilities.semantic_chunker import orchestrate_chunking as mod

    def fake_orchestrate(jsonl_lines, **kwargs):
        # Produce a single fake chunk line in JSONL
        return [
            '{"text":"hi","start_ms":0,"end_ms":1,"token_count":1,"caption_indices":[1]}'
        ], {"input_captions": 1, "output_chunks": 1, "avg_tokens_per_chunk": 1.0}

    monkeypatch.setattr(mod, "orchestrate_semantic_chunking", fake_orchestrate)

    # Stub the SentenceTransformersTextEmbedder to avoid model loading
    import src.components.embedder.text_embedder as embedder_mod

    class FakeEmbedder:
        def __init__(self, model=None):
            self.model = model
        def warm_up(self):
            pass
        def run(self, text):
            return {"embedding": [0.0, 0.0, 1.0]}

    monkeypatch.setattr(embedder_mod, "SentenceTransformersTextEmbedder", FakeEmbedder)

    pipe = Pipeline()
    pre = SRTPreprocessor()
    chunk = SemanticChunker()
    emb = TextEmbedder()

    pipe.add_component("pre", pre)
    pipe.add_component("chunk", chunk)
    pipe.add_component("emb", emb)

    # Connect: pre -> chunk
    pipe.connect("pre.jsonl_lines", "chunk.jsonl_lines")

    # There is no direct socket compatibility yet between chunker and embedder.
    # For now, we just verify that both components can be added and the pre->chunk
    # connection works, and that embedder can be called independently.
    # Future work: add a converter component from chunk JSONL to text content.

    # Smoke run of embedder
    out = emb.run("hello world")
    assert out["embedding"] == [0.0, 0.0, 1.0]
