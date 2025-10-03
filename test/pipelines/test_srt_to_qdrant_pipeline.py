import json
from pathlib import Path
from src.pipelines.srt_to_qdrant import build_srt_to_qdrant_pipeline


def test_end_to_end_pipeline_writes_embedded_chunks(monkeypatch):
    # Patch chunker orchestrator to produce deterministic simple chunks
    from src.components.chunker.utilities.semantic_chunker import orchestrate_chunking as orch_mod

    def fake_orchestrate(jsonl_lines, **kwargs):
        # Emit a single chunk that concatenates all texts, preserve first/last times
        texts = []
        start = None
        end = None
        for line in jsonl_lines:
            obj = json.loads(line)
            texts.append(obj.get("text", ""))
            start = obj.get("start_ms") if start is None else start
            end = obj.get("end_ms") or end
        out = {
            "text": " ".join(texts),
            "start_ms": start or 0,
            "end_ms": end or 0,
            "token_count": 1,
            "caption_indices": list(range(1, len(texts) + 1)),
        }
        return [json.dumps(out)], {"input_captions": len(jsonl_lines), "output_chunks": 1}

    monkeypatch.setattr(orch_mod, "orchestrate_semantic_chunking", fake_orchestrate)

    # Patch TextEmbedder to avoid loading a real model
    import src.components.embedder.text_embedder as embed_mod

    class FakeDocumentEmbedder:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, documents):
            # Return documents with embeddings based on content length
            for doc in documents:
                doc.embedding = [float(len(doc.content))]
            return {"documents": documents}

        def warm_up(self):
            pass

    monkeypatch.setattr(embed_mod, "SentenceTransformersDocumentEmbedder", lambda *a, **k: FakeDocumentEmbedder())
    
    # Patch resolve_model_path in the embedder module where it's imported
    from src.components.embedder import text_embedder
    monkeypatch.setattr(text_embedder, "resolve_model_path", lambda x: Path("/fake/model/path"))

    # Patch QdrantWriter store to capture docs instead of hitting Qdrant
    import src.components.writer.qdrant_writer as writer_mod

    written = {}

    class FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def write_documents(self, docs):
            written["docs"] = docs

    monkeypatch.setattr(writer_mod, "QdrantDocumentStore", FakeStore)

    # Patch language detection to ensure English text passes through
    import src.components.preprocessor.utilities.srt_preprocessor.language_filter as lang_filter
    monkeypatch.setattr(lang_filter, "is_english_text", lambda text: True)  # Always consider text as English

    pipe = build_srt_to_qdrant_pipeline()

    srt_text = (
        "1\n00:00:01,000 --> 00:00:02,000\nHello!\n\n"
        "2\n00:00:02,200 --> 00:00:03,000\nAgain.\n\n"
    )

    out = pipe.run({"pre": {"srt_text": srt_text}})
    # The writer should have received documents equal to the number of produced chunks (1 per fake orchestrator)
    assert out["write"]["stats"]["written"] == 1
    assert "docs" in written and len(written["docs"]) == 1

    # Validate that content and meta look reasonable
    doc = written["docs"][0]
    assert isinstance(doc.content, str) and len(doc.content) > 0
    assert isinstance(doc.embedding, list) and len(doc.embedding) == 1
