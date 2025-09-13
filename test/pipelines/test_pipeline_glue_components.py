from src.pipelines.srt_to_qdrant import ChunkJsonlToTexts, BuildDocuments


def test_chunk_jsonl_to_texts_parses_and_preserves_meta():
    lines = [
        '{"text": "Hello world", "start_ms": 1000, "end_ms": 2000, "token_count": 3, "caption_indices": [1,2]}',
        '{"text": "Again", "start_ms": 2100, "end_ms": 2200}',
        'not a json line',  # should be skipped
        '{"no_text": true}',  # should be skipped
    ]

    comp = ChunkJsonlToTexts()
    out = comp.run(lines)

    assert out["texts"] == ["Hello world", "Again"]
    metas = out["metas"]
    assert isinstance(metas, list) and len(metas) == 2
    assert metas[0]["start_ms"] == 1000 and metas[0]["end_ms"] == 2000
    assert metas[0]["token_count"] == 3
    assert metas[0]["caption_indices"] == [1, 2]
    assert metas[1]["start_ms"] == 2100 and metas[1]["end_ms"] == 2200


def test_build_documents_aligns_inputs_and_constructs_dicts():
    texts = ["a", "bb", "ccc"]
    metas = [{"i": 0}, {"i": 1}, {"i": 2}]
    embeddings = [[0.1], [0.2], [0.3]]

    comp = BuildDocuments()
    out = comp.run(texts=texts, metas=metas, embeddings=embeddings)

    docs = out["documents"]
    assert isinstance(docs, list) and len(docs) == 3
    assert docs[0]["content"] == "a" and docs[0]["embedding"] == [0.1] and docs[0]["meta"] == {"i": 0}
    assert docs[2]["content"] == "ccc" and docs[2]["embedding"] == [0.3] and docs[2]["meta"] == {"i": 2}

    # Inputs of uneven length should truncate to the shortest
    out2 = comp.run(texts=texts[:2], metas=metas, embeddings=embeddings)
    assert len(out2["documents"]) == 2
