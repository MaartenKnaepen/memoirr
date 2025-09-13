from haystack.dataclasses.document import Document

from src.components.writer.qdrant_writer import QdrantWriter


def test_qdrant_writer_writes_documents(monkeypatch):
    # Arrange: fake settings and store
    import src.components.writer.qdrant_writer as mod

    class FakeSettings:
        qdrant_url = ":memory:"
        qdrant_collection = "test_collection"
        qdrant_recreate_index = True
        qdrant_return_embedding = True
        qdrant_wait_result = True

    written_payload = {}

    class FakeStore:
        def __init__(self, *args, **kwargs):
            # Capture init to verify settings were passed
            written_payload["init_args"] = args
            written_payload["init_kwargs"] = kwargs

        def write_documents(self, docs):
            written_payload["docs"] = docs

    monkeypatch.setattr(mod, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(mod, "QdrantDocumentStore", FakeStore)

    writer = QdrantWriter()

    docs_in = [
        {"content": "This is first", "embedding": [0.0] * 5},
        {"content": "This is second", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]},
    ]

    # Act
    out = writer.run(docs_in)

    # Assert init kwargs reflect settings
    init_kwargs = written_payload["init_kwargs"]
    assert init_kwargs["index"] == FakeSettings.qdrant_collection
    assert init_kwargs["recreate_index"] == FakeSettings.qdrant_recreate_index
    assert init_kwargs["return_embedding"] == FakeSettings.qdrant_return_embedding
    assert init_kwargs["wait_result_from_api"] == FakeSettings.qdrant_wait_result

    # Assert documents were converted to Haystack Document objects
    docs_written = written_payload["docs"]
    assert isinstance(docs_written, list) and len(docs_written) == 2
    assert all(isinstance(d, Document) for d in docs_written)
    assert docs_written[0].content == "This is first"
    assert docs_written[1].content == "This is second"

    # Assert output stats
    assert out["stats"]["written"] == 2
