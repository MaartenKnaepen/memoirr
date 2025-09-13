from src.components.embedder.text_embedder import TextEmbedder


def test_text_embedder_component_smoke(monkeypatch):
    # Monkeypatch the underlying Haystack embedder to avoid loading real models
    import src.components.embedder.text_embedder as mod

    class FakeEmbedder:
        def __init__(self, model=None):
            self.model = model
            self.warmed = False

        def warm_up(self):
            self.warmed = True

        def run(self, text):
            # Return a deterministic small vector
            return {"embedding": [0.1, -0.2, 0.3]}

    monkeypatch.setattr(mod, "SentenceTransformersTextEmbedder", FakeEmbedder)

    comp = TextEmbedder()
    out = comp.run("I love pizza!")

    assert "embedding" in out
    emb = out["embedding"]
    assert isinstance(emb, list)
    assert emb == [0.1, -0.2, 0.3]
