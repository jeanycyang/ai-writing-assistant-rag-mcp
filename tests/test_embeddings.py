from shared import embeddings


def test_get_embedding_provider_reuses_single_instance(monkeypatch):
    created: list[str] = []

    class FakeProvider:
        pass

    def fake_constructor(settings):
        created.append(settings.embedding_model)
        return FakeProvider()

    monkeypatch.setattr(embeddings, "SentenceTransformerEmbeddingProvider", fake_constructor)
    monkeypatch.setattr(embeddings, "_provider_instance", None)

    first = embeddings.get_embedding_provider()
    second = embeddings.get_embedding_provider()

    assert first is second
    assert created == ["BAAI/bge-m3"]
