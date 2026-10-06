"""Small retriever tests that do not need a ChromaDB server or Ollama."""

import pytest

from rag.retriever import create_collection, retrieve_chunks, store_chunks


class FakeEmbeddingModel:
    def encode(self, texts, convert_to_numpy=True):
        return [[float(len(text)), 1.0] for text in texts]


class FakeCollection:
    def __init__(self):
        self.items = []

    def get(self, include=None):
        return {"ids": [item["id"] for item in self.items]}

    def delete(self, ids):
        self.items = [item for item in self.items if item["id"] not in ids]

    def add(self, ids, documents, embeddings, metadatas):
        self.items.extend(
            {"id": identifier, "text": text, "embedding": embedding, "metadata": metadata}
            for identifier, text, embedding, metadata in zip(ids, documents, embeddings, metadatas)
        )

    def count(self):
        return len(self.items)

    def query(self, query_embeddings, n_results, include):
        selected = self.items[:n_results]
        return {
            "documents": [[item["text"] for item in selected]],
            "metadatas": [[item["metadata"] for item in selected]],
            "distances": [[0.2 for item in selected]],
        }


def test_store_and_retrieve_chunks():
    collection = FakeCollection()
    model = FakeEmbeddingModel()
    store_chunks(
        collection,
        [{"text": "Attendance details", "metadata": {"file_name": "rules.pdf", "page_number": 3}}],
        model,
    )

    matches = retrieve_chunks(collection, "attendance", model)

    assert matches[0]["text"] == "Attendance details"
    assert matches[0]["metadata"]["page_number"] == 3


def test_store_replaces_old_chunks():
    collection = FakeCollection()
    model = FakeEmbeddingModel()
    store_chunks(collection, [{"text": "Old", "metadata": {"file_name": "old.pdf"}}], model)
    store_chunks(collection, [{"text": "New", "metadata": {"file_name": "new.pdf"}}], model)

    assert [item["text"] for item in collection.items] == ["New"]


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="question"):
        retrieve_chunks(FakeCollection(), "  ", FakeEmbeddingModel())


def test_distant_results_are_filtered():
    collection = FakeCollection()
    collection.add(["one"], ["unrelated"], [[1.0]], [{"file_name": "x.pdf"}])
    collection.query = lambda **kwargs: {
        "documents": [["unrelated"]],
        "metadatas": [[{"file_name": "x.pdf"}]],
        "distances": [[0.9]],
    }

    assert retrieve_chunks(collection, "question", FakeEmbeddingModel()) == []


def test_real_chromadb_persists_and_searches_chunks(tmp_path):
    # This checks the real local ChromaDB API while using a tiny deterministic
    # embedding double, so it does not download or load a machine learning model.
    _client, collection = create_collection(tmp_path / "chroma")
    model = FakeEmbeddingModel()
    store_chunks(
        collection,
        [{"text": "Attendance details", "metadata": {"file_name": "rules.pdf", "page_number": 3}}],
        model,
    )

    matches = retrieve_chunks(collection, "attendance", model)

    assert len(matches) == 1
    assert matches[0]["text"] == "Attendance details"
    assert matches[0]["metadata"]["file_name"] == "rules.pdf"
