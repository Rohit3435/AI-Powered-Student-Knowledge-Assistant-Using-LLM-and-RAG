"""Tests for the RAG flow that use local doubles instead of external services."""

import pytest

from rag.pipeline import answer_question, prepare_pipeline


class FakeCollection:
    def __init__(self, matches):
        self.matches = matches

    def count(self):
        return len(self.matches)

    def query(self, **_kwargs):
        return {
            "documents": [[match["text"] for match in self.matches]],
            "metadatas": [[match["metadata"] for match in self.matches]],
            "distances": [[0.1 for _match in self.matches]],
        }


class FakeEmbeddingModel:
    def encode(self, texts, convert_to_numpy=True):
        return [[1.0, 2.0] for _text in texts]


def test_basic_pipeline_passes_context_and_returns_citations():
    collection = FakeCollection(
        [
            {
                "text": "Students need 75 percent attendance.",
                "metadata": {"file_name": "attendance.pdf", "page_number": 3},
            }
        ]
    )
    calls = []

    def fake_answer(question, context):
        calls.append((question, context))
        return "The document says 75 percent."

    result = answer_question("What is the requirement?", collection, FakeEmbeddingModel(), fake_answer)

    assert calls == [("What is the requirement?", "Students need 75 percent attendance.")]
    assert result == {
        "answer": "The document says 75 percent.",
        "sources": [{"file_name": "attendance.pdf", "page_number": 3}],
    }


def test_missing_context_does_not_call_language_model():
    result = answer_question(
        "What is the rule?", FakeCollection([]), FakeEmbeddingModel(),
        lambda *_args: pytest.fail("Ollama should not be called without context"),
    )

    assert result["sources"] == []
    assert "could not find" in result["answer"]


def test_empty_question_is_rejected_before_retrieval():
    with pytest.raises(ValueError, match="question"):
        answer_question(" ", FakeCollection([]), FakeEmbeddingModel())


def test_answer_sources_keep_non_pdf_locations():
    collection = FakeCollection(
        [
            {
                "text": "The lab is in room B-12.",
                "metadata": {
                    "file_name": "schedule.xlsx",
                    "sheet_name": "Labs",
                    "row_number": 8,
                },
            }
        ]
    )

    result = answer_question(
        "Where is the lab?", collection, FakeEmbeddingModel(), lambda *_args: "Room B-12."
    )

    assert result["sources"] == [
        {"file_name": "schedule.xlsx", "sheet_name": "Labs", "row_number": 8}
    ]


def test_prepare_pipeline_reports_empty_document_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="No readable documents"):
        prepare_pipeline(tmp_path, tmp_path / "chroma")


def test_prepare_pipeline_loads_and_indexes_documents(tmp_path, monkeypatch):
    calls = []
    fake_collection = object()
    fake_model = object()
    monkeypatch.setattr(
        "rag.pipeline.load_documents",
        lambda _folder: [{"text": "Some policy text", "metadata": {"file_name": "policy.pdf", "page_number": 1}}],
    )
    monkeypatch.setattr("rag.pipeline.load_embedding_model", lambda: fake_model)
    monkeypatch.setattr("rag.pipeline.create_collection", lambda _path: ("client", fake_collection))
    monkeypatch.setattr("rag.pipeline.store_chunks", lambda collection, chunks, model: calls.append((collection, chunks, model)))

    result = prepare_pipeline(tmp_path, tmp_path / "chroma")

    assert result == {"client": "client", "collection": fake_collection, "embedding_model": fake_model}
    assert calls[0][0] is fake_collection
    assert calls[0][2] is fake_model
    assert calls[0][1][0]["metadata"]["file_name"] == "policy.pdf"
