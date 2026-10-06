"""Tests for PDF loading and chunking that do not need Ollama."""

import pytest

from rag.chunking import split_documents
from rag.loader import load_documents


def test_load_documents_keeps_filename_and_real_page_numbers(tmp_path, monkeypatch):
    (tmp_path / "rules.pdf").write_bytes(b"test placeholder")

    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        def __init__(self, _path):
            self.pages = [FakePage("First page"), FakePage("Second page")]

    monkeypatch.setattr("rag.loader.PdfReader", FakeReader)

    assert load_documents(tmp_path) == [
        {"text": "First page", "metadata": {"file_name": "rules.pdf", "page_number": 1}},
        {"text": "Second page", "metadata": {"file_name": "rules.pdf", "page_number": 2}},
    ]


def test_empty_document_folder_returns_no_documents(tmp_path):
    assert load_documents(tmp_path / "missing") == []


def test_invalid_pdf_is_skipped(tmp_path):
    (tmp_path / "broken.pdf").write_text("not a PDF", encoding="utf-8")
    assert load_documents(tmp_path) == []


def test_chunking_preserves_metadata_and_overlap():
    document = {
        "text": "one two three four five six seven eight nine ten",
        "metadata": {"file_name": "guide.pdf", "page_number": 2},
    }
    chunks = split_documents([document], chunk_size=4, overlap=1)

    assert [chunk["text"] for chunk in chunks] == [
        "one two three four",
        "four five six seven",
        "seven eight nine ten",
    ]
    assert chunks[1]["metadata"] == {"file_name": "guide.pdf", "page_number": 2}


def test_chunking_rejects_invalid_sizes():
    with pytest.raises(ValueError):
        split_documents([], chunk_size=0)
