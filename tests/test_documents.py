"""Tests for document loading and chunking that do not need Ollama."""

import csv
import json
import sys
from types import SimpleNamespace

import pytest
from openpyxl import Workbook

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


def test_csv_rows_keep_file_and_row_number(tmp_path):
    csv_path = tmp_path / "courses.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Course", "Credits"])
        writer.writerow(["Biology", "4"])

    assert load_documents(tmp_path) == [
        {
            "text": "Course: Biology. Credits: 4",
            "metadata": {"file_name": "courses.csv", "row_number": 2},
        }
    ]


def test_json_array_items_keep_json_paths(tmp_path):
    (tmp_path / "students.json").write_text(
        json.dumps([{"name": "Asha"}, {"name": "Ravi"}]), encoding="utf-8"
    )

    documents = load_documents(tmp_path)

    assert len(documents) == 2
    assert documents[1]["metadata"] == {"file_name": "students.json", "json_path": "$[1]"}
    assert '"name": "Ravi"' in documents[1]["text"]


def test_xlsx_rows_keep_sheet_and_row_number(tmp_path):
    excel_path = tmp_path / "schedule.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Semester 1"
    sheet.append(["Subject", "Room"])
    sheet.append(["Physics", "B-12"])
    workbook.save(excel_path)

    assert load_documents(tmp_path) == [
        {
            "text": "Subject: Physics. Room: B-12",
            "metadata": {
                "file_name": "schedule.xlsx",
                "sheet_name": "Semester 1",
                "row_number": 2,
            },
        }
    ]


def test_xls_rows_keep_sheet_and_row_number(tmp_path, monkeypatch):
    xls_path = tmp_path / "legacy.xls"
    xls_path.write_bytes(b"test workbook placeholder")

    class FakeSheet:
        name = "Results"
        nrows = 2

        def row_values(self, row_index):
            return [["Name", "Status"], ["Mina", "Enrolled"]][row_index]

    class FakeWorkbook:
        def sheets(self):
            return [FakeSheet()]

        def release_resources(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "xlrd",
        SimpleNamespace(open_workbook=lambda *_args, **_kwargs: FakeWorkbook()),
    )

    assert load_documents(tmp_path) == [
        {
            "text": "Name: Mina. Status: Enrolled",
            "metadata": {
                "file_name": "legacy.xls",
                "sheet_name": "Results",
                "row_number": 2,
            },
        }
    ]


def test_text_files_keep_filename_without_invented_location(tmp_path):
    (tmp_path / "notice.txt").write_text("Bring your student ID.", encoding="utf-8")

    assert load_documents(tmp_path) == [
        {"text": "Bring your student ID.", "metadata": {"file_name": "notice.txt"}}
    ]


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
