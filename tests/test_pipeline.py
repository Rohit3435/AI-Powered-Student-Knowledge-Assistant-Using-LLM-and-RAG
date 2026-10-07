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

    assert calls == [(
        "What is the requirement?",
        "[Source: attendance.pdf, page 3]\nStudents need 75 percent attendance.",
    )]
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


def test_combined_context_labels_each_source_for_the_language_model():
    captured = []
    collection = FakeCollection([
        {"text": "Riya CGPA is 8.4.", "metadata": {"file_name": "students.csv", "row_number": 3}},
        {"text": "TechNova requires a 7.0 CGPA.", "metadata": {"file_name": "policy.pdf", "page_number": 1}},
    ])
    answer_question(
        "Does Riya qualify?", collection, FakeEmbeddingModel(),
        lambda _question, context: captured.append(context) or "Answer",
    )
    assert "[Source: students.csv, row 3]" in captured[0]
    assert "[Source: policy.pdf, page 1]" in captured[0]


def test_eligibility_uses_uploaded_student_row_and_policy_rules():
    documents = [
        {
            "text": (
                "Company\nMinimum CGPA\nMinimum Attendance\nMaximum Backlogs\nEligible Branches\n"
                "FinSecure Technologies\n8.0\n70%\n0\nCSE, ECE\n"
            ),
            "metadata": {"file_name": "eligibility.pdf", "page_number": 1},
        },
        {
            "text": (
                "student_id: STU001. name: Aarav Sharma. branch: ECE. semester: 6. "
                "attendance_percent: 82.5. cgpa: 7.8. backlogs: 0. attendance_status: Eligible"
            ),
            "metadata": {"file_name": "students.csv", "row_number": 2},
        },
    ]

    result = answer_question(
        "Is Aarav Sharma eligible for infosec technologies?",
        FakeCollection([]),
        FakeEmbeddingModel(),
        lambda *_args: pytest.fail("Verified eligibility should not use LLM arithmetic"),
        documents=documents,
    )

    assert "Aarav Sharma is not eligible for FinSecure Technologies" in result["answer"]
    assert "CGPA: FAIL" in result["answer"]
    assert result["sources"] == [
        {"file_name": "students.csv", "row_number": 2},
        {"file_name": "eligibility.pdf", "page_number": 1},
    ]


def test_eligibility_combines_current_student_csv_with_company_policy():
    from rag.loader import load_documents

    documents = load_documents("data/documents")
    result = answer_question(
        "Is Riya Mehta eligible for TechNova?",
        FakeCollection([]),
        FakeEmbeddingModel(),
        lambda *_args: pytest.fail("Deterministic eligibility must not depend on Ollama"),
        documents=documents,
    )

    assert "Riya Mehta is eligible for TechNova Solutions" in result["answer"]
    assert any(source.get("row_number") == 3 for source in result["sources"])
    assert any(source.get("page_number") == 1 for source in result["sources"])


def test_eligibility_reports_missing_required_student_fields():
    documents = [
        {
            "text": "Company\nMinimum CGPA\nMinimum Attendance\nMaximum Backlogs\nEligible Branches\nTechNova Solutions\n7.0\n60%\n1\nCSE, ECE, IT",
            "metadata": {"file_name": "policy.pdf", "page_number": 1},
        },
        {
            "text": "name: Riya Mehta. cgpa: 8.4",
            "metadata": {"file_name": "students.csv", "row_number": 3},
        },
    ]
    result = answer_question(
        "Is Riya Mehta eligible for TechNova?", FakeCollection([]), FakeEmbeddingModel(),
        lambda *_args: pytest.fail("Missing criteria must not be guessed"), documents=documents,
    )
    assert "could not be completely verified" in result["answer"]
    assert "attendance" in result["answer"]


def test_prepare_pipeline_keeps_only_authorized_student_and_public_policy(tmp_path, monkeypatch):
    all_documents = [
        {"text": "name: Riya Mehta. cgpa: 8.4", "metadata": {"file_name": "students.csv", "row_number": 3}},
        {"text": "name: Aarav Sharma. cgpa: 7.8", "metadata": {"file_name": "students.csv", "row_number": 2}},
        {"text": "secret: password", "metadata": {"file_name": "login_data.xlsx", "row_number": 2}},
        {"text": "TechNova requires 7.0 CGPA", "metadata": {"file_name": "placement_policy.pdf", "page_number": 1}},
    ]
    monkeypatch.setattr("rag.pipeline.load_documents", lambda _folder: all_documents)
    monkeypatch.setattr("rag.pipeline.load_embedding_model", lambda: "model")
    monkeypatch.setattr("rag.pipeline.create_collection", lambda _path: ("client", "collection"))
    monkeypatch.setattr("rag.pipeline.store_chunks", lambda *_args: None)

    result = prepare_pipeline(tmp_path, tmp_path / "chroma", authorized_student_name="Riya Mehta")

    filenames = [doc["metadata"]["file_name"] for doc in result["documents"]]
    assert filenames == ["students.csv", "placement_policy.pdf"]
    assert "Riya Mehta" in result["documents"][0]["text"]


def test_policy_question_returns_policy_source_without_student_record():
    documents = [{
        "text": "Company\nMinimum CGPA\nMinimum Attendance\nMaximum Backlogs\nEligible Branches\nTechNova Solutions\n7.0\n60%\n1\nCSE, ECE, IT",
        "metadata": {"file_name": "policy.pdf", "page_number": 1},
    }]
    result = answer_question(
        "What CGPA does TechNova require?", FakeCollection([]), FakeEmbeddingModel(),
        lambda *_args: pytest.fail("Policy values should be read from the policy"), documents=documents,
    )
    assert "7" in result["answer"]
    assert result["sources"] == [{"file_name": "policy.pdf", "page_number": 1}]


def test_student_comparison_question_uses_deterministic_eligibility_check():
    from rag.loader import load_documents

    result = answer_question(
        "Does Riya's CGPA meet TechNova's requirement?",
        FakeCollection([]), FakeEmbeddingModel(),
        lambda *_args: pytest.fail("A comparison should use the deterministic check"),
        documents=load_documents("data/documents"),
    )
    assert "Riya Mehta is eligible for TechNova Solutions" in result["answer"]


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

    assert result == {
        "client": "client",
        "collection": fake_collection,
        "embedding_model": fake_model,
        "documents": [{
            "text": "Some policy text",
            "metadata": {"file_name": "policy.pdf", "page_number": 1},
        }],
    }
    assert calls[0][0] is fake_collection
    assert calls[0][2] is fake_model
    assert calls[0][1][0]["metadata"]["file_name"] == "policy.pdf"
