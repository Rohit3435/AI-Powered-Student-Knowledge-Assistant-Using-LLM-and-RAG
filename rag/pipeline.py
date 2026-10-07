"""Connect retrieval to Ollama and prepare the local document index."""

from pathlib import Path
import re
from typing import Any, Callable

from llm.ollama import ask_ollama
from rag.chunking import split_documents
from rag.eligibility import answer_eligibility_question
from rag.embeddings import load_embedding_model
from rag.loader import load_documents
from rag.retriever import create_collection, retrieve_chunks, store_chunks


# ==========================================
# 1. LOAD DOCUMENTS AND BUILD THE LOCAL INDEX
# ==========================================

def prepare_pipeline(
    documents_folder: str | Path = "data/documents",
    database_path: str | Path = "chroma_db",
    excluded_file_name_keywords: tuple[str, ...] = (),
    allow_empty: bool = False,
    shared_documents_folder: str | Path | None = None,
    authorized_student_name: str | None = None,
) -> dict[str, Any]:
    """Read supported documents, optionally filter filenames, then index them."""
    documents = load_documents(documents_folder)
    # Never put login credentials in the search index. Student data files are
    # private; a student's index gets only that student's records.
    documents = [
        document for document in documents
        if "login" not in document["metadata"]["file_name"].casefold()
    ]
    private_documents = []
    for document in documents:
        filename = document["metadata"]["file_name"].casefold()
        if "student" not in filename and "back" not in filename:
            private_documents.append(document)
            continue
        if not authorized_student_name:
            continue
        name = authorized_student_name.casefold()
        # CSV and spreadsheet loaders produce one labeled row per record.
        if re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", document["text"].casefold()):
            private_documents.append(document)
            continue
        # A PDF page may contain a table of many students. Keep only the
        # matching student's block, starting at their ID where available.
        lines = document["text"].splitlines()
        student_id_match = re.search(r"student_id\s*:\s*([\w-]+)", document["text"], re.I)
        if student_id_match and name in document["text"].casefold():
            private_documents.append(document)
            continue
        if name in document["text"].casefold():
            name_index = next(i for i, line in enumerate(lines) if name in line.casefold())
            start = max((i for i, line in enumerate(lines[:name_index]) if re.fullmatch(r"STU\d+", line.strip(), re.I)), default=name_index)
            end = next((i for i in range(name_index + 1, len(lines)) if re.fullmatch(r"STU\d+", lines[i].strip(), re.I)), len(lines))
            private_documents.append({**document, "text": "\n".join(lines[start:end])})
    documents = private_documents
    excluded_keywords = tuple(keyword.casefold() for keyword in excluded_file_name_keywords)
    if excluded_keywords:
        documents = [
            document for document in documents
            if not any(
                keyword in document["metadata"]["file_name"].casefold()
                for keyword in excluded_keywords
            )
        ]
    if shared_documents_folder is not None:
        # Shared sources are appended after role-specific filtering so the
        # approved NSUT knowledge file remains available to both user types.
        documents.extend(load_documents(shared_documents_folder))
    if not documents and not allow_empty:
        raise FileNotFoundError(
            f"No readable documents were found in '{documents_folder}'. "
            "Add a supported PDF, CSV, Excel, JSON/JSONL, TXT, or Markdown file and run again."
        )

    chunks = split_documents(documents)
    # An empty role-specific index is valid when a user is allowed to ask only
    # about a narrower set of documents than the folder currently contains.
    embedding_model = load_embedding_model() if chunks else None
    client, collection = create_collection(database_path)
    store_chunks(collection, chunks, embedding_model)
    return {
        "client": client,
        "collection": collection,
        "embedding_model": embedding_model,
        "documents": documents,
    }


# ==========================================
# 2. RETRIEVE CONTEXT AND GENERATE AN ANSWER
# ==========================================

def answer_question(
    question: str,
    collection: Any,
    embedding_model: Any,
    answer_function: Callable[[str, str], str] = ask_ollama,
    documents: list[dict[str, Any]] | None = None,
    authorized_student_name: str | None = None,
) -> dict[str, Any]:
    """Retrieve matching passages, ask Ollama, and return answer plus sources."""
    if not question.strip():
        raise ValueError("Please enter a question.")

    # Some questions, such as placement eligibility, need exact comparisons
    # across a student table and a policy document. Check their actual values
    # directly instead of asking semantic search or the LLM to do arithmetic.
    if documents:
        eligibility_result = answer_eligibility_question(
            question, documents, authorized_student_name=authorized_student_name
        )
        if eligibility_result is not None:
            return eligibility_result

    matches = retrieve_chunks(collection, question, embedding_model)
    if not matches:
        return {
            "answer": "I could not find relevant information in the available documents.",
            "sources": [],
        }

    # Labels keep the model's input readable. They also reinforce that source
    # text is evidence to answer from, not a new instruction for the model.
    context_parts = []
    seen_context = set()
    for match in matches:
        metadata = match["metadata"]
        title = metadata.get("document_title") or metadata.get("file_name") or "Source document"
        location = []
        if metadata.get("page_number"):
            location.append(f"page {metadata['page_number']}")
        elif metadata.get("page_label"):
            location.append(f"page {metadata['page_label']}")
        if metadata.get("sheet_name"):
            location.append(f"sheet {metadata['sheet_name']}")
        if metadata.get("row_number"):
            location.append(f"row {metadata['row_number']}")
        source_label = title + (f", {', '.join(location)}" if location else "")
        passage_key = (source_label, match["text"])
        if passage_key not in seen_context:
            seen_context.add(passage_key)
            context_parts.append(f"[Source: {source_label}]\n{match['text']}")
    context = "\n\n".join(context_parts)
    answer = answer_function(question, context)

    sources = []
    seen = set()
    for match in matches:
        metadata = match["metadata"]
        # Only include source locations the loader actually knows. For example,
        # PDFs have page numbers, while spreadsheets have sheet and row names.
        source = {key: metadata[key] for key in (
            "file_name", "document_title", "doc_id", "page_number", "page_label",
            "section", "clause", "category", "source_url", "sheet_name", "row_number", "json_path"
        ) if key in metadata}
        source_key = tuple(sorted(source.items()))
        if source_key not in seen:
            seen.add(source_key)
            sources.append(source)

    return {"answer": answer, "sources": sources}
