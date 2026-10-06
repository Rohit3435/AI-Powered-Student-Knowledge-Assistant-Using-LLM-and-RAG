"""Connect retrieval to Ollama and prepare the local document index."""

from pathlib import Path
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
) -> dict[str, Any]:
    """Read supported documents, create chunks and embeddings, and index them."""
    documents = load_documents(documents_folder)
    if not documents:
        raise FileNotFoundError(
            f"No readable documents were found in '{documents_folder}'. "
            "Add a supported PDF, CSV, Excel, JSON, TXT, or Markdown file and run again."
        )

    chunks = split_documents(documents)
    embedding_model = load_embedding_model()
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
) -> dict[str, Any]:
    """Retrieve matching passages, ask Ollama, and return answer plus sources."""
    if not question.strip():
        raise ValueError("Please enter a question.")

    # Some questions, such as placement eligibility, need exact comparisons
    # across a student table and a policy document. Check their actual values
    # directly instead of asking semantic search or the LLM to do arithmetic.
    if documents:
        eligibility_result = answer_eligibility_question(question, documents)
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
    context = "\n\n".join(match["text"] for match in matches)
    answer = answer_function(question, context)

    sources = []
    seen = set()
    for match in matches:
        metadata = match["metadata"]
        # Only include source locations the loader actually knows. For example,
        # PDFs have page numbers, while spreadsheets have sheet and row names.
        source = {key: metadata[key] for key in (
            "file_name", "page_number", "sheet_name", "row_number", "json_path"
        ) if key in metadata}
        source_key = tuple(sorted(source.items()))
        if source_key not in seen:
            seen.add(source_key)
            sources.append(source)

    return {"answer": answer, "sources": sources}
