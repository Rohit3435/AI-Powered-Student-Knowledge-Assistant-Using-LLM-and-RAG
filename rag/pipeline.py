"""Connect retrieval to Ollama and prepare the local document index."""

from pathlib import Path
from typing import Any, Callable

from llm.ollama import ask_ollama
from rag.chunking import split_documents
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
    """Read PDFs, create chunks and embeddings, and store them in ChromaDB."""
    documents = load_documents(documents_folder)
    if not documents:
        raise FileNotFoundError(
            f"No readable PDF text was found in '{documents_folder}'. "
            "Add a text-based PDF to data/documents and run again."
        )

    chunks = split_documents(documents)
    embedding_model = load_embedding_model()
    client, collection = create_collection(database_path)
    store_chunks(collection, chunks, embedding_model)
    return {"client": client, "collection": collection, "embedding_model": embedding_model}


# ==========================================
# 2. RETRIEVE CONTEXT AND GENERATE AN ANSWER
# ==========================================

def answer_question(
    question: str,
    collection: Any,
    embedding_model: Any,
    answer_function: Callable[[str, str], str] = ask_ollama,
) -> dict[str, Any]:
    """Retrieve matching passages, ask Ollama, and return answer plus sources."""
    if not question.strip():
        raise ValueError("Please enter a question.")

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
        source = (metadata.get("file_name"), metadata.get("page_number"))
        if source not in seen:
            seen.add(source)
            sources.append({"file_name": source[0], "page_number": source[1]})

    return {"answer": answer, "sources": sources}
