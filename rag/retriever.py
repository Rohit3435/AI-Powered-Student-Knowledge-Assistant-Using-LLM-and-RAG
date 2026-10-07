"""Store document chunks in local ChromaDB and search them by meaning."""

from pathlib import Path
from typing import Any

from rag.embeddings import create_embeddings


# ==========================================
# 1. PREPARE THE LOCAL CHROMADB COLLECTION
# ==========================================

def create_collection(database_path: str | Path, collection_name: str = "mind_meshers"):
    """Open a persistent local database and its cosine-similarity collection."""
    import chromadb

    client = chromadb.PersistentClient(path=str(database_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


# ==========================================
# 2. STORE CHUNKS AND THEIR EMBEDDINGS
# ==========================================

def store_chunks(collection: Any, chunks: list[dict[str, Any]], embedding_model: Any) -> None:
    """Replace the collection contents with the current document chunks."""
    # Rebuilding on startup prevents old copies of changed or removed PDFs from
    # appearing in search results. ChromaDB is only used as a local index here.
    existing_ids = collection.get(include=[]).get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)
    if not chunks:
        return

    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(embedding_model, texts)
    collection.add(
        ids=[f"chunk-{index}" for index in range(len(chunks))],
        documents=texts,
        embeddings=embeddings,
        metadatas=[chunk["metadata"] for chunk in chunks],
    )


# ==========================================
# 3. RETRIEVE THE MOST RELEVANT CHUNKS
# ==========================================

def retrieve_chunks(
    collection: Any,
    question: str,
    embedding_model: Any,
    top_k: int = 8,
    max_distance: float = 0.75,
    candidates_per_document: int = 3,
) -> list[dict[str, Any]]:
    """Return nearby chunks, including only results under a distance limit.

    Cosine distance is lower for more similar text. The limit allows the
    application to report that no useful context was found instead of sending
    clearly unrelated passages to the language model.
    """
    if not question.strip():
        raise ValueError("Please enter a question.")
    if top_k <= 0:
        return []
    if collection.count() == 0:
        return []

    question_embedding = create_embeddings(embedding_model, [question])[0]
    results = collection.query(
        query_embeddings=[question_embedding],
        # Fetch extra nearby chunks so one long document does not occupy every
        # available result before we can select evidence from other files.
        n_results=max(top_k * 4, top_k),
        include=["documents", "metadatas", "distances"],
    )

    candidates: list[dict[str, Any]] = []
    for text, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        if distance <= max_distance:
            candidates.append({"text": text, "metadata": metadata, "distance": distance})

    # Keep evidence from separate files in the context. A question often needs
    # one fact from a student record and another from a policy, so selecting
    # only the globally closest chunks can hide one of those documents.
    candidates.sort(key=lambda match: match["distance"])
    matches: list[dict[str, Any]] = []
    per_document: dict[str, int] = {}
    deferred: list[dict[str, Any]] = []
    for match in candidates:
        metadata = match["metadata"]
        document_key = str(
            metadata.get("file_name")
            or metadata.get("document_title")
            or metadata.get("doc_id")
            or "unknown document"
        ).casefold()
        if per_document.get(document_key, 0) < candidates_per_document:
            matches.append(match)
            per_document[document_key] = per_document.get(document_key, 0) + 1
        else:
            deferred.append(match)

    # If several documents have no match, prefer those before adding extra
    # chunks from a file already represented, while respecting the context cap.
    return (matches + deferred)[:top_k]
