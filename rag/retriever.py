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
    top_k: int = 3,
    max_distance: float = 0.65,
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
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    matches: list[dict[str, Any]] = []
    for text, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        if distance <= max_distance:
            matches.append({"text": text, "metadata": metadata, "distance": distance})
    return matches
