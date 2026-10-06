"""Split page text into smaller pieces that can be searched independently."""

from typing import Any


# ==========================================
# 1. SPLIT DOCUMENTS INTO CHUNKS
# ==========================================

def split_documents(
    documents: list[dict[str, Any]], chunk_size: int = 500, overlap: int = 80
) -> list[dict[str, Any]]:
    """Split document text into word-based chunks and keep its metadata.

    A chunk is a short section of a document. Smaller sections make it easier
    to retrieve only the passage that relates to a question. Overlap repeats a
    few words between neighboring chunks so a sentence at a boundary is not lost.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be zero or greater and smaller than chunk_size.")

    chunks: list[dict[str, Any]] = []
    step = chunk_size - overlap
    for document in documents:
        words = document["text"].split()
        for start in range(0, len(words), step):
            chunk_text = " ".join(words[start : start + chunk_size]).strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "metadata": dict(document["metadata"])})
            # The final full chunk already covers the remaining words. Avoid
            # creating a tiny duplicate chunk from only its overlap.
            if start + chunk_size >= len(words):
                break

    return chunks
