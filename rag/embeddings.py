"""Create sentence embeddings for document chunks and questions."""

# ==========================================
# 1. LOAD THE LOCAL EMBEDDING MODEL
# ==========================================

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load a compact model that maps text to numeric meaning vectors.

    Embeddings are needed because ChromaDB compares meaning vectors to find
    text related to a question, even when the wording is not identical.
    The model is downloaded the first time and then cached by sentence-transformers.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def create_embeddings(model, texts: list[str]) -> list[list[float]]:
    """Convert a list of text passages into vectors for ChromaDB."""
    if not texts:
        return []
    vectors = model.encode(texts, convert_to_numpy=True)
    # sentence-transformers returns a NumPy array; accepting lists as well
    # keeps this small helper easy to exercise with a lightweight test double.
    return vectors.tolist() if hasattr(vectors, "tolist") else vectors
