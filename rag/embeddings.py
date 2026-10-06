"""Create sentence embeddings for document chunks and questions."""

import os
from pathlib import Path

# ==========================================
# 1. LOAD THE LOCAL EMBEDDING MODEL
# ==========================================

def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load a compact model that maps text to numeric meaning vectors.

    Embeddings are needed because ChromaDB compares meaning vectors to find
    text related to a question, even when the wording is not identical.
    The model is downloaded the first time and then cached by sentence-transformers.
    """
    from huggingface_hub import try_to_load_from_cache

    # A cached model can be opened offline. This avoids contacting the Hub on
    # every new run, so users do not see the unauthenticated-request warning
    # after the first successful download.
    cached_model_config = try_to_load_from_cache(model_name, "modules.json")
    is_cached = isinstance(cached_model_config, str)
    model_location = str(Path(cached_model_config).parent) if is_cached else model_name

    if is_cached:
        # Prevent even lightweight Hub checks on later runs. The model files
        # are already local, so the embedding step has no reason to use network.
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        print("The embedding model is not cached yet; downloading it for first use.")

    from sentence_transformers import SentenceTransformer
    from transformers.utils import logging as transformer_logging

    # The weights still need to be read into memory once per app start, but the
    # large progress bar can make that normal local step look like a download.
    transformer_logging.disable_progress_bar()

    return SentenceTransformer(model_location, local_files_only=is_cached)


def create_embeddings(model, texts: list[str]) -> list[list[float]]:
    """Convert a list of text passages into vectors for ChromaDB."""
    if not texts:
        return []
    vectors = model.encode(texts, convert_to_numpy=True)
    # sentence-transformers returns a NumPy array; accepting lists as well
    # keeps this small helper easy to exercise with a lightweight test double.
    return vectors.tolist() if hasattr(vectors, "tolist") else vectors
