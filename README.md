# Mind Meshers: Local RAG with Ollama

## Project Overview

This project can answer general questions with local Ollama, or answer questions from local PDF documents using retrieval augmented generation (RAG). RAG searches for relevant passages with embeddings and ChromaDB before asking Ollama to write a grounded answer. No university policy documents are included; add approved, real documents to `data/documents/` to use PDF search.

## Current Architecture

```text
PDF files in data/documents/
        ↓
PDF page text and source metadata (rag/loader.py)
        ↓
Word based chunks with overlap (rag/chunking.py)
        ↓
Sentence Transformer embeddings (rag/embeddings.py)
        ↓
Persistent local ChromaDB collection (rag/retriever.py)
        ↓
Top matching passages and citations (rag/pipeline.py)
        ↓
Ollama llama3.2 answer (llm/ollama.py)
        ↓
Terminal output (main.py)
```

## Folder Structure

```text
data/documents/       Add local PDF documents here
rag/                  Loading, chunking, embeddings, search, and pipeline
llm/                  Local Ollama integration
tests/                Tests that do not require a running Ollama service
main.py               Terminal question interface
requirements.txt      Current Python dependencies
AGENTS.md             Project-specific development guidance
```

The local ChromaDB files are created in `chroma_db/` and ignored by Git. Files placed in `data/documents/` are also ignored so private documents are not uploaded accidentally.

## Technologies Used

- Python
- Ollama and the `ollama` Python package
- ChromaDB for local vector storage and search
- `sentence-transformers` with `all-MiniLM-L6-v2` for embeddings
- `pypdf` for PDF text extraction
- `pytest` for tests

## Installation

Use Python 3.10 or newer. From the project folder, create and activate a virtual environment, then install the packages:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The embedding model downloads the first time it is used and is cached locally afterward. An internet connection is needed for that initial model download.

## Ollama Setup

Install Ollama for your operating system, start it, and download the default model:

```powershell
ollama pull llama3.2
```

The program connects to `http://localhost:11434`. This local setup does not require an API key. If the service is stopped or the model is missing, the program prints a helpful message.

## How to Run

1. Put text-based PDF files in `data/documents/`.
2. Start Ollama and make sure `llama3.2` is available.
3. In the project folder, run:

```powershell
python main.py
```

Choose general question mode to ask Ollama without PDFs, or PDF question mode to search your documents. In PDF mode, the app reads the PDFs and rebuilds the local index so removed or changed files do not leave stale search results.

## How RAG Works

RAG means retrieval augmented generation. The loader extracts each PDF page separately and keeps its filename and actual one-based page number. The chunker splits each page into groups of 500 words with 80 words repeated between adjacent groups. This keeps passages small enough to search while helping preserve meaning across chunk boundaries.

The embedding model turns every chunk and question into a list of numbers representing its meaning. ChromaDB stores those vectors with the original text and metadata. A question retrieves up to three nearby chunks; results that are too far away are ignored. If no relevant context is found, Ollama is not called.

## How Ollama Works

For general questions, the prompt asks local `llama3.2` to answer from general knowledge and say when it is uncertain. For PDF questions, retrieved passages and the question are sent to the model with instructions to answer from the supplied context, avoid unsupported claims, say when information was not found, and treat document text as data rather than instructions. Ollama generates answers on the local machine.

## How Citations Work

PDF grounded answers include sources from metadata attached when each PDF page is loaded. The terminal prints the original filename and page number for each retrieved source. If page metadata is unavailable, it prints only the filename. General answers do not use PDF sources. No source or page number is guessed.

## Example Question

After adding a real document that covers the topic, you could ask:

```text
What is the attendance requirement?
```

The answer and source details depend entirely on the content of your PDFs.

## Testing

Run the tests with:

```powershell
python -m pytest -q
```

The tests cover document loading behavior, empty folders, invalid PDFs, chunking and overlap, retrieval, empty questions, missing context, source metadata, pipeline setup, and prompt construction. Ollama is mocked so the tests do not require a running model service.

## Current Limitations

- PDF is the only supported document format.
- Scanned PDFs need OCR, which is not included.
- The program accepts one terminal question per run.
- The embedding model must be downloaded once before first use.
- Answer quality depends on the documents, retrieval results, and local model.
- The local database is rebuilt from all PDFs on each run.

## Future Integration

Streamlit, FastAPI, SQLite, LangGraph, authentication, deployment, and other application components will be integrated later. They are outside the scope of this local RAG and Ollama stage.
