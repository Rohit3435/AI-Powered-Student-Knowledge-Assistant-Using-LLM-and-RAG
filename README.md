# AI-Powered Student Knowledge Assistant

## Project Overview

A privacy focused student knowledge assistant that answers questions from local PDF, CSV, Excel, JSON, TXT, and Markdown documents using retrieval augmented generation (RAG) and a locally run large language model. It searches for relevant passages with embeddings and ChromaDB, then asks a local Ollama model to write a grounded answer. No university policy documents are included; add approved, real documents to `data/documents/`.

## Current Architecture

```text
PDF, CSV, Excel, JSON, TXT, or Markdown files in data/documents/
        ↓
Extracted text and source metadata (rag/loader.py)
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
Streamlit page (app.py) or terminal (main.py)
```

## Folder Structure

```text
data/documents/       Add supported source files here
rag/                  Loading, chunking, embeddings, search, and pipeline
llm/                  Local Ollama integration
tests/                Tests that do not require a running Ollama service
knowledge_base/       Shared, public NSUT knowledge chunks
app.py                Basic Streamlit question interface
main.py               Terminal question interface
requirements.txt      Current Python dependencies
AGENTS.md             Project-specific development guidance
```

The local ChromaDB files are created in `chroma_db/` and ignored by Git. Files placed in `data/documents/` are also ignored so private documents are not uploaded accidentally. The curated shared NSUT chunks live in `knowledge_base/` and are used by both login types.

## Technologies Used

- Python
- Ollama and the `ollama` Python package
- ChromaDB for local vector storage and search
- `sentence-transformers` with `all-MiniLM-L6-v2` for embeddings
- `pypdf` for PDF text extraction
- Python CSV and JSON readers
- `openpyxl` for `.xlsx` and `.xlsm` spreadsheets
- `xlrd` for older `.xls` spreadsheets
- `pytest` for tests

## Installation

Use Python 3.10 or newer. From the project folder, create and activate a virtual environment, then install the packages:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The embedding model downloads the first time it is used and is cached locally afterward. An internet connection is needed for that initial model download. Later runs use the cached model without contacting Hugging Face. The model is loaded into memory once when the app starts; leave the question session open to avoid loading it again between questions.

## Ollama Setup

Install Ollama for your operating system, start it, and download the default model:

```powershell
ollama pull llama3.2
```

The program connects to `http://localhost:11434`. This local setup does not require an API key. If the service is stopped or the model is missing, the program prints a helpful message.

## How to Run the Streamlit Page

1. Put supported files in `data/documents/`: PDF, CSV, `.xlsx`, `.xlsm`, `.xls`, JSON, TXT, or Markdown.
2. Start Ollama and make sure `llama3.2` is available.
3. In the project folder, run:

```powershell
python -m streamlit run app.py
```

The page offers two options:

- **Student login** checks the email and password against `data/documents/id_pass/MindMesh_30_Synthetic_Student_Login_Data.csv`. Student email addresses must end with `@nsut.ac.in`. The CSV needs `nsut_email` and `test_password` columns.
- **General user** can ask about the shared NSUT rules and notices, plus supported local files whose filenames do not contain `student`, `placement`, or `eligibility` (case-insensitive). General users do not search local student-record files. Student users can search all supported local files and the shared NSUT knowledge base.

Each role uses its own local ChromaDB index. **Sign out / change user** returns to the two options. Keep the credential CSV private; it is ignored by Git along with the other files under `data/documents/`.

The terminal interface is still available with `python main.py`.

## Shared NSUT Knowledge Base

The file `knowledge_base/NSUT_Student_Knowledge_Base_Chunks.jsonl` comes from the [NSUT Knowledge Base handoff on the `aditya` branch](https://github.com/Rohit3435/AI_HCL_TECH_MIND_MESHERS/tree/aditya/NSUT_Knowledge_Base_Handoff). The handoff provides JSONL chunks rather than the original PDFs. This project includes only chunks whose source document title contains `student`: four chunks from two regulations/notices. Both login types can search this file. Citations use the source document title, page, and section stored in the chunk metadata. The upstream metadata does not include a usable source URL for every chunk.

## How RAG Works

RAG means retrieval augmented generation. The loader extracts PDF pages separately, CSV rows, spreadsheet rows, JSON items/JSONL records, or text content. It keeps filenames and real source locations: PDF page, CSV row, Excel sheet and row, JSON array path, or the shared knowledge base document title, page, and section. The chunker splits extracted text into groups of 500 words with 80 words repeated between adjacent groups. This keeps passages small enough to search while helping preserve meaning across chunk boundaries.

The embedding model turns every chunk and question into a list of numbers representing its meaning. ChromaDB stores those vectors with the original text and metadata. A question retrieves up to three nearby chunks; results that are too far away are ignored. If no relevant context is found, Ollama is not called.

For placement eligibility questions, the app reads the student values from the uploaded table and the company thresholds from the uploaded policy. It compares CGPA, attendance, backlogs, and branch directly before generating an answer, so those numeric checks do not depend on the language model guessing.

## How Ollama Works

The retrieved passages and question are sent to the local `llama3.2` model. The prompt instructs it to answer from the supplied context, avoid unsupported claims, say when information was not found, and treat document text as data rather than instructions. Ollama generates the answer on the local machine.

## How Citations Work

Sources come from metadata attached while each file is loaded. The Streamlit page and terminal show the filename and available page, sheet, row, or JSON path for each retrieved source. Shared NSUT chunks cite their document title, page, and section. The interface only displays locations the loader actually has; it does not guess page or row numbers.

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

- Supported formats are PDF, CSV, Excel (`.xlsx`, `.xlsm`, `.xls`), JSON/JSONL, TXT, and Markdown.
- Scanned PDFs need OCR, which is not included.
- The app processes one question at a time.
- The embedding model must be downloaded once before first use.
- Answer quality depends on the documents, retrieval results, and local model.
- The local database is rebuilt from all supported files when the app starts.
