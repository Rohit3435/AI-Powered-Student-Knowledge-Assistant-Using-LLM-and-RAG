"""Read local PDF files and keep useful source information with their text."""

from pathlib import Path
from typing import Any

from pypdf import PdfReader


# ==========================================
# 1. LOAD PDF DOCUMENTS
# ==========================================

def load_documents(folder_path: str | Path) -> list[dict[str, Any]]:
    """Return one record per readable PDF page in the given folder.

    Keeping one page per record lets later chunks retain the real page number.
    Invalid or scanned PDFs with no extractable text are skipped with a message.
    """
    folder = Path(folder_path)
    if not folder.exists():
        return []

    documents: list[dict[str, Any]] = []
    for pdf_path in sorted(folder.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(
                        {
                            "text": text,
                            "metadata": {
                                "file_name": pdf_path.name,
                                "page_number": page_number,
                            },
                        }
                    )
        except Exception as error:
            print(f"Could not read {pdf_path.name}: {error}")

    return documents
