"""Simple terminal entry point for the local Mind Meshers RAG pipeline."""

from llm.ollama import OllamaConnectionError, OllamaModelError
from rag.pipeline import answer_question, prepare_pipeline


# ==========================================
# 1. ASK ONE QUESTION FROM THE TERMINAL
# ==========================================

def main() -> None:
    """Prepare local search once, then answer questions until the user exits."""
    try:
        pipeline = prepare_pipeline()
    except FileNotFoundError as error:
        print(error)
        return
    except Exception as error:
        print(f"Could not prepare the local document search: {error}")
        print("Check that the required packages are installed and try again.")
        return

    print("Ask questions about the loaded documents. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nEnter your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting the document question session.")
            break

        if question.lower() in {"exit", "quit"}:
            print("Exiting the document question session.")
            break

        try:
            result = answer_question(
                question,
                pipeline["collection"],
                pipeline["embedding_model"],
            )
        except ValueError as error:
            print(error)
            continue
        except (OllamaModelError, OllamaConnectionError) as error:
            print(error)
            continue
        except Exception as error:
            print(f"Could not complete the search: {error}")
            continue

        print("\nAnswer:")
        print(result["answer"])
        if result["sources"]:
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source['file_name']}")
                if "page_number" in source:
                    print(f"  Page {source['page_number']}")
                if "sheet_name" in source:
                    print(f"  Sheet: {source['sheet_name']}")
                if "row_number" in source:
                    print(f"  Row: {source['row_number']}")
                if "json_path" in source:
                    print(f"  JSON item: {source['json_path']}")


if __name__ == "__main__":
    main()
