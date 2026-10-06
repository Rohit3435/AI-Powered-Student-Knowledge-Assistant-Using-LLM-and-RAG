"""Simple terminal entry point for the local Mind Meshers RAG pipeline."""

from llm.ollama import OllamaConnectionError, OllamaModelError
from rag.pipeline import answer_question, prepare_pipeline


# ==========================================
# 1. ASK ONE QUESTION FROM THE TERMINAL
# ==========================================

def main() -> None:
    """Prepare local document search, ask one question, and print citations."""
    try:
        pipeline = prepare_pipeline()
    except FileNotFoundError as error:
        print(error)
        return
    except Exception as error:
        print(f"Could not prepare the local document search: {error}")
        print("Check that the required packages are installed and try again.")
        return

    try:
        question = input("Enter your question: ").strip()
        result = answer_question(
            question,
            pipeline["collection"],
            pipeline["embedding_model"],
        )
    except ValueError as error:
        print(error)
        return
    except OllamaModelError as error:
        print(error)
        return
    except OllamaConnectionError as error:
        print(error)
        return
    except Exception as error:
        print(f"Could not complete the search: {error}")
        return

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
