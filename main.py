"""Simple terminal entry point for the local Mind Meshers RAG pipeline."""

from llm.ollama import (
    OllamaConnectionError,
    OllamaModelError,
    ask_general_ollama,
)
from rag.pipeline import answer_question, prepare_pipeline


# ==========================================
# 1. ASK ONE QUESTION FROM THE TERMINAL
# ==========================================

def main() -> None:
    """Let the user choose a general question or a PDF grounded question."""
    print("Choose how to ask your question:")
    print("1. General question using Ollama")
    print("2. Question from local PDF documents (RAG)")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        question = input("Enter your question: ").strip()
        try:
            answer = ask_general_ollama(question)
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
            print(f"Could not get an answer from Ollama: {error}")
            return

        print("\nGeneral answer (not based on local PDFs):")
        print(answer)
        return

    if mode != "2":
        print("Please choose 1 for a general question or 2 for PDF search.")
        return

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
            if source["page_number"] is not None:
                print(f"  Page {source['page_number']}")


if __name__ == "__main__":
    main()
