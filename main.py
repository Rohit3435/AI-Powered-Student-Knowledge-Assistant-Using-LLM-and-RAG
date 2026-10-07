"""Simple terminal entry point for the local Mind Meshers RAG pipeline."""

from getpass import getpass

from llm.ollama import OllamaConnectionError, OllamaModelError
from rag.access import (
    asks_for_another_student,
    authenticate_student,
    is_student_email,
    student_name_for_email,
)
from rag.pipeline import answer_question, prepare_pipeline


# ==========================================
# 1. ASK ONE QUESTION FROM THE TERMINAL
# ==========================================

def main() -> None:
    """Prepare local search once, then answer questions until the user exits."""
    role = input("Continue as student or general user? [s/g]: ").strip().casefold()
    if role not in {"s", "student", "g", "general"}:
        print("Choose student or general user.")
        return
    student_email = ""
    student_name = None
    if role in {"s", "student"}:
        student_email = input("Student email: ").strip()
        if not is_student_email(student_email) or not authenticate_student(
            student_email, getpass("Password: ")
        ):
            print("Student email or password does not match the local login data.")
            return
        student_name = student_name_for_email(student_email)

    try:
        if student_name:
            pipeline = prepare_pipeline(
                database_path=f"chroma_db/student/{student_email.casefold().replace('@', '_at_')}",
                shared_documents_folder="knowledge_base",
                authorized_student_name=student_name,
            )
        else:
            pipeline = prepare_pipeline(
                excluded_file_name_keywords=("student", "login", "back"),
                allow_empty=True,
                shared_documents_folder="knowledge_base",
            )
    except FileNotFoundError as error:
        print(error)
        return
    except Exception as error:
        print(f"Could not prepare the local document search: {error}")
        print("Check that the required packages are installed and try again.")
        return

    loaded_files = sorted({
        document["metadata"]["file_name"] for document in pipeline["documents"]
    })
    print(f"Loaded {len(loaded_files)} document file(s): {', '.join(loaded_files)}")
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

        if student_name and asks_for_another_student(question, student_email):
            print("Access denied. Student records are private; ask about your own record and public policies.")
            continue

        try:
            result = answer_question(
                question,
                pipeline["collection"],
                pipeline["embedding_model"],
                documents=pipeline["documents"],
                authorized_student_name=student_name,
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
