"""A Streamlit page with student and general-user access to local documents."""

import csv
import re
from pathlib import Path

import streamlit as st

from llm.ollama import OllamaConnectionError, OllamaModelError
from rag.pipeline import answer_question, prepare_pipeline


st.set_page_config(page_title="Mind Meshers", page_icon=":books:")

STUDENT_CREDENTIALS_FILE = Path("data/documents/id_pass/MindMesh_30_Synthetic_Student_Login_Data.csv")
PERSONAL_STUDENT_DATA_QUESTION = re.compile(
    r"\b(student\s+(?:id|name|record|profile|details?)|list\s+of\s+students)\b",
    flags=re.IGNORECASE,
)


def is_nsut_student_email(email: str) -> bool:
    """Accept only a complete email address ending in the NSUT domain."""
    email = email.strip()
    return bool(re.fullmatch(r"[^\s@]+@nsut\.ac\.in", email, flags=re.IGNORECASE))


def authenticate_student(email: str, password: str) -> bool:
    """Check a student's email and password against the local credentials CSV."""
    if not is_nsut_student_email(email) or not STUDENT_CREDENTIALS_FILE.is_file():
        return False

    with STUDENT_CREDENTIALS_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        credentials = csv.DictReader(file)
        return any(
            (row.get("nsut_email") or "").strip().casefold() == email.strip().casefold()
            and (row.get("test_password") or "") == password
            for row in credentials
        )


def show_sign_in() -> None:
    """Offer separate student login and general-user access options."""
    st.title("Mind Meshers")
    st.write("Choose how you want to use the document question page.")
    option = st.radio("Continue as", ["Student login", "General user"], horizontal=True)

    if option == "Student login":
        with st.form("student_sign_in_form"):
            email = st.text_input("Student email ID", placeholder="name@nsut.ac.in")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Student sign in", type="primary")

        if submitted:
            if not is_nsut_student_email(email):
                st.error("Invalid ID. Student email IDs must end with `@nsut.ac.in`.")
            elif authenticate_student(email, password):
                st.session_state["user_role"] = "student"
                st.session_state["user_email"] = email.strip()
                st.rerun()
            else:
                st.error("Student email ID or password does not match the student data file.")
    else:
        st.info("General users can ask questions only about general information related to NSUT.")
        if st.button("Continue as general user", type="primary"):
            st.session_state["user_role"] = "general"
            st.session_state["user_email"] = ""
            st.rerun()


@st.cache_resource(show_spinner="Reading your documents and preparing local search...")
def get_pipeline(user_role: str):
    """Prepare a separate local index for each access role."""
    if user_role == "general":
        return prepare_pipeline(
            "data/documents",
            "chroma_db/general",
            excluded_file_name_keywords=("student", "placement", "eligibility"),
            allow_empty=True,
            shared_documents_folder="knowledge_base",
        )
    return prepare_pipeline(
        "data/documents",
        "chroma_db/student",
        shared_documents_folder="knowledge_base",
    )


def show_question_page() -> None:
    """Show the signed-in user a simple question form and its answer."""
    user_role = st.session_state["user_role"]
    email = st.session_state.get("user_email", "")
    title_col, signout_col = st.columns([4, 1])
    with title_col:
        st.title("Ask your documents")
        if user_role == "student":
            st.caption(f"Student access | {email}")
        else:
            st.caption("General user access")
    with signout_col:
        if st.button("Sign out / change user"):
            st.session_state.pop("user_role", None)
            st.session_state.pop("user_email", None)
            st.rerun()

    try:
        pipeline = get_pipeline(user_role)
    except FileNotFoundError as error:
        st.error(str(error))
        st.info("Add your files to `data/documents/`, then refresh this page.")
        return
    except Exception as error:
        st.error(f"Could not prepare local document search: {error}")
        return

    file_names = sorted({doc["metadata"]["file_name"] for doc in pipeline["documents"]})
    if file_names:
        st.caption("Available files: " + ", ".join(file_names))
    else:
        st.info("No general NSUT information files are available yet.")

    with st.form("question_form"):
        question = st.text_input("Your question", placeholder="What do the documents say about?")
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        if user_role == "general" and PERSONAL_STUDENT_DATA_QUESTION.search(question):
            st.warning(
                "This knowledge base does not provide individual student records. "
                "General users can ask about NSUT rules and notices instead."
            )
            return

        try:
            result = answer_question(
                question,
                pipeline["collection"],
                pipeline["embedding_model"],
                documents=pipeline["documents"],
            )
        except ValueError as error:
            st.warning(str(error))
            return
        except (OllamaConnectionError, OllamaModelError) as error:
            st.error(str(error))
            return
        except Exception as error:
            st.error(f"Could not answer the question: {error}")
            return

        st.subheader("Answer")
        st.write(result["answer"])
        if result["sources"]:
            st.subheader("Sources")
            citations = {}
            for source in result["sources"]:
                source_title = source.get("document_title") or source["file_name"]
                page = source.get("page_number") or source.get("page_label")
                section = source.get("section", "")
                sheet_name = source.get("sheet_name", "")
                row_number = source.get("row_number", "")
                json_path = source.get("json_path", "")
                citation_key = (source_title, page, section, sheet_name, row_number, json_path)
                citation = citations.setdefault(
                    citation_key,
                    {
                        "title": source_title,
                        "page": page,
                        "section": section,
                        "sheet_name": sheet_name,
                        "row_number": row_number,
                        "json_path": json_path,
                        "clauses": [],
                    },
                )
                clause = source.get("clause")
                if clause and clause not in citation["clauses"]:
                    citation["clauses"].append(clause)

            for citation in citations.values():
                details = []
                if citation["page"]:
                    details.append(f"page {citation['page']}")
                if citation["section"]:
                    details.append(citation["section"])
                if citation["clauses"]:
                    details.append(", ".join(citation["clauses"]))
                if citation["sheet_name"]:
                    details.append(f"sheet {citation['sheet_name']}")
                if citation["row_number"]:
                    details.append(f"row {citation['row_number']}")
                if citation["json_path"]:
                    details.append(citation["json_path"])
                suffix = f" ({'; '.join(details)})" if details else ""
                st.write(f"- **{citation['title']}**{suffix}")


if "user_role" not in st.session_state:
    show_sign_in()
else:
    show_question_page()
