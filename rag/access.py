"""Small local helpers for student sign-in and private-record access."""

import csv
import re
from pathlib import Path


STUDENT_CREDENTIALS_FILE = Path("data/documents/MindMesh_30_Synthetic_Student_Login_Data.csv")


def is_student_email(email: str) -> bool:
    """Accept only an email address at the configured university domain."""
    return bool(re.fullmatch(r"[^\s@]+@nsut\.ac\.in", email.strip(), re.I))


def _credential_rows():
    if not STUDENT_CREDENTIALS_FILE.is_file():
        return []
    with STUDENT_CREDENTIALS_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def authenticate_student(email: str, password: str) -> bool:
    """Check the provided credentials against the local login file."""
    return is_student_email(email) and any(
        (row.get("nsut_email") or "").strip().casefold() == email.strip().casefold()
        and (row.get("test_password") or "") == password
        for row in _credential_rows()
    )


def student_name_for_email(email: str) -> str | None:
    """Return the name linked to the signed-in user's email."""
    return next((
        (row.get("student_name") or "").strip()
        for row in _credential_rows()
        if (row.get("nsut_email") or "").strip().casefold() == email.strip().casefold()
    ), None)


def asks_for_another_student(question: str, email: str) -> bool:
    """Detect requests naming another student's private record."""
    own_name = student_name_for_email(email)
    if not own_name:
        return False
    return any(
        (row.get("student_name") or "").strip().casefold() in question.casefold()
        for row in _credential_rows()
        if (row.get("student_name") or "").strip()
        and (row.get("student_name") or "").strip().casefold() != own_name.casefold()
    )
