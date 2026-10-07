"""Tests for local student identity and per-student privacy checks."""

from rag.access import (
    asks_for_another_student,
    authenticate_student,
    is_student_email,
    student_name_for_email,
)


def test_local_student_identity_and_cross_student_access():
    email = "riya.mehta.ug23@nsut.ac.in"
    assert is_student_email(email)
    assert authenticate_student(email, "riya3435")
    assert student_name_for_email(email) == "Riya Mehta"
    assert not asks_for_another_student("What is my CGPA?", email)
    assert asks_for_another_student("What is Aarav Sharma's CGPA?", email)
