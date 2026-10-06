"""Apply placement eligibility rules found in uploaded documents to student rows."""

import re
from difflib import SequenceMatcher
from typing import Any


# ==========================================
# 1. READ ELIGIBILITY RULES FROM DOCUMENT TEXT
# ==========================================

def _parse_company_rules(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find company rows in extracted policy text without hardcoding rule values."""
    rules = []
    for document in documents:
        lines = [line.strip() for line in document["text"].splitlines() if line.strip()]
        for index in range(max(0, len(lines) - 4)):
            company = lines[index]
            try:
                minimum_cgpa = float(lines[index + 1])
                attendance = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", lines[index + 2])
                maximum_backlogs = int(lines[index + 3])
            except (ValueError, IndexError):
                continue

            branches = [part.strip().upper() for part in lines[index + 4].split(",")]
            if attendance and branches and all(re.fullmatch(r"[A-Z]+", item) for item in branches):
                rules.append(
                    {
                        "company": company,
                        "minimum_cgpa": minimum_cgpa,
                        "minimum_attendance": float(attendance.group(1)),
                        "maximum_backlogs": maximum_backlogs,
                        "eligible_branches": branches,
                        "source": document["metadata"],
                    }
                )
    return rules


# ==========================================
# 2. FIND THE STUDENT AND COMPANY IN THE QUESTION
# ==========================================

def _parse_student_fields(text: str) -> dict[str, str]:
    """Read labeled CSV or spreadsheet fields from a loaded row."""
    field_pattern = re.compile(r"(?:^|\.\s*)([A-Za-z_]+):\s*(.*?)(?=\.\s+[A-Za-z_]+:|$)")
    return {key.casefold(): value.strip() for key, value in field_pattern.findall(text)}


def _company_from_question(question: str, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Match a mentioned company, allowing small spelling mistakes in its name."""
    words = re.findall(r"[a-z0-9]+", question.casefold())
    best_rule = None
    best_score = 0.0
    for rule in rules:
        company_words = rule["company"].casefold()
        if company_words in question.casefold():
            return rule
        for phrase_length in range(1, min(5, len(words) + 1)):
            for start in range(len(words) - phrase_length + 1):
                phrase = " ".join(words[start : start + phrase_length])
                score = SequenceMatcher(None, phrase, company_words).ratio()
                if score > best_score:
                    best_rule, best_score = rule, score

    return best_rule if best_score >= 0.80 else None


# ==========================================
# 3. CHECK THE STUDENT AGAINST THE POLICY
# ==========================================

def answer_eligibility_question(
    question: str, documents: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return a verified eligibility result when both student and policy data exist.

    The comparison is done with ordinary Python checks. The language model is
    not asked to guess thresholds or perform the eligibility calculation.
    Return None when the question or uploaded documents do not contain enough
    information, allowing the normal document retrieval path to handle it.
    """
    if not re.search(r"eligib|qualif", question, re.IGNORECASE):
        return None

    rules = _parse_company_rules(documents)
    rule = _company_from_question(question, rules)
    if not rule:
        return None

    student = None
    for document in documents:
        fields = _parse_student_fields(document["text"])
        name = fields.get("name", "")
        if name and name.casefold() in question.casefold():
            student = {"fields": fields, "source": document["metadata"]}
            break
    if not student:
        return None

    fields = student["fields"]
    try:
        cgpa = float(fields["cgpa"])
        attendance = float(fields["attendance_percent"])
        backlogs = int(float(fields["backlogs"]))
        branch = fields["branch"].upper()
    except (KeyError, ValueError):
        return None

    checks = [
        cgpa >= rule["minimum_cgpa"],
        attendance >= rule["minimum_attendance"],
        backlogs <= rule["maximum_backlogs"],
        branch in rule["eligible_branches"],
    ]
    eligible = all(checks)
    name = fields["name"]
    if eligible:
        answer = (
            f"Yes. {name} meets the listed requirements for {rule['company']}: "
            f"CGPA {cgpa:g}, attendance {attendance:g}%, {backlogs} backlogs, "
            f"and branch {branch}."
        )
    else:
        failed_requirements = []
        if cgpa < rule["minimum_cgpa"]:
            failed_requirements.append(
                f"CGPA {cgpa:g} is below the required {rule['minimum_cgpa']:g}"
            )
        if attendance < rule["minimum_attendance"]:
            failed_requirements.append(
                f"attendance {attendance:g}% is below the required {rule['minimum_attendance']:g}%"
            )
        if backlogs > rule["maximum_backlogs"]:
            failed_requirements.append(
                f"{backlogs} backlogs exceed the allowed {rule['maximum_backlogs']}"
            )
        if branch not in rule["eligible_branches"]:
            failed_requirements.append(f"branch {branch} is not listed as eligible")
        answer = f"No. {name} is not eligible for {rule['company']}: " + "; ".join(failed_requirements) + "."

    sources = []
    for metadata in (student["source"], rule["source"]):
        source = {key: metadata[key] for key in (
            "file_name", "page_number", "sheet_name", "row_number", "json_path"
        ) if key in metadata}
        if source and source not in sources:
            sources.append(source)

    return {"answer": answer, "sources": sources}
