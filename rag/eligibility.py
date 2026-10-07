"""Check placement eligibility by combining private student and public policy data."""

import re
from difflib import SequenceMatcher
from typing import Any


def _parse_company_rules(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read company rows from the extracted policy table without fixed thresholds."""
    rules = []
    for document in documents:
        lines = [line.strip() for line in document["text"].splitlines() if line.strip()]
        for index, company in enumerate(lines):
            if not re.search(r"[A-Za-z]", company) or len(company) > 100:
                continue
            if index + 4 >= len(lines):
                continue
            try:
                cgpa = float(lines[index + 1])
                attendance_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", lines[index + 2])
                backlogs = int(float(lines[index + 3]))
            except ValueError:
                continue
            branches = [value.strip().upper() for value in lines[index + 4].split(",")]
            if attendance_match and branches and all(re.fullmatch(r"[A-Z]+", b) for b in branches):
                rules.append({
                    "company": company,
                    "minimum_cgpa": cgpa,
                    "minimum_attendance": float(attendance_match.group(1)),
                    "maximum_backlogs": backlogs,
                    "eligible_branches": branches,
                    "source": document["metadata"],
                })
    return rules


def _parse_student_fields(text: str) -> dict[str, str]:
    """Read labeled CSV or spreadsheet fields from one student record."""
    pattern = re.compile(r"(?:^|\.\s*)([A-Za-z_ ]+):\s*(.*?)(?=\.\s+[A-Za-z_ ]+:|$)")
    fields = {}
    for key, value in pattern.findall(text):
        fields[re.sub(r"\s+", "_", key.strip()).casefold()] = value.strip()
    return fields


def _company_from_question(question: str, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Match a company name mentioned in a question, including its short name."""
    normalized = question.casefold()
    best, score = None, 0.0
    words = re.findall(r"[a-z0-9]+", normalized)
    for rule in rules:
        name = rule["company"].casefold()
        short_name = re.sub(r"\s+(solutions|systems|analytics|technologies|labs|limited|ltd)\.?$", "", name)
        if name in normalized or short_name in normalized:
            return rule
        for size in range(1, min(5, len(words)) + 1):
            for start in range(len(words) - size + 1):
                candidate = " ".join(words[start:start + size])
                ratio = max(
                    SequenceMatcher(None, candidate, short_name).ratio(),
                    SequenceMatcher(None, candidate, name).ratio(),
                )
                if ratio > score:
                    best, score = rule, ratio
    return best if score >= 0.80 else None


def _source(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return only source details that the loader actually recorded."""
    keys = ("file_name", "document_title", "page_number", "page_label", "sheet_name", "row_number", "json_path")
    return {key: metadata[key] for key in keys if key in metadata and metadata[key]}


def answer_eligibility_question(
    question: str,
    documents: list[dict[str, Any]],
    authorized_student_name: str | None = None,
) -> dict[str, Any] | None:
    """Answer policy and eligibility questions using all authorized document records.

    Eligibility is calculated in Python. Missing required criteria are reported
    as unknown; they never turn into a positive eligibility decision.
    """
    asks_eligibility = bool(re.search(
        r"eligib|qualif|apply|satisf|meet.{0,30}(requirement|criteria|minimum|threshold)|"
        r"(requirement|criteria|minimum|threshold).{0,30}meet",
        question, re.I,
    ))
    asks_policy = bool(re.search(r"what.{0,25}(cgpa|attendance|backlog|branch)|requirement|criteria", question, re.I))
    if not asks_eligibility and not asks_policy:
        return None

    rules = _parse_company_rules(documents)
    rule = _company_from_question(question, rules)
    if not rule:
        return None

    # Policy-only questions do not need or expose any student record.
    if not asks_eligibility:
        answer = (
            f"{rule['company']} requires a CGPA of at least {rule['minimum_cgpa']:g}, "
            f"attendance of at least {rule['minimum_attendance']:g}%, no more than "
            f"{rule['maximum_backlogs']} backlogs, and an eligible branch: "
            f"{', '.join(rule['eligible_branches'])}."
        )
        return {"answer": answer, "sources": [_source(rule["source"])]}

    student_name = authorized_student_name
    if not student_name:
        student_names = [
            fields["name"] for document in documents
            if (fields := _parse_student_fields(document["text"])).get("name")
        ]
        student_name = next((name for name in student_names if name.casefold() in question.casefold()), None)
        if not student_name:
            # Natural questions often use a possessive first name, such as
            # "Does Riya's CGPA meet the requirement?" Match only when unique.
            possessive_names = [
                name for name in student_names
                if re.search(rf"\b{re.escape(name.split()[0])}['’]s\b", question, re.I)
            ]
            if len(possessive_names) == 1:
                student_name = possessive_names[0]
    if not student_name:
        return None

    student = next((
        {"fields": fields, "source": document["metadata"]}
        for document in documents
        if (fields := _parse_student_fields(document["text"])).get("name", "").casefold() == student_name.casefold()
    ), None)
    if not student:
        return {"answer": f"I could not find an authorized student record for {student_name}.", "sources": [_source(rule["source"])]}

    fields = student["fields"]
    criteria = [
        ("CGPA", fields.get("cgpa"), rule["minimum_cgpa"], lambda value, limit: value >= limit,
         f"at least {rule['minimum_cgpa']:g}"),
        ("attendance", fields.get("attendance_percent"), rule["minimum_attendance"], lambda value, limit: value >= limit,
         f"at least {rule['minimum_attendance']:g}%"),
        ("backlogs", fields.get("backlogs"), rule["maximum_backlogs"], lambda value, limit: value <= limit,
         f"no more than {rule['maximum_backlogs']}"),
        ("branch", fields.get("branch"), rule["eligible_branches"], lambda value, allowed: value.upper() in allowed,
         f"one of {', '.join(rule['eligible_branches'])}"),
    ]
    results, missing, failures = [], [], []
    for label, raw, limit, compare, requirement in criteria:
        if raw is None or not raw.strip():
            missing.append(label)
            continue
        try:
            value: Any = raw.upper() if label == "branch" else float(raw)
            passed = compare(value, limit)
        except ValueError:
            missing.append(label)
            continue
        results.append(f"{label}: {'PASS' if passed else 'FAIL'} ({raw}; required {requirement})")
        if not passed:
            failures.append(label)

    sources = [_source(student["source"]), _source(rule["source"])]
    sources = [source for i, source in enumerate(sources) if source and source not in sources[:i]]
    if missing:
        answer = (
            f"Eligibility could not be completely verified for {student_name} and {rule['company']}. "
            f"I could verify {', '.join(results) if results else 'no criteria'}; missing required information: "
            f"{', '.join(missing)}."
        )
    elif failures:
        answer = f"No. {student_name} is not eligible for {rule['company']}. " + "; ".join(results) + "."
    else:
        answer = f"Yes. {student_name} is eligible for {rule['company']}. " + "; ".join(results) + "."
    return {"answer": answer, "sources": sources}
