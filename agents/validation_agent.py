"""Deterministic template validation agent. Never invents values."""

from __future__ import annotations

from typing import Any

from agents.state import IntakeState, Issue
from schema.template_schema import SHEET_SPECS, classify_sheet


def _issue(**kwargs: Any) -> Issue:
    return Issue(**kwargs)  # type: ignore[typeddict-item]


def validate_parsed_sheet(parsed: dict[str, Any]) -> tuple[dict[str, Any], list[Issue]]:
    issues: list[Issue] = []
    sheet = parsed.get("source_sheet") or ""
    spec = classify_sheet(sheet)

    if spec is None:
        issues.append(
            _issue(
                stage="validation",
                code="unknown_sheet",
                message=(
                    f"Sheet '{sheet}' is not a known UP template sheet. "
                    f"Known: {sorted(SHEET_SPECS)}"
                ),
                sheet=sheet,
            )
        )
        return {"ok": False, "checks": []}, issues

    checks: list[dict[str, Any]] = []

    # Required metadata
    if not parsed.get("instrument_name"):
        issues.append(
            _issue(
                stage="validation",
                code="missing_instrument_name",
                message="Instrument name (A1) is missing.",
                sheet=sheet,
                field="instrument_name",
            )
        )
    else:
        checks.append({"check": "instrument_name", "ok": True})

    if not parsed.get("description"):
        issues.append(
            _issue(
                stage="validation",
                code="missing_description",
                message="Pipemaker / type description line is missing.",
                sheet=sheet,
                field="description",
            )
        )
    else:
        checks.append({"check": "description", "ok": True})

    if parsed.get("units") not in {"imperial", "metric"}:
        issues.append(
            _issue(
                stage="validation",
                code="missing_units",
                message="Units must be imperial or metric.",
                sheet=sheet,
                field="units",
                value=parsed.get("units"),
            )
        )
    else:
        checks.append({"check": "units", "ok": True})

    # OAL labels
    oals = parsed.get("oals") or {}
    for label in spec.oal_labels:
        if label not in oals:
            issues.append(
                _issue(
                    stage="validation",
                    code="missing_oal_label",
                    message=f"Expected OAL label '{label}' in header block.",
                    sheet=sheet,
                    field=label,
                )
            )
        else:
            checks.append({"check": f"oal_label:{label}", "ok": True})

    sections = parsed.get("sections") or []
    if not sections:
        issues.append(
            _issue(
                stage="validation",
                code="no_sections",
                message="No data sections were parsed from the sheet.",
                sheet=sheet,
            )
        )

    for section in sections:
        name = section.get("name") or "main"
        if section.get("error"):
            issues.append(
                _issue(
                    stage="validation",
                    code="section_parse_error",
                    message=section["error"],
                    sheet=sheet,
                    section=name,
                )
            )
            continue
        if not section.get("headers_found", True):
            issues.append(
                _issue(
                    stage="validation",
                    code="missing_headers",
                    message=f"Section '{name}' is missing required column headers.",
                    sheet=sheet,
                    section=name,
                )
            )
            continue
        checks.append({"check": f"headers:{name}", "ok": True})

        rows = section.get("rows") or []
        # Tonehole sheets may be empty templates; bore sections with zero rows fail.
        # Folded bass-upper Reed Seat / U-bend blocks, and drone U3, may be empty.
        optional_names = set(spec.optional_sections) | {"U3"}
        if section.get("kind") in {"bore", "u_bend", "slide_brass_tube"} and not rows:
            if name in optional_names or section.get("optional"):
                checks.append({"check": f"empty_allowed:{name}", "ok": True})
            else:
                issues.append(
                    _issue(
                        stage="validation",
                        code="empty_section",
                        message=f"Section '{name}' has no data rows.",
                        sheet=sheet,
                        section=name,
                    )
                )
        else:
            checks.append(
                {
                    "check": f"row_count:{name}",
                    "ok": True,
                    "rows": len(rows),
                }
            )

        # Column presence on each row for required numeric-ish fields
        if section.get("kind") == "bore":
            for i, row in enumerate(rows):
                # A notes-only row may record a length or explanation without a diameter.
                if row.get("Diameter") is None and not row.get("Notes"):
                    issues.append(
                        _issue(
                            stage="validation",
                            code="missing_diameter",
                            message="Bore row missing Diameter.",
                            sheet=sheet,
                            section=name,
                            row=i,
                            field="Diameter",
                        )
                    )

    ok = not any(i.get("stage") == "validation" for i in issues)
    # also treat unknown issues collected above
    ok = len(issues) == 0
    return {"ok": ok, "checks": checks, "sheet": sheet, "spec": spec.name}, issues


def validation_node(state: IntakeState) -> dict[str, Any]:
    parsed = state.get("parsed")
    if not parsed:
        issue = _issue(
            stage="validation",
            code="no_parsed_payload",
            message="Validation cannot run without a parsed sheet payload.",
        )
        return {
            "validation_report": {"ok": False},
            "needs_human_review": True,
            "status": "needs_human_review",
            "issues": [issue],
            "messages": ["Validation skipped: nothing parsed."],
        }

    report, issues = validate_parsed_sheet(parsed)
    needs_review = not report.get("ok", False)
    return {
        "validation_report": report,
        "needs_human_review": needs_review,
        "status": "validated" if report.get("ok") else "needs_human_review",
        "issues": issues,
        "messages": [
            "Validation passed."
            if report.get("ok")
            else f"Validation failed with {len(issues)} issue(s)."
        ],
    }
