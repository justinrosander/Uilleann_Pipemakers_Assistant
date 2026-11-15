"""Sanity-check agent: numeric consistency against template rules."""

from __future__ import annotations

from typing import Any

from agents.state import IntakeState, Issue


def _issue(**kwargs: Any) -> Issue:
    return Issue(**kwargs)  # type: ignore[typeddict-item]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def sanity_check_parsed(parsed: dict[str, Any]) -> tuple[dict[str, Any], list[Issue]]:
    issues: list[Issue] = []
    checks: list[dict[str, Any]] = []
    sheet = parsed.get("source_sheet") or ""

    # OAL values should be positive when present
    for label, oal in (parsed.get("oals") or {}).items():
        if oal is None:
            # Allowed for optional OALs (e.g. Bass U3 / Bass Toneholes)
            checks.append({"check": f"oal_optional:{label}", "ok": True})
            continue
        if not _is_number(oal) or oal <= 0:
            issues.append(
                _issue(
                    stage="sanity",
                    code="invalid_oal",
                    message=f"OAL '{label}' must be a positive number.",
                    sheet=sheet,
                    field=label,
                    value=oal,
                )
            )
        else:
            checks.append({"check": f"oal_positive:{label}", "ok": True, "value": oal})

    for section in parsed.get("sections") or []:
        name = section.get("name") or "main"
        kind = section.get("kind")
        rows = section.get("rows") or []

        if kind == "bore":
            has_round = any("Round" in r for r in rows)
            for i, row in enumerate(rows):
                diameter = row.get("Diameter")
                round_val = row.get("Round")
                oval = row.get("Oval")

                if diameter is not None and not _is_number(diameter):
                    # allow only if somehow text; treat as error
                    issues.append(
                        _issue(
                            stage="sanity",
                            code="non_numeric_diameter",
                            message="Diameter must be numeric.",
                            sheet=sheet,
                            section=name,
                            row=i,
                            field="Diameter",
                            value=diameter,
                        )
                    )
                elif _is_number(diameter) and diameter <= 0:
                    issues.append(
                        _issue(
                            stage="sanity",
                            code="non_positive_diameter",
                            message="Diameter must be > 0.",
                            sheet=sheet,
                            section=name,
                            row=i,
                            field="Diameter",
                            value=diameter,
                        )
                    )

                # Length axis: Round (chanter/regulator) or Oval (drone)
                length_val = round_val if has_round else oval
                length_field = "Round" if has_round else "Oval"

                # Oval may be the text marker 'through'. It is stored in oval_text
                # and does not require a particular Notes value.
                if isinstance(oval, str) and oval.lower() == "through":
                    if has_round:
                        length_val = round_val
                elif length_val is not None and not _is_number(length_val):
                    # drone sheets keep Oval numeric; text oval is unexpected except notes
                    if not (isinstance(length_val, str) and length_val.lower() == "through"):
                        issues.append(
                            _issue(
                                stage="sanity",
                                code="non_numeric_length",
                                message=f"{length_field} must be numeric for length station.",
                                sheet=sheet,
                                section=name,
                                row=i,
                                field=length_field,
                                value=length_val,
                            )
                        )

                # Nominal may carry the maker's full-precision station, which
                # does not have to match Round or Oval. Extra digits are rounded
                # to 2 decimal places when the row is stored.

                if _is_number(diameter) and diameter > 2.0 and (
                    parsed.get("units") == "imperial"
                ):
                    issues.append(
                        _issue(
                            stage="sanity",
                            code="diameter_out_of_range",
                            message=(
                                "Imperial bore diameter > 2.0 inches looks implausible; "
                                "confirm units/value."
                            ),
                            sheet=sheet,
                            section=name,
                            row=i,
                            field="Diameter",
                            value=diameter,
                        )
                    )

            checks.append(
                {
                    "check": f"bore_rows:{name}",
                    "ok": True,
                    "rows": len(rows),
                }
            )

        elif kind == "u_bend":
            # Makers may record only the taper/length fields they can measure.
            # OD is an estimated roundness from the two axes, not a formula
            # we enforce. Notes stay free text. Blank cells are allowed.
            for i, row in enumerate(rows):
                for field in (
                    "ID",
                    "OD x-axis",
                    "OD y-axis",
                    "OD",
                    "Wall",
                    "Diameter OD",
                    "Radius",
                    "Length",
                ):
                    val = row.get(field)
                    if val is not None and not _is_number(val):
                        issues.append(
                            _issue(
                                stage="sanity",
                                code="non_numeric_u_field",
                                message=f"U-bend field {field} must be numeric when entered.",
                                sheet=sheet,
                                section=name,
                                row=i,
                                field=field,
                                value=val,
                            )
                        )
            checks.append({"check": f"u_bend:{name}", "ok": True, "rows": len(rows)})

        elif kind == "slide_brass_tube":
            for i, row in enumerate(rows):
                for field in ("ID", "OD", "Wall", "Length"):
                    val = row.get(field)
                    if val is not None and not _is_number(val):
                        issues.append(
                            _issue(
                                stage="sanity",
                                code="non_numeric_sbt_field",
                                message=f"Slide brass tube field {field} must be numeric.",
                                sheet=sheet,
                                section=name,
                                row=i,
                                field=field,
                                value=val,
                            )
                        )
            checks.append({"check": f"sbt:{name}", "ok": True, "rows": len(rows)})

        elif kind == "toneholes":
            checks.append(
                {"check": f"toneholes:{name}", "ok": True, "rows": len(rows)}
            )

    ok = len(issues) == 0
    return {"ok": ok, "checks": checks, "sheet": sheet}, issues


def sanity_node(state: IntakeState) -> dict[str, Any]:
    # Only run when validation passed (orchestrator routes), but stay defensive
    parsed = state.get("parsed")
    if not parsed:
        issue = _issue(
            stage="sanity",
            code="no_parsed_payload",
            message="Sanity check cannot run without parsed data.",
        )
        return {
            "sanity_report": {"ok": False},
            "needs_human_review": True,
            "status": "needs_human_review",
            "issues": [issue],
            "messages": ["Sanity check skipped: nothing parsed."],
        }

    report, issues = sanity_check_parsed(parsed)
    needs_review = not report.get("ok", False)
    return {
        "sanity_report": report,
        "needs_human_review": needs_review,
        "status": "sanity_ok" if report.get("ok") else "needs_human_review",
        "issues": issues,
        "messages": [
            "Sanity checks passed."
            if report.get("ok")
            else f"Sanity checks failed with {len(issues)} issue(s)."
        ],
    }
