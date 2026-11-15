"""Deterministic writers for validated measurement batches."""

from __future__ import annotations

import json
from typing import Any

from db.connection import get_connection


def _row_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    for key in keys:
        if key in row:
            return row[key]
    return None


def _as_number(value: Any) -> float | None:
    """Return a float for numeric cells. Text markers such as 'through' are not numbers."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _nominal_for_storage(value: Any) -> float | None:
    """Store Nominal at 2 decimal places when the source has more."""
    number = _as_number(value)
    if number is None:
        return None
    return round(number, 2)


def _oval_for_storage(value: Any) -> tuple[float | None, str | None]:
    """Split Oval into a numeric measurement and an optional text marker."""
    number = _as_number(value)
    if number is not None:
        return number, None
    if value is None:
        return None, None
    text = str(value).strip()
    return None, text or None


def upsert_instrument(
    *,
    instrument_family: str,
    instrument_name: str,
    pipemaker: str | None,
    description: str | None,
    units: str | None,
    bore_type: str | None = None,
    musical_key: str | None = None,
) -> int:
    with get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO instruments (
                instrument_family,
                instrument_name,
                pipemaker,
                bore_type,
                musical_key,
                description,
                units
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (instrument_family, instrument_name, pipemaker, description)
            DO UPDATE SET
                units = COALESCE(EXCLUDED.units, instruments.units),
                bore_type = COALESCE(EXCLUDED.bore_type, instruments.bore_type),
                musical_key = COALESCE(EXCLUDED.musical_key, instruments.musical_key)
            RETURNING id
            """,
            (
                instrument_family,
                instrument_name,
                pipemaker,
                bore_type,
                musical_key,
                description,
                units if units in ("imperial", "metric") else None,
            ),
        ).fetchone()
        return int(row["id"])


def queue_human_review(
    *,
    batch_id: int | None,
    issue_stage: str,
    issues: list[dict[str, Any]],
    source_filename: str,
    source_sheet: str,
) -> int:
    """Create or reuse a draft batch, then enqueue human review."""
    with get_connection() as conn:
        if batch_id is None:
            batch = conn.execute(
                """
                INSERT INTO measurement_batches (
                    source_filename,
                    source_sheet,
                    status,
                    validation_report
                )
                VALUES (%s, %s, 'needs_human_review', %s::jsonb)
                RETURNING id
                """,
                (
                    source_filename,
                    source_sheet,
                    json.dumps({"issues": issues, "stage": issue_stage}),
                ),
            ).fetchone()
            batch_id = int(batch["id"])
        else:
            conn.execute(
                """
                UPDATE measurement_batches
                SET status = 'needs_human_review',
                    validation_report = %s::jsonb
                WHERE id = %s
                """,
                (json.dumps({"issues": issues, "stage": issue_stage}), batch_id),
            )

        review = conn.execute(
            """
            INSERT INTO human_review_queue (
                batch_id,
                issue_stage,
                issues,
                status
            )
            VALUES (%s, %s, %s::jsonb, 'open')
            RETURNING id
            """,
            (batch_id, issue_stage, json.dumps(issues)),
        ).fetchone()
        return int(review["id"])


def commit_batch(parsed_sheet: dict[str, Any], reports: dict[str, Any]) -> int:
    """
    Persist a fully validated sheet.

    parsed_sheet keys expected:
      source_filename, source_sheet, instrument_family, instrument_name,
      pipemaker, description, units, oals, sections
    """
    instrument_id = upsert_instrument(
        instrument_family=parsed_sheet["instrument_family"],
        instrument_name=parsed_sheet["instrument_name"],
        pipemaker=parsed_sheet.get("pipemaker"),
        description=parsed_sheet.get("description"),
        units=parsed_sheet.get("units"),
        bore_type=parsed_sheet.get("bore_type"),
        musical_key=parsed_sheet.get("musical_key"),
    )

    with get_connection() as conn:
        batch = conn.execute(
            """
            INSERT INTO measurement_batches (
                source_filename,
                source_sheet,
                instrument_id,
                status,
                measurement_method,
                collector,
                measurement_date,
                header_notes,
                validation_report,
                sanity_report,
                validated_at,
                committed_at
            )
            VALUES (
                %s, %s, %s, 'committed', %s, %s, %s, %s, %s::jsonb, %s::jsonb, now(), now()
            )
            RETURNING id
            """,
            (
                parsed_sheet["source_filename"],
                parsed_sheet["source_sheet"],
                instrument_id,
                parsed_sheet.get("measurement_method"),
                parsed_sheet.get("collector"),
                parsed_sheet.get("measurement_date"),
                parsed_sheet.get("header_notes"),
                json.dumps(reports.get("validation", {})),
                json.dumps(reports.get("sanity", {})),
            ),
        ).fetchone()
        batch_id = int(batch["id"])

        for section_name, oal in (parsed_sheet.get("oals") or {}).items():
            conn.execute(
                """
                INSERT INTO section_oals (batch_id, section_name, oal)
                VALUES (%s, %s, %s)
                """,
                (batch_id, section_name, oal),
            )

        for section in parsed_sheet.get("sections") or []:
            kind = section["kind"]
            section_name = section.get("name") or "main"
            rows = section.get("rows") or []

            if kind == "bore":
                for idx, row in enumerate(rows):
                    oval_number, oval_text = _oval_for_storage(
                        _row_value(row, "Oval", "oval")
                    )
                    conn.execute(
                        """
                        INSERT INTO bore_measurements (
                            batch_id, section_name, row_index,
                            nominal, diameter, round_val, oval, oval_text, notes
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            batch_id,
                            section_name,
                            idx,
                            _nominal_for_storage(_row_value(row, "Nominal", "nominal")),
                            _as_number(_row_value(row, "Diameter", "diameter")),
                            _as_number(_row_value(row, "Round", "round")),
                            oval_number,
                            oval_text,
                            _row_value(row, "Notes", "notes"),
                        ),
                    )
            elif kind == "toneholes":
                for idx, row in enumerate(rows):
                    conn.execute(
                        """
                        INSERT INTO tonehole_measurements (
                            batch_id, row_index, keyhole,
                            d_bottom_bell, d_top_reedseat, d_ctr_bell,
                            d_axial, d_xverse, chimney, ch_diam, notes
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            batch_id,
                            idx,
                            row.get("keyhole"),
                            row.get("d-bottom-bell"),
                            row.get("d-top-reedseat"),
                            row.get("d-ctr-bell"),
                            row.get("d-axial"),
                            row.get("d-xverse"),
                            row.get("chimney"),
                            row.get("ch_diam"),
                            row.get("notes"),
                        ),
                    )
            elif kind == "u_bend":
                for idx, row in enumerate(rows):
                    conn.execute(
                        """
                        INSERT INTO u_bend_measurements (
                            batch_id, section_name, row_index,
                            id_inner, od_x_axis, od_y_axis, od, wall,
                            diameter_od, radius, length, notes
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            batch_id,
                            section_name,
                            idx,
                            row.get("ID"),
                            row.get("OD x-axis"),
                            row.get("OD y-axis"),
                            row.get("OD"),
                            row.get("Wall"),
                            row.get("Diameter OD"),
                            row.get("Radius"),
                            row.get("Length"),
                            row.get("Notes"),
                        ),
                    )
            elif kind == "slide_brass_tube":
                for row in rows:
                    conn.execute(
                        """
                        INSERT INTO slide_brass_tube (
                            batch_id, id_inner, od, wall, length
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            batch_id,
                            row.get("ID"),
                            row.get("OD"),
                            row.get("Wall"),
                            row.get("Length"),
                        ),
                    )
            else:
                raise ValueError(f"Unknown section kind for commit: {kind}")

        return batch_id
