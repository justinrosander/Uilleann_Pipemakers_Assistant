"""Intake agent: parse spreadsheet workbooks into structured sheet payloads.

Supports Excel (.xlsx/.xlsm/.xltx/.xltm) via openpyxl and OpenDocument
(.ods) via odfpy. PDF / handwritten intake is stubbed for a later phase.
Validation stays deterministic and never invents measurement values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from agents.ods_reader import list_ods_sheets, read_ods_sheet_rows
from agents.state import IntakeState
from schema.template_schema import (
    BASS_UPPER_U_BEND_HEADERS,
    SLIDE_BRASS_HEADERS,
    U_BEND_HEADERS,
    classify_sheet,
    normalize_header,
)

EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xltx", ".xltm"}
ODS_SUFFIXES = {".ods"}
SUPPORTED_SUFFIXES = EXCEL_SUFFIXES | ODS_SUFFIXES


def _cell_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_number(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    # simple fraction a/b or mixed n a/b
    if "/" in text:
        try:
            parts = text.replace("-", " ").split()
            total = 0.0
            for part in parts:
                if "/" in part:
                    num, den = part.split("/", 1)
                    total += float(num) / float(den)
                else:
                    total += float(part)
            return total
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _find_header_row(rows: list[tuple[Any, ...]], expected: tuple[str, ...]) -> int | None:
    expected_norm = [h.lower() for h in expected]
    for idx, row in enumerate(rows):
        values = [normalize_header(c).lower() for c in row[: len(expected)]]
        if values == expected_norm:
            return idx
    return None


def _split_legacy_description(description: str) -> dict[str, str | None]:
    """Split an older one-line maker string into name, bore type, and key.

    Examples:
      'A. Maker Wide Bore D Chanter' -> A. Maker, Wide, D
      'A. Maker D Tenor Drone' -> A. Maker, None, D
      'A. Maker Narrow Bore D chanter' -> A. Maker, Narrow, D
    """
    text = description.strip()
    for marker, kind in ((" wide bore", "Wide"), (" narrow bore", "Narrow")):
        lowered = text.lower()
        idx = lowered.find(marker)
        if idx == -1:
            continue
        pipemaker = text[:idx].strip()
        rest = text[idx + len(marker) :].strip()
        token = rest.split(" ", 1)[0] if rest else ""
        musical_key = token.upper() if len(token) == 1 and token.isalpha() else None
        return {
            "pipemaker": pipemaker,
            "bore_type": kind,
            "musical_key": musical_key,
        }
    parts = text.split(" ")
    for index, part in enumerate(parts):
        if index > 0 and len(part) == 1 and part.isalpha():
            return {
                "pipemaker": " ".join(parts[:index]).strip(),
                "bore_type": None,
                "musical_key": part.upper(),
            }
    return {"pipemaker": text, "bore_type": None, "musical_key": None}


def _compose_description(
    pipemaker: str | None,
    bore_type: str | None,
    musical_key: str | None,
    instrument_name: str | None,
) -> str | None:
    parts: list[str] = []
    if pipemaker:
        parts.append(pipemaker)
    if bore_type:
        parts.append(f"{bore_type} Bore")
    if musical_key:
        parts.append(musical_key)
    if instrument_name:
        parts.append(instrument_name)
    return " ".join(parts) or None


def _parse_header_fields(text: str) -> dict[str, str | None]:
    """Split a header line such as 'Method: ...; Collector: ...; Date: ...; Notes: ...'."""
    import re

    fields: dict[str, str | None] = {
        "measurement_method": None,
        "collector": None,
        "measurement_date": None,
        "header_notes": None,
    }
    label_to_key = {
        "method": "measurement_method",
        "collector": "collector",
        "date": "measurement_date",
        "notes": "header_notes",
    }
    pattern = re.compile(
        r"(?:^|;)\s*(Method|Collector|Date|Notes)\s*:\s*",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    for index, match in enumerate(matches):
        key = label_to_key[match.group(1).lower()]
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[start:end].strip().strip(";").strip()
        if value:
            fields[key] = value
    return fields


def _header_fields_from_rows(rows: list[tuple[Any, ...]]) -> dict[str, str | None]:
    """Collect Method, Collector, Date, and Notes from the header area when present."""
    found: dict[str, str | None] = {
        "measurement_method": None,
        "collector": None,
        "measurement_date": None,
        "header_notes": None,
    }
    label_to_key = {
        "method": "measurement_method",
        "collector": "collector",
        "date": "measurement_date",
        "notes": "header_notes",
    }
    for row in rows[:15]:
        if not row:
            continue
        for col, raw in enumerate(row[:4]):
            text = _cell_str(raw)
            if not text:
                continue
            lower = text.lower()
            parsed = _parse_header_fields(text)
            if any(parsed.values()):
                for key, value in parsed.items():
                    if value and not found[key]:
                        found[key] = value
                continue
            # Label in this cell, value in the next cell.
            bare = lower.rstrip(":").strip()
            if bare in label_to_key and col + 1 < len(row):
                value = _cell_str(row[col + 1])
                key = label_to_key[bare]
                if value and not found[key]:
                    found[key] = value
    return found


def _parse_metadata(rows: list[tuple[Any, ...]]) -> dict[str, Any]:
    instrument_name = _cell_str(rows[0][0]) if rows else None
    description = None
    pipemaker = None
    bore_type = None
    musical_key = None
    method_block = None
    header_fields = _header_fields_from_rows(rows)
    units = None
    oals: dict[str, float | None] = {}

    header = _cell_str(rows[1][0]) if len(rows) > 1 and rows[1] else None
    if header and header.lower() == "pipemaker" and len(rows) > 2 and rows[2]:
        # Current template: row 2 labels, row 3 values (pipemaker, bore type, key).
        pipemaker = _cell_str(rows[2][0])
        bore_type = _cell_str(rows[2][1]) if len(rows[2]) > 1 else None
        musical_key = _cell_str(rows[2][2]) if len(rows[2]) > 2 else None
        description = _compose_description(
            pipemaker, bore_type, musical_key, instrument_name
        )
    elif len(rows) > 2:
        # Older template: one description line in A3.
        description = _cell_str(rows[2][0]) if rows[2] else None
        if description:
            split = _split_legacy_description(description)
            pipemaker = split["pipemaker"]
            bore_type = split["bore_type"]
            musical_key = split["musical_key"]

    for row in rows[:25]:
        label = _cell_str(row[0]) if row else None
        if not label:
            continue
        lower = label.lower()
        if lower.startswith("method:"):
            method_block = label
        if lower.startswith("units:"):
            if "imperial" in lower:
                units = "imperial"
            elif "metric" in lower:
                units = "metric"
        if lower.endswith("oal") or lower == "oal":
            value = _as_number(row[1]) if len(row) > 1 else None
            oals[label] = value

    return {
        "instrument_name_cell": instrument_name,
        "description": description,
        "pipemaker": pipemaker,
        "bore_type": bore_type,
        "musical_key": musical_key,
        "method_block": method_block or header_fields.get("measurement_method"),
        "measurement_method": header_fields.get("measurement_method"),
        "collector": header_fields.get("collector"),
        "measurement_date": header_fields.get("measurement_date"),
        "header_notes": header_fields.get("header_notes"),
        "units": units,
        "oals": oals,
    }


def _rows_matrix(ws) -> list[tuple[Any, ...]]:
    return [tuple(row) for row in ws.iter_rows(values_only=True)]


def list_workbook_sheets(path: Path | str) -> list[str]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in ODS_SUFFIXES:
        return list_ods_sheets(path)
    if suffix in EXCEL_SUFFIXES:
        wb = load_workbook(path, read_only=True)
        try:
            return list(wb.sheetnames)
        finally:
            wb.close()
    raise ValueError(f"Unsupported workbook type for sheet listing: {suffix}")


def load_sheet_rows(path: Path | str, sheet_name: str) -> list[tuple[Any, ...]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in ODS_SUFFIXES:
        return read_ods_sheet_rows(path, sheet_name)
    if suffix in EXCEL_SUFFIXES:
        wb = load_workbook(path, data_only=True)
        try:
            if sheet_name not in wb.sheetnames:
                raise FileNotFoundError(
                    f"Sheet '{sheet_name}' not found in {path.name}. "
                    f"Available: {list(wb.sheetnames)}"
                )
            return _rows_matrix(wb[sheet_name])
        finally:
            wb.close()
    raise ValueError(f"Unsupported workbook type for parsing: {suffix}")


def _read_table(
    rows: list[tuple[Any, ...]],
    header_idx: int,
    headers: tuple[str, ...],
    stop_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    stop_labels = {s.lower() for s in (stop_labels or set())}
    out: list[dict[str, Any]] = []
    for row in rows[header_idx + 1 :]:
        first = _cell_str(row[0]) if row else None
        if first and first.lower() in stop_labels:
            break
        # blank row ends a contiguous block when next section labels appear later
        if all(c is None or str(c).strip() == "" for c in row[: len(headers)]):
            # peek: allow single blank spacer inside? treat blank as end
            break
        # skip if this looks like another header row
        norm = [normalize_header(c) for c in row[: len(headers)]]
        if [h.lower() for h in norm] == [h.lower() for h in headers]:
            break

        record: dict[str, Any] = {}
        any_value = False
        for i, header in enumerate(headers):
            raw = row[i] if i < len(row) else None
            if header.lower() in {
                "notes",
                "keyhole",
            } or header in {"Notes", "keyhole"}:
                record[header] = _cell_str(raw)
            elif header == "Nominal":
                # may be numeric
                num = _as_number(raw)
                record[header] = num if num is not None else _cell_str(raw)
            else:
                # numeric preferred; keep text for rare Oval='through' style cells
                num = _as_number(raw)
                if num is not None:
                    record[header] = num
                else:
                    text = _cell_str(raw)
                    record[header] = text
                    if text:
                        any_value = True
                        continue
            if record[header] is not None:
                any_value = True
        if any_value:
            out.append(record)
    return out


def _canonical_section_label(label: str) -> str:
    """Strip a trailing '(optional)' marker from a section or OAL label."""
    text = label.strip()
    suffix = "(optional)"
    if text.lower().endswith(suffix):
        text = text[: -len(suffix)].strip()
    return text


def _find_section_blocks(
    rows: list[tuple[Any, ...]],
    section_names: list[str],
    headers: tuple[str, ...],
) -> list[dict[str, Any]]:
    name_map = {n.lower(): n for n in section_names}
    blocks: list[dict[str, Any]] = []
    indices = []
    for idx, row in enumerate(rows):
        label = _cell_str(row[0]) if row else None
        if not label:
            continue
        key = _canonical_section_label(label).lower()
        if key in name_map:
            indices.append((idx, name_map[key]))

    for i, (idx, name) in enumerate(indices):
        # header should be next non-empty structural row
        header_idx = None
        for j in range(idx + 1, min(idx + 5, len(rows))):
            vals = [normalize_header(c) for c in rows[j][: len(headers)]]
            if [v.lower() for v in vals] == [h.lower() for h in headers]:
                header_idx = j
                break
        if header_idx is None:
            blocks.append(
                {
                    "name": name,
                    "kind": "bore",
                    "headers_found": False,
                    "rows": [],
                    "error": f"Missing headers {headers} under section {name}",
                }
            )
            continue
        stop = {n.lower() for _, n in indices[i + 1 :]} | {
            "slide brass tube",
            "upper bore",
            "lower bore",
        }
        # don't stop on own name
        stop.discard(name.lower())
        data = _read_table(rows, header_idx, headers, stop_labels=stop)
        blocks.append(
            {
                "name": name,
                "kind": "bore",
                "headers_found": True,
                "headers": list(headers),
                "rows": data,
            }
        )
    return blocks


def _parse_bass_upper_bore(
    rows: list[tuple[Any, ...]],
    payload: dict[str, Any],
    bore_headers: tuple[str, ...],
) -> None:
    """Parse a Bass Upper sheet, including optional folded Reed Seat and U-bend.

    Builds that are not folded may omit those sections entirely. An empty
    optional section on the template is also valid. The upper bore itself
    remains required.
    """
    reed_blocks = _find_section_blocks(rows, ["Reed Seat"], bore_headers)
    for block in reed_blocks:
        block["optional"] = True
    payload["sections"].extend(reed_blocks)

    u_blocks = _find_section_blocks(rows, ["U-bend"], BASS_UPPER_U_BEND_HEADERS)
    for block in u_blocks:
        block["kind"] = "u_bend"
        block["optional"] = True
    payload["sections"].extend(u_blocks)

    upper_blocks = _find_section_blocks(rows, ["Upper Bore"], bore_headers)
    if upper_blocks:
        payload["sections"].extend(upper_blocks)
        return

    # Legacy layout: one bore table and no section labels.
    header_idx = _find_header_row(rows, bore_headers)
    if header_idx is None:
        payload["parse_warnings"].append(
            f"Could not locate bore headers {bore_headers}."
        )
        return
    data = _read_table(rows, header_idx, bore_headers)
    payload["sections"].append(
        {
            "name": "main",
            "kind": "bore",
            "headers_found": True,
            "headers": list(bore_headers),
            "rows": data,
        }
    )


def parse_workbook_sheet(path: Path, sheet_name: str) -> dict[str, Any]:
    spec = classify_sheet(sheet_name)
    rows = load_sheet_rows(path, sheet_name)
    meta = _parse_metadata(rows)

    payload: dict[str, Any] = {
        "source_filename": path.name,
        "source_sheet": sheet_name,
        "spec_name": spec.name if spec else None,
        "kind": spec.kind if spec else "unknown",
        "instrument_family": spec.instrument_family if spec else "unknown",
        "instrument_name": (
            meta.get("instrument_name_cell")
            or (spec.instrument_name if spec else None)
            or sheet_name
        ),
        "pipemaker": meta.get("pipemaker"),
        "bore_type": meta.get("bore_type"),
        "musical_key": meta.get("musical_key"),
        "description": meta.get("description"),
        "method_block": meta.get("method_block"),
        "measurement_method": meta.get("measurement_method"),
        "collector": meta.get("collector"),
        "measurement_date": meta.get("measurement_date"),
        "header_notes": meta.get("header_notes"),
        "units": meta.get("units"),
        "oals": meta.get("oals") or {},
        "sections": [],
        "parse_warnings": [],
    }

    if spec is None:
        payload["parse_warnings"].append(
            f"Sheet '{sheet_name}' is not in the canonical template map."
        )
        return payload

    if spec.kind == "chanter_regulator_bore" and spec.name == "Bass Upper Bore":
        _parse_bass_upper_bore(rows, payload, spec.expected_bore_headers)
    elif spec.kind == "chanter_regulator_bore":
        headers = spec.expected_bore_headers
        header_idx = _find_header_row(rows, headers)
        if header_idx is None:
            payload["parse_warnings"].append(
                f"Could not locate bore headers {headers}."
            )
        else:
            data = _read_table(rows, header_idx, headers)
            payload["sections"].append(
                {
                    "name": "main",
                    "kind": "bore",
                    "headers_found": True,
                    "headers": list(headers),
                    "rows": data,
                }
            )

    elif spec.kind == "toneholes":
        headers = spec.expected_tonehole_headers
        header_idx = _find_header_row(rows, headers)
        if header_idx is None:
            payload["parse_warnings"].append(
                f"Could not locate tonehole headers {headers}."
            )
        else:
            data = _read_table(rows, header_idx, headers)
            payload["sections"].append(
                {
                    "name": "toneholes",
                    "kind": "toneholes",
                    "headers_found": True,
                    "headers": list(headers),
                    "rows": data,
                }
            )

    elif spec.kind == "drone_bore":
        sections = _find_section_blocks(
            rows,
            ["Upper Bore", "Lower Bore"],
            spec.expected_bore_headers,
        )
        payload["sections"].extend(sections)

    elif spec.kind == "bass_drone_bore":
        bore_sections = _find_section_blocks(
            rows,
            ["J1", "J2", "J3", "Slide"],
            spec.expected_bore_headers,
        )
        payload["sections"].extend(bore_sections)

        u_blocks = _find_section_blocks(rows, ["U1", "U2", "U3"], U_BEND_HEADERS)
        for block in u_blocks:
            block["kind"] = "u_bend"
            # remap keys already match headers
        payload["sections"].extend(u_blocks)

        # Slide Brass Tube
        sbt_idx = None
        for idx, row in enumerate(rows):
            label = _cell_str(row[0]) if row else None
            if label and label.lower() == "slide brass tube":
                sbt_idx = idx
                break
        if sbt_idx is not None:
            header_idx = None
            for j in range(sbt_idx + 1, min(sbt_idx + 5, len(rows))):
                vals = [normalize_header(c) for c in rows[j][:4]]
                if [v.lower() for v in vals] == [h.lower() for h in SLIDE_BRASS_HEADERS]:
                    header_idx = j
                    break
            if header_idx is not None:
                data = _read_table(rows, header_idx, SLIDE_BRASS_HEADERS)
                payload["sections"].append(
                    {
                        "name": "Slide Brass Tube",
                        "kind": "slide_brass_tube",
                        "headers_found": True,
                        "headers": list(SLIDE_BRASS_HEADERS),
                        "rows": data,
                    }
                )

    return payload


def intake_node(state: IntakeState) -> dict[str, Any]:
    source = state.get("source_path")
    if not source:
        return {
            "status": "failed",
            "needs_human_review": True,
            "issues": [
                {
                    "stage": "intake",
                    "code": "missing_source_path",
                    "message": (
                        "State is missing source_path. This usually means the "
                        "LangGraph thread checkpoint was not found (for example "
                        "an old in-memory run). Start a new intake with "
                        "--source and --thread-id, then resume that same thread."
                    ),
                }
            ],
            "messages": [
                "Intake failed: missing source_path in graph state "
                "(checkpoint not found or empty resume)."
            ],
        }

    path = Path(source)
    if not path.exists():
        return {
            "status": "failed",
            "needs_human_review": True,
            "issues": [
                {
                    "stage": "intake",
                    "code": "missing_file",
                    "message": f"Source file not found: {path}",
                }
            ],
            "messages": [f"Intake failed: missing file {path}"],
        }

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return {
            "status": "needs_human_review",
            "needs_human_review": True,
            "issues": [
                {
                    "stage": "intake",
                    "code": "pdf_not_implemented",
                    "message": (
                        "PDF intake is not yet implemented. "
                        "Provide an Excel sheet matching UP_measurement_template.xlsx "
                        "or complete human transcription."
                    ),
                }
            ],
            "messages": ["PDF intake requires human transcription in this phase."],
        }

    if suffix not in SUPPORTED_SUFFIXES:
        return {
            "status": "needs_human_review",
            "needs_human_review": True,
            "issues": [
                {
                    "stage": "intake",
                    "code": "unsupported_format",
                    "message": (
                        f"Unsupported file type: {suffix}. "
                        f"Supported: {sorted(SUPPORTED_SUFFIXES)}"
                    ),
                }
            ],
            "messages": [f"Unsupported intake format: {suffix}"],
        }

    sheet_name = state.get("sheet_name")
    sheets = list_workbook_sheets(path)

    if sheet_name is None:
        # Prefer first known template sheet; otherwise first sheet
        sheet_name = next(
            (s for s in sheets if classify_sheet(s) is not None),
            sheets[0] if sheets else None,
        )
    if sheet_name is None:
        return {
            "status": "failed",
            "needs_human_review": True,
            "issues": [
                {
                    "stage": "intake",
                    "code": "empty_workbook",
                    "message": "Workbook has no sheets.",
                }
            ],
            "messages": ["Intake failed: empty workbook"],
        }

    parsed = parse_workbook_sheet(path, sheet_name)
    warnings = parsed.get("parse_warnings") or []
    issues = [
        {
            "stage": "intake",
            "code": "parse_warning",
            "message": w,
            "sheet": sheet_name,
        }
        for w in warnings
    ]
    return {
        "parsed": parsed,
        "sheet_name": sheet_name,
        "status": "parsed",
        "needs_human_review": bool(issues),
        "issues": issues,
        "messages": [
            f"Parsed '{sheet_name}' from {path.name}: "
            f"{sum(len(s.get('rows') or []) for s in parsed.get('sections') or [])} data rows"
        ],
    }
