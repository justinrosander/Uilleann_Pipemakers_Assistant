"""Read .ods spreadsheets into sheet name + row matrices via odfpy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from odf.opendocument import load
from odf.table import CoveredTableCell, Table, TableCell, TableRow
from odf.text import P


def _local_name(node) -> str:
    tag = getattr(node, "tagName", "") or ""
    return tag.split(":")[-1]


def _repeat_count(element, attr: str) -> int:
    raw = element.getAttribute(attr)
    if raw is None or raw == "":
        return 1
    try:
        count = int(raw)
    except ValueError:
        return 1
    # ODS exporters sometimes pad with huge repeats; cap for safety
    return max(min(count, 512), 1)


def _cell_value(cell: TableCell) -> Any:
    value_type = (cell.getAttribute("valuetype") or "").lower()
    if value_type in {"float", "currency", "percentage"}:
        raw = cell.getAttribute("value")
        if raw is not None and raw != "":
            try:
                return float(raw)
            except ValueError:
                pass
    if value_type == "boolean":
        raw = (cell.getAttribute("booleanvalue") or "").lower()
        if raw in {"true", "false"}:
            return raw == "true"
    texts = [str(p) for p in cell.getElementsByType(P)]
    joined = "\n".join(t for t in texts if t is not None).strip()
    return joined or None


def list_ods_sheets(path: Path | str) -> list[str]:
    doc = load(str(path))
    return [
        table.getAttribute("name") or f"Sheet{i + 1}"
        for i, table in enumerate(doc.spreadsheet.getElementsByType(Table))
    ]


def read_ods_sheet_rows(path: Path | str, sheet_name: str) -> list[tuple[Any, ...]]:
    doc = load(str(path))
    tables = {
        (table.getAttribute("name") or f"Sheet{i + 1}"): table
        for i, table in enumerate(doc.spreadsheet.getElementsByType(Table))
    }
    if sheet_name not in tables:
        raise FileNotFoundError(
            f"Sheet '{sheet_name}' not found in {Path(path).name}. "
            f"Available: {sorted(tables)}"
        )

    matrix: list[tuple[Any, ...]] = []
    for row in tables[sheet_name].getElementsByType(TableRow):
        values: list[Any] = []
        for child in row.childNodes:
            name = _local_name(child)
            if name == "table-cell":
                repeats = _repeat_count(child, "numbercolumnsrepeated")
                value = _cell_value(child)
                values.extend([value] * repeats)
            elif name == "covered-table-cell":
                repeats = _repeat_count(child, "numbercolumnsrepeated")
                values.extend([None] * repeats)

        while values and values[-1] is None:
            values.pop()

        row_tuple = tuple(values)
        for _ in range(_repeat_count(row, "numberrowsrepeated")):
            matrix.append(row_tuple)

    return matrix
