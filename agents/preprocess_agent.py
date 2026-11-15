"""Interactive preprocessor for non-standard measurement files.

Raw sources (.csv, .txt, .xlsx, .xlsm, .ods, .pdf) are copied to data/raw.
The user reviews each structured table and any unstructured text, then decides
where it goes in a copy of UP_measurement_template.xlsx, whether to leave it
out, or whether to finish by hand. The copy is saved under data/processed.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Border, Side

from agents.mapping_memory import (
    format_examples_for_prompt,
    import_mapping_logs,
    memory_column_map,
    record_mappings,
    retrieve_examples,
)
from config import PROCESSED_DIR, RAW_DIR, TEMPLATE_PATH, XAI_API_KEY, XAI_BASE_URL, XAI_MODEL
from schema.template_schema import (
    BASS_UPPER_U_BEND_HEADERS,
    CHANTER_REG_BORE_HEADERS,
    DRONE_BORE_HEADERS,
    SHEET_SPECS,
    SLIDE_BRASS_HEADERS,
    U_BEND_HEADERS,
    SheetSpec,
)

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]

HEADER_SETS: tuple[tuple[str, ...], ...] = (
    CHANTER_REG_BORE_HEADERS,
    DRONE_BORE_HEADERS,
    U_BEND_HEADERS,
    BASS_UPPER_U_BEND_HEADERS,
    SLIDE_BRASS_HEADERS,
    SHEET_SPECS["Chanter Toneholes"].expected_tonehole_headers,
)

SECTION_NAMES = (
    "Reed Seat",
    "U-bend",
    "Upper Bore",
    "Lower Bore",
    "Slide Brass Tube",
    "J1",
    "J2",
    "J3",
    "Slide",
    "U1",
    "U2",
    "U3",
)


@dataclass
class SourceBlock:
    name: str
    kind: str  # structured or unstructured
    frame: pd.DataFrame | None = None
    text: str = ""
    detail: str = ""


@dataclass
class PlacedTable:
    sheet_name: str
    section_name: str | None
    column_map: dict[str, str]
    frame: pd.DataFrame


@dataclass
class SessionChoices:
    pipemaker: str = ""
    last_name: str = ""
    bore_type: str = ""
    musical_key: str = ""
    voice: str = ""
    instrument_name: str = ""
    units: str = "imperial"
    method: str = ""
    collector: str = ""
    measurement_date: str = ""
    header_notes: str = ""
    oal: str = ""
    tables: list[PlacedTable] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    hand_edit: bool = False


def _ask(prompt: str, input_fn: InputFn, output_fn: OutputFn, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    output_fn(prompt + suffix)
    answer = input_fn("> ").strip()
    return answer or default


def _filename_part(value: str) -> str:
    cleaned = value.replace("'", "").replace("’", "")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", cleaned).strip("_")
    return cleaned or "Unknown"


def _last_name(full_name: str) -> str:
    parts = [part for part in re.split(r"\s+", full_name.strip()) if part]
    if not parts:
        return ""
    return parts[-1]


def store_raw_copy(source: Path) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / source.name
    if dest.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = RAW_DIR / f"{source.stem}_{stamp}{source.suffix}"
    shutil.copy2(source, dest)
    return dest


def _sniff_delimiter(path: Path) -> str | None:
    sample = path.read_text(encoding="utf-8", errors="replace")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        return None
    return dialect.delimiter


def _detect_header_index(frame: pd.DataFrame) -> int | None:
    """Return a row index that looks like column names above numeric data."""
    limit = min(len(frame), 20)
    for index in range(limit):
        row = frame.iloc[index]
        labels = [str(value).strip() for value in row if pd.notna(value) and str(value).strip()]
        if len(labels) < 2:
            continue
        if not all(len(label) < 40 for label in labels):
            continue
        following = frame.iloc[index + 1 : index + 6]
        if following.empty:
            continue
        numeric_cells = 0
        total_cells = 0
        for value in following.to_numpy().ravel():
            if pd.isna(value) or str(value).strip() == "":
                continue
            total_cells += 1
            try:
                float(str(value).strip())
                numeric_cells += 1
            except ValueError:
                pass
        if total_cells and numeric_cells / total_cells >= 0.4:
            return index
    return None


def _frame_from_header(raw: pd.DataFrame, header_index: int) -> pd.DataFrame:
    headers = [
        str(value).strip() if pd.notna(value) and str(value).strip() else f"column_{i + 1}"
        for i, value in enumerate(raw.iloc[header_index])
    ]
    data = raw.iloc[header_index + 1 :].copy()
    data.columns = headers
    data = data.dropna(how="all")
    data = data.loc[:, ~data.columns.duplicated()]
    return data.reset_index(drop=True)


def _read_delimited(path: Path) -> list[SourceBlock]:
    delimiter = _sniff_delimiter(path)
    if delimiter is None:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        return [
            SourceBlock(
                name=path.name,
                kind="unstructured",
                text=text[:4000],
                detail="No consistent delimiter was found.",
            )
        ]
    raw = pd.read_csv(path, sep=delimiter, header=None, dtype=str, engine="python")
    header_index = _detect_header_index(raw)
    if header_index is None:
        return [
            SourceBlock(
                name=path.name,
                kind="unstructured",
                text=raw.head(20).to_csv(index=False),
                detail="Rows were read, but no header-plus-values table was found.",
            )
        ]
    frame = _frame_from_header(raw, header_index)
    return [
        SourceBlock(
            name=path.name,
            kind="structured",
            frame=frame,
            detail=f"Delimiter {delimiter!r}, header on row {header_index + 1}.",
        )
    ]


def _read_spreadsheet(path: Path) -> list[SourceBlock]:
    suffix = path.suffix.lower()
    # Legacy .xls needs the calamine engine (or xlrd). Prefer calamine.
    kwargs: dict[str, Any] = {"sheet_name": None, "header": None, "dtype": str}
    if suffix == ".xls":
        try:
            import python_calamine  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Reading .xls files needs python-calamine in the same Python "
                "that runs this script. Install with: py -m pip install python-calamine"
            ) from exc
        kwargs["engine"] = "calamine"
    try:
        sheets = pd.read_excel(path, **kwargs)
    except ImportError as exc:
        raise ImportError(
            "Could not load the spreadsheet engine for this file. "
            "For .xls use: py -m pip install python-calamine. "
            f"Original error: {exc}"
        ) from exc
    blocks: list[SourceBlock] = []
    for sheet_name, raw in sheets.items():
        if raw.dropna(how="all").empty:
            continue
        header_index = _detect_header_index(raw)
        if header_index is None:
            preview = raw.dropna(how="all").head(15).fillna("").to_csv(index=False)
            blocks.append(
                SourceBlock(
                    name=str(sheet_name),
                    kind="unstructured",
                    text=preview[:4000],
                    detail="Sheet has content, but it does not look like a measurement table.",
                )
            )
            continue
        blocks.append(
            SourceBlock(
                name=str(sheet_name),
                kind="structured",
                frame=_frame_from_header(raw, header_index),
                detail=f"Header on row {header_index + 1}.",
            )
        )
    return blocks


def _read_pdf(path: Path) -> list[SourceBlock]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n".join(pages).strip()
    if not text:
        return [
            SourceBlock(
                name=path.name,
                kind="unstructured",
                text="",
                detail="No text could be extracted from this PDF.",
            )
        ]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    delimited = [line for line in lines if line.count(",") >= 2 or line.count("\t") >= 2]
    blocks = [
        SourceBlock(
            name=path.name,
            kind="unstructured",
            text=text[:4000],
            detail=f"Extracted text from {len(reader.pages)} page(s).",
        )
    ]
    if len(delimited) >= 3:
        raw = pd.DataFrame([line.split("\t" if "\t" in line else ",") for line in delimited])
        header_index = _detect_header_index(raw)
        if header_index is not None:
            blocks.insert(
                0,
                SourceBlock(
                    name=f"{path.name} table",
                    kind="structured",
                    frame=_frame_from_header(raw, header_index),
                    detail="Built from delimited lines found in the PDF text.",
                ),
            )
    return blocks


def load_source_blocks(path: Path) -> list[SourceBlock]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        return _read_delimited(path)
    if suffix in {".xlsx", ".xlsm", ".xls", ".ods"}:
        return _read_spreadsheet(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    raise ValueError(
        f"Unsupported preprocess type {suffix}. "
        "Use .csv, .txt, .tsv, .xlsx, .xlsm, .ods, or .pdf."
    )


def _template_columns_for(spec: SheetSpec, section_name: str | None) -> tuple[str, ...]:
    if section_name in {"U1", "U2", "U3"}:
        return U_BEND_HEADERS
    if section_name == "U-bend" and spec.name == "Bass Upper Bore":
        return BASS_UPPER_U_BEND_HEADERS
    if section_name == "Slide Brass Tube":
        return SLIDE_BRASS_HEADERS
    if spec.kind == "toneholes":
        return spec.expected_tonehole_headers
    if spec.kind == "chanter_regulator_bore":
        return spec.expected_bore_headers
    return spec.expected_bore_headers


def _sections_for(spec: SheetSpec) -> list[str]:
    if spec.name == "Bass Upper Bore":
        return ["Upper Bore", "Reed Seat", "U-bend"]
    if spec.kind == "drone_bore":
        return ["Upper Bore", "Lower Bore"]
    if spec.kind == "bass_drone_bore":
        return ["J1", "J2", "J3", "Slide", "U1", "U2", "U3", "Slide Brass Tube"]
    return ["main"]


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def suggest_column_map(
    source_columns: list[str],
    targets: tuple[str, ...],
    *,
    sheet_name: str | None = None,
) -> dict[str, str]:
    """Match source names to template columns. Prior accepted mappings win ties."""
    suggestions: dict[str, str] = {}
    used: set[str] = set()
    for source, target in memory_column_map(
        source_columns, targets, sheet_name=sheet_name
    ).items():
        if target in targets and target not in used:
            suggestions[source] = target
            used.add(target)
    aliases = {
        "diameter": "Diameter",
        "diam": "Diameter",
        "bore": "Diameter",
        "nominal": "Nominal",
        "round": "Round",
        "oval": "Oval",
        "notes": "Notes",
        "note": "Notes",
        "length": "Length",
        "position": "Round",
        "station": "Nominal",
        "keyhole": "keyhole",
        "chimney": "chimney",
    }
    target_by_norm = {_normalize(target): target for target in targets}
    for source in source_columns:
        if source in suggestions:
            continue
        norm = _normalize(source)
        chosen = None
        if norm in aliases and aliases[norm] in targets and aliases[norm] not in used:
            chosen = aliases[norm]
        elif norm in target_by_norm and target_by_norm[norm] not in used:
            chosen = target_by_norm[norm]
        else:
            for target in targets:
                if target in used:
                    continue
                if _normalize(target) and _normalize(target) in norm:
                    chosen = target
                    break
        if chosen:
            suggestions[source] = chosen
            used.add(chosen)
    return suggestions


def suggest_with_llm(
    source_columns: list[str],
    sample_rows: list[dict[str, str]],
    targets: tuple[str, ...],
    *,
    sheet_name: str | None = None,
    prior_examples: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Ask Grok for a mapping, using prior accepted mappings as context."""
    if not XAI_API_KEY:
        return {}
    from openai import OpenAI

    examples = prior_examples
    if examples is None:
        examples = retrieve_examples(
            source_columns, sheet_name=sheet_name, targets=targets
        )
    client = OpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL, timeout=60.0)
    prompt = (
        "Map measurement columns onto this template. "
        "Reply with JSON only, an object whose keys are source column names "
        "and whose values are template column names or null when there is no match. "
        "Prefer the previous accepted mappings when the source column or sample values are similar. "
        "Do not invent measurements. "
        f"Previous accepted mappings:\n{format_examples_for_prompt(examples)}\n"
        f"Template columns: {list(targets)}. "
        f"Target sheet: {sheet_name or 'unknown'}. "
        f"Source columns: {source_columns}. "
        f"Sample rows: {sample_rows[:5]}."
    )
    completion = client.chat.completions.create(
        model=XAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You match spreadsheet columns. Do not invent measurements.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = completion.choices[0].message.content or "{}"
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        return {}
    parsed = json.loads(content[start : end + 1])
    allowed = set(targets)
    mapping: dict[str, str] = {}
    for key, value in parsed.items():
        if value in allowed and key in source_columns:
            mapping[str(key)] = str(value)
    return mapping


def _preview_frame(frame: pd.DataFrame, limit: int = 5) -> str:
    return frame.head(limit).to_string(index=False)


def _choose_sheet(input_fn: InputFn, output_fn: OutputFn) -> SheetSpec:
    names = list(SHEET_SPECS)
    output_fn("Template sheets:")
    for name in names:
        output_fn(f"  - {name}")
    while True:
        choice = _ask("Which template sheet should receive this table?", input_fn, output_fn)
        spec = SHEET_SPECS.get(choice)
        if spec:
            return spec
        output_fn("That sheet is not in the template. Type the name exactly.")


def _choose_section(spec: SheetSpec, input_fn: InputFn, output_fn: OutputFn) -> str | None:
    sections = _sections_for(spec)
    if sections == ["main"]:
        return None
    output_fn("Sections: " + ", ".join(sections))
    while True:
        choice = _ask("Which section?", input_fn, output_fn, default=sections[0])
        if choice in sections:
            return choice
        output_fn("Choose one of the section names above.")


def _map_columns(
    block: SourceBlock,
    targets: tuple[str, ...],
    input_fn: InputFn,
    output_fn: OutputFn,
    *,
    sheet_name: str | None = None,
) -> dict[str, str] | None:
    assert block.frame is not None
    columns = [str(column) for column in block.frame.columns]
    prior = retrieve_examples(columns, sheet_name=sheet_name, targets=targets)
    if prior:
        output_fn(f"Using {len(prior)} prior accepted mapping(s) as context.")
    heuristic = suggest_column_map(columns, targets, sheet_name=sheet_name)
    llm_map: dict[str, str] = {}
    if XAI_API_KEY:
        try:
            sample = block.frame.head(3).fillna("").astype(str).to_dict(orient="records")
            llm_map = suggest_with_llm(
                columns,
                sample,
                targets,
                sheet_name=sheet_name,
                prior_examples=prior,
            )
        except Exception as exc:
            output_fn(f"Column suggestion from the model was not used ({exc}).")
    suggested = {**heuristic, **llm_map}
    output_fn("Suggested mapping (source -> template):")
    if suggested:
        for source, target in suggested.items():
            output_fn(f"  {source} -> {target}")
    else:
        output_fn("  (no confident match)")
    output_fn("Unmapped source columns: " + ", ".join(
        column for column in columns if column not in suggested
    ))
    action = _ask(
        "Action: [a]ccept suggestions, [m]ap each column, [s]kip this table, [h]and-edit",
        input_fn,
        output_fn,
        default="m",
    ).lower()
    if action.startswith("s"):
        return None
    if action.startswith("h"):
        return {"__hand_edit__": "1"}
    if action.startswith("a"):
        return suggested
    mapping: dict[str, str] = {}
    claimed_targets: set[str] = set()
    output_fn("Template columns: " + ", ".join(targets))
    output_fn(
        "If you map a second source column onto a target that is already mapped, "
        "those values are appended in empty rows below existing data."
    )
    for column in columns:
        target = _ask(
            f"Where should '{column}' go? Enter a template column, or skip",
            input_fn,
            output_fn,
            default=suggested.get(column, "skip"),
        )
        if target.lower() == "skip":
            continue
        if target not in targets:
            output_fn(f"'{target}' is not a template column. Skipping '{column}'.")
            continue
        if target in claimed_targets:
            output_fn(
                f"'{target}' is already mapped. Values from '{column}' will be "
                "appended below existing rows in that column."
            )
        mapping[column] = target
        claimed_targets.add(target)
    return mapping


def _review_blocks(
    blocks: list[SourceBlock],
    input_fn: InputFn,
    output_fn: OutputFn,
    choices: SessionChoices,
) -> None:
    for number, block in enumerate(blocks, start=1):
        output_fn("")
        output_fn(f"Block {number}: {block.kind} — {block.name}")
        output_fn(block.detail)
        if block.kind == "structured" and block.frame is not None:
            output_fn(_preview_frame(block.frame))
            output_fn(
                f"{len(block.frame)} data rows, columns: "
                + ", ".join(str(column) for column in block.frame.columns)
            )
            action = _ask(
                "Review this table? [y]es, [s]kip, [h]and-edit the workbook and stop",
                input_fn,
                output_fn,
                default="y",
            ).lower()
            if action.startswith("h"):
                choices.hand_edit = True
                return
            if action.startswith("s"):
                choices.skipped.append(block.name)
                continue
            spec = _choose_sheet(input_fn, output_fn)
            section = _choose_section(spec, input_fn, output_fn)
            targets = _template_columns_for(spec, section)
            mapping = _map_columns(
                block, targets, input_fn, output_fn, sheet_name=spec.name
            )
            if mapping is None:
                choices.skipped.append(block.name)
                continue
            if mapping.get("__hand_edit__"):
                choices.hand_edit = True
                return
            record_mappings(
                frame=block.frame,
                column_map=mapping,
                sheet_name=spec.name,
                section_name=section,
                source_block=block.name,
                pipemaker=choices.pipemaker,
            )
            output_fn(
                f"Saved {len(mapping)} accepted mapping(s) for later preprocess runs."
            )
            choices.tables.append(
                PlacedTable(
                    sheet_name=spec.name,
                    section_name=section,
                    column_map=mapping,
                    frame=block.frame,
                )
            )
            continue

        preview = block.text[:800] if block.text else "(empty)"
        output_fn(preview)
        action = _ask(
            "Put this text in header [n]otes, [m]ethod, [c]ollector, [d]ate, "
            "[s]kip, or [h]and-edit",
            input_fn,
            output_fn,
            default="s",
        ).lower()
        if action.startswith("h"):
            choices.hand_edit = True
            return
        if action.startswith("n"):
            choices.header_notes = (choices.header_notes + "\n" + block.text).strip()
        elif action.startswith("m"):
            choices.method = block.text.strip()
        elif action.startswith("c"):
            choices.collector = block.text.strip()[:200]
        elif action.startswith("d"):
            choices.measurement_date = block.text.strip()[:40]
        else:
            choices.skipped.append(block.name)


def _template_sheet_title(sheet_name: str) -> str:
    """A1 on each template sheet is the instrument title for that sheet."""
    wb = load_workbook(TEMPLATE_PATH, read_only=True, data_only=True)
    try:
        if sheet_name not in wb.sheetnames:
            return sheet_name
        value = wb[sheet_name]["A1"].value
        return str(value).strip() if value is not None else sheet_name
    finally:
        wb.close()


def _collect_identity(
    input_fn: InputFn,
    output_fn: OutputFn,
) -> SessionChoices:
    output_fn("Shared identity for the processed workbook.")
    output_fn(
        "Instrument title and optional voice stay on each sheet as shown in "
        "UP_measurement_template.xlsx cell A1 (for example Chanter, "
        "Tenor Regulator, Bass Drone). They are not entered once for every sheet."
    )
    choices = SessionChoices()
    choices.pipemaker = _ask("Pipemaker full name", input_fn, output_fn)
    guessed_last = _last_name(choices.pipemaker)
    choices.last_name = _ask("Last name for the file name", input_fn, output_fn, guessed_last)
    choices.bore_type = _ask("Bore type (Wide, Narrow, or leave blank)", input_fn, output_fn)
    choices.musical_key = _ask("Key", input_fn, output_fn, "D")
    choices.units = _ask("Units", input_fn, output_fn, "imperial").lower()
    choices.method = _ask("Measurement method, if you want it in the header", input_fn, output_fn)
    choices.collector = _ask("Collector, if indicated", input_fn, output_fn)
    choices.measurement_date = _ask("Date, if indicated", input_fn, output_fn)
    choices.header_notes = _ask("Header notes, if indicated", input_fn, output_fn)
    choices.oal = _ask(
        "Overall length for the next mapped section, if known (leave blank to keep sheet OAL)",
        input_fn,
        output_fn,
    )
    return choices


def _row_values(ws, row: int, width: int) -> list[Any]:
    return [ws.cell(row, col).value for col in range(1, width + 1)]


def _header_matches(values: list[Any], headers: tuple[str, ...]) -> bool:
    found = [str(value).strip() if value is not None else "" for value in values[: len(headers)]]
    return [item.lower() for item in found] == [item.lower() for item in headers]


def _find_header_row(ws, headers: tuple[str, ...], start_row: int = 1) -> int | None:
    width = len(headers)
    for row in range(start_row, ws.max_row + 1):
        if _header_matches(_row_values(ws, row, width), headers):
            return row
    return None


def _section_row(ws, section_name: str | None) -> int | None:
    if not section_name:
        return None
    target = section_name.lower()
    for row in range(1, ws.max_row + 1):
        value = ws.cell(row, 1).value
        if value is None:
            continue
        text = str(value).strip()
        if text.lower() == target or text.lower() == f"{target} (optional)":
            return row
    return None


def _is_stop_row(ws, row: int, width: int, stop_labels: set[str]) -> bool:
    label = ws.cell(row, 1).value
    if isinstance(label, str) and label.strip().lower() in stop_labels:
        return True
    values = _row_values(ws, row, max(width, 5))
    for headers in HEADER_SETS:
        if _header_matches(values, headers):
            return True
    return False


def _clear_below_header(
    ws, header_row: int, width: int, stop_labels: set[str]
) -> tuple[int, int | None]:
    thin = Border(
        left=Side(style="thin", color="B0B0B0"),
        right=Side(style="thin", color="B0B0B0"),
        top=Side(style="thin", color="B0B0B0"),
        bottom=Side(style="thin", color="B0B0B0"),
    )
    first_empty = None
    stop_row = None
    for row in range(header_row + 1, ws.max_row + 1):
        if _is_stop_row(ws, row, width, stop_labels):
            stop_row = row
            break
        for col in range(1, width + 1):
            cell = ws.cell(row, col)
            cell.value = None
            cell.border = thin
        if first_empty is None:
            first_empty = row
    return first_empty or (header_row + 1), stop_row


def _write_identity(ws, choices: SessionChoices, oal_label: str | None) -> None:
    """Write shared maker metadata. Leave A1 and voice/instrument as the template sheet defines them."""
    # A1 is the sheet instrument title from UP_measurement_template.xlsx.
    # Voice and instrument columns under the header row stay as in the template.
    for row in range(1, 15):
        label = ws.cell(row, 1).value
        if isinstance(label, str) and label.strip().lower() == "pipemaker":
            values = {
                "pipemaker": choices.pipemaker or None,
                "bore type": choices.bore_type or None,
                "key": choices.musical_key or None,
            }
            for col in range(1, 8):
                heading = ws.cell(row, col).value
                if not isinstance(heading, str):
                    continue
                key = heading.strip().lower()
                if key in values:
                    ws.cell(row + 1, col).value = values[key]
        if isinstance(label, str) and label.lower().startswith("method:"):
            ws.cell(row, 1).value = (
                f"Method: {choices.method or '[not indicated]'}; "
                f"Collector: {choices.collector or '[not indicated]'}; "
                f"Date: {choices.measurement_date or '[not indicated]'}; "
                f"Notes: {choices.header_notes or '[not indicated]'}"
            )
        if isinstance(label, str) and label.lower().startswith("units:"):
            ws.cell(row, 1).value = f"Units: {choices.units or 'imperial'}"
        if oal_label and isinstance(label, str) and label.strip().lower() == oal_label.lower():
            if choices.oal:
                try:
                    ws.cell(row, 2).value = float(choices.oal)
                except ValueError:
                    ws.cell(row, 2).value = choices.oal


def _oal_label_for(section_name: str | None, spec: SheetSpec) -> str | None:
    if section_name in {"Upper Bore", "Lower Bore", "Reed Seat", "U-bend"}:
        if section_name == "Upper Bore" and spec.kind == "drone_bore":
            return "Upper Bore OAL"
        if section_name == "Lower Bore":
            return "Lower Bore OAL"
        if section_name == "Reed Seat":
            return "Reed Seat OAL"
        if section_name == "U-bend":
            return "U-bend OAL"
    if spec.oal_labels:
        return spec.oal_labels[0]
    return "OAL"


def _stop_labels() -> set[str]:
    labels = {name.lower() for name in SECTION_NAMES}
    labels.update(f"{name.lower()} (optional)" for name in SECTION_NAMES)
    return labels


def _group_column_map(
    column_map: dict[str, str],
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Split mappings into one primary source per target, plus extras that append."""
    primary: dict[str, str] = {}
    extras: list[tuple[str, str]] = []
    seen_targets: set[str] = set()
    for source_name, target_name in column_map.items():
        if target_name in seen_targets:
            extras.append((source_name, target_name))
        else:
            primary[source_name] = target_name
            seen_targets.add(target_name)
    return primary, extras


def _coerce_cell_value(value: Any) -> Any:
    if pd.isna(value) or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        return text


def _section_write_start(
    ws, header_row: int, width: int, stop_labels: set[str]
) -> tuple[int, int | None]:
    """Find the first empty row after existing section data, and the stop boundary."""
    stop_row = None
    last_used = header_row
    for row in range(header_row + 1, ws.max_row + 1):
        if _is_stop_row(ws, row, width, stop_labels):
            stop_row = row
            break
        if any(ws.cell(row, column).value is not None for column in range(1, width + 1)):
            last_used = row
    return last_used + 1, stop_row


def _last_used_row_in_column(
    ws, column: int, start_row: int, stop_row: int | None
) -> int:
    last = start_row - 1
    end = stop_row if stop_row is not None else ws.max_row + 1
    for row in range(start_row, end):
        if ws.cell(row, column).value is not None:
            last = row
    return last


def _ensure_row(ws, excel_row: int, stop_row: int | None) -> int | None:
    """Insert a row if writing would collide with the next section; return updated stop_row."""
    if stop_row is not None and excel_row >= stop_row:
        ws.insert_rows(excel_row)
        return stop_row + 1
    return stop_row


def _write_table(ws, placed: PlacedTable, choices: SessionChoices) -> int:
    spec = SHEET_SPECS[placed.sheet_name]
    headers = _template_columns_for(spec, placed.section_name)
    start_search = 1
    section_row = _section_row(ws, placed.section_name)
    if section_row:
        start_search = section_row
    header_row = _find_header_row(ws, headers, start_search)
    if header_row is None:
        raise ValueError(
            f"Could not find columns {headers} on sheet '{placed.sheet_name}'."
        )
    stop_labels = _stop_labels()
    # Do not clear prior mapped data. Append after whatever is already in the section.
    first_row, stop_row = _section_write_start(
        ws, header_row, len(headers), stop_labels
    )
    header_index = {header: index + 1 for index, header in enumerate(headers)}
    primary_map, extras = _group_column_map(placed.column_map)

    written = 0
    for _, source_row in placed.frame.iterrows():
        excel_row = first_row + written
        stop_row = _ensure_row(ws, excel_row, stop_row)
        for source_name, target_name in primary_map.items():
            if target_name not in header_index:
                continue
            value = _coerce_cell_value(source_row.get(source_name))
            if value is None:
                continue
            ws.cell(excel_row, header_index[target_name]).value = value
        written += 1

    # Additional source columns that share a target are appended below existing values.
    for source_name, target_name in extras:
        if target_name not in header_index:
            continue
        column = header_index[target_name]
        append_at = (
            _last_used_row_in_column(ws, column, header_row + 1, stop_row) + 1
        )
        for value in placed.frame[source_name].tolist():
            coerced = _coerce_cell_value(value)
            if coerced is None:
                continue
            stop_row = _ensure_row(ws, append_at, stop_row)
            ws.cell(append_at, column).value = coerced
            append_at += 1
            written += 1

    _write_identity(ws, choices, _oal_label_for(placed.section_name, spec))
    return written


def _clear_example_rows(workbook) -> None:
    stop_labels = {name.lower() for name in SECTION_NAMES}
    stop_labels.update(f"{name.lower()} (optional)" for name in SECTION_NAMES)
    for sheet_name in workbook.sheetnames:
        if sheet_name not in SHEET_SPECS:
            continue
        ws = workbook[sheet_name]
        spec = SHEET_SPECS[sheet_name]
        header_groups = []
        if spec.kind == "toneholes":
            header_groups.append(spec.expected_tonehole_headers)
        elif spec.kind == "bass_drone_bore":
            header_groups.extend(
                [DRONE_BORE_HEADERS, U_BEND_HEADERS, SLIDE_BRASS_HEADERS]
            )
        elif spec.name == "Bass Upper Bore":
            header_groups.extend([CHANTER_REG_BORE_HEADERS, BASS_UPPER_U_BEND_HEADERS])
        else:
            header_groups.append(spec.expected_bore_headers)
        for headers in header_groups:
            row = 1
            while row is not None and row <= ws.max_row:
                found = _find_header_row(ws, headers, row)
                if found is None:
                    break
                _clear_below_header(ws, found, len(headers), stop_labels)
                row = found + 1


def processed_filename(choices: SessionChoices, used_sheets: set[str] | None = None) -> str:
    last = _filename_part(choices.last_name or _last_name(choices.pipemaker))
    key = _filename_part(choices.musical_key or "Key")
    titles: list[str] = []
    for sheet_name in sorted(used_sheets or []):
        title = _template_sheet_title(sheet_name)
        if title and title not in titles:
            titles.append(title)
    if not titles and choices.instrument_name:
        titles.append(choices.instrument_name)
    instrument = _filename_part("_".join(titles) if titles else "Instrument")
    return f"{last}_{key}_{instrument}.xlsx"


def _remove_unused_sheets(workbook, used_sheets: set[str]) -> list[str]:
    """Drop template sheets that received no mapped data after preprocess."""
    keep = {name for name in used_sheets if name in workbook.sheetnames}
    if not keep:
        # Excel workbooks need at least one sheet.
        keep.add(workbook.sheetnames[0])
    removed: list[str] = []
    for name in list(workbook.sheetnames):
        if name not in keep:
            del workbook[name]
            removed.append(name)
    return removed


def write_processed_workbook(choices: SessionChoices) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    used_sheets: set[str] = {placed.sheet_name for placed in choices.tables}
    if not used_sheets and choices.instrument_name:
        # Identity-only / hand-edit path when the user named a sheet explicitly.
        for spec in SHEET_SPECS.values():
            if (
                spec.name.lower() == choices.instrument_name.lower()
                or spec.instrument_name.lower() == choices.instrument_name.lower()
            ):
                used_sheets.add(spec.name)
                break
    dest = PROCESSED_DIR / processed_filename(choices, used_sheets)
    if dest.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = PROCESSED_DIR / f"{dest.stem}_{stamp}{dest.suffix}"
    shutil.copy2(TEMPLATE_PATH, dest)
    workbook = load_workbook(dest)
    _clear_example_rows(workbook)
    for placed in choices.tables:
        if placed.sheet_name not in workbook.sheetnames:
            raise ValueError(f"Template is missing sheet '{placed.sheet_name}'.")
        _write_table(workbook[placed.sheet_name], placed, choices)
        used_sheets.add(placed.sheet_name)
    if not choices.tables and used_sheets:
        sheet_name = next(iter(used_sheets))
        spec = SHEET_SPECS.get(sheet_name)
        _write_identity(
            workbook[sheet_name],
            choices,
            spec.oal_labels[0] if spec and spec.oal_labels else None,
        )
    removed_sheets = _remove_unused_sheets(workbook, used_sheets)
    sheet_titles = {
        name: _template_sheet_title(name) for name in sorted(used_sheets)
    }
    if sheet_titles:
        choices.instrument_name = " / ".join(sheet_titles.values())
    workbook.save(dest)
    log_path = dest.with_suffix(".mapping.txt")
    lines = [
        f"Processed file: {dest.name}",
        f"Pipemaker: {choices.pipemaker}",
        f"Last name: {choices.last_name}",
        f"Key: {choices.musical_key}",
        f"Sheet titles (from template A1): {choices.instrument_name}",
        f"Hand edit requested: {choices.hand_edit}",
        "Skipped blocks: " + (", ".join(choices.skipped) or "(none)"),
        "Kept sheets: " + (", ".join(sorted(used_sheets)) or "(none)"),
        "Removed unused sheets: " + (", ".join(removed_sheets) or "(none)"),
        "Mapped tables:",
    ]
    for placed in choices.tables:
        section = placed.section_name or "main"
        title = sheet_titles.get(placed.sheet_name, placed.sheet_name)
        lines.append(
            f"  {placed.sheet_name} (A1={title}) / {section}: {placed.column_map}"
        )
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def run_preprocess(
    source: Path,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    imported = import_mapping_logs()
    if imported:
        output_fn(f"Loaded {imported} earlier mapping(s) into preprocess memory.")
    raw_path = store_raw_copy(source)
    output_fn(f"Raw copy saved to {raw_path}")
    blocks = load_source_blocks(source)
    output_fn(f"Found {len(blocks)} block(s) in {source.name}.")
    structured = sum(1 for block in blocks if block.kind == "structured")
    unstructured = len(blocks) - structured
    output_fn(
        f"Structured tables: {structured}. Unstructured or non-standard blocks: {unstructured}."
    )
    choices = _collect_identity(input_fn, output_fn)
    _review_blocks(blocks, input_fn, output_fn, choices)
    if not choices.tables:
        output_fn(
            "No tables were mapped. Choose a template sheet to keep "
            "(instrument title stays as that sheet's A1 from the template)."
        )
        spec = _choose_sheet(input_fn, output_fn)
        choices.instrument_name = spec.name
    dest = write_processed_workbook(choices)
    output_fn(f"Processed workbook: {dest}")
    output_fn(f"Mapping notes: {dest.with_suffix('.mapping.txt')}")
    output_fn(
        "Sheets with no entered data were removed from the processed workbook."
    )
    output_fn(
        "Each kept sheet keeps its own instrument title from template cell A1."
    )
    if choices.hand_edit:
        output_fn(
            "Hand edit is on. Open that workbook, move any remaining values yourself, "
            "then run intake with --source pointed at the processed file."
        )
    else:
        output_fn(
            "Next step, when you are ready to validate and store it: "
            f'py main.py --source "{dest}" --all-sheets --thread-id preprocess-1'
        )
    return dest
