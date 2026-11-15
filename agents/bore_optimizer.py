"""Bore-optimizer agent: area-equivalent design from intercepted probe diameters.

Round probes follow the minor axis and oval probes the major axis. A probe of
one diameter usually seats at two different depths. The other axis at each of
those depths is not measured, so it is intercepted from the opposite profile.
The design quantity is the area-equivalent diameter, not the insertion depth.

Results are written to a workbook and an HTML figure. Nothing is written to
PostgreSQL.
"""

from __future__ import annotations

import base64
import html
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

from agents.eda_execute import _numeric, _overlay_label, _report_sort_key, load_bore_tables
from config import ROOT

OUTPUT_DIR = ROOT / "data" / "bore_optimizer"

_STYLE = {
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
}

_CALC_COLUMNS = (
    "Probe diameter (in)",
    "Round insertion depth (in)",
    "Oval insertion depth (in)",
    "Distance from top at round probe (in)",
    "Distance from top at oval probe (in)",
    "Minor diameter at round station (in)",
    "Intercepted major diameter at round station (in)",
    "D_eq at round station (in)",
    "Intercepted minor diameter at oval station (in)",
    "Major diameter at oval station (in)",
    "D_eq at oval station (in)",
    "Note",
)

_DESIGN_COLUMNS = (
    "Instrument",
    "Design diameter D_eq (in)",
    "Distance from top (in)",
    "Insertion depth (in)",
    "D_min (in)",
    "D_maj (in)",
    "Station",
    "Source probe diameter (in)",
)


def _axis_curve(depth: np.ndarray, diameter: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Diameter against insertion depth, averaged where a depth repeats."""
    mask = np.isfinite(depth) & np.isfinite(diameter)
    if int(mask.sum()) < 2:
        return None
    order = np.argsort(depth[mask])
    depth = depth[mask][order]
    diameter = diameter[mask][order]
    rounded = np.round(depth, 6)
    unique, inverse = np.unique(rounded, return_inverse=True)
    averaged = np.array([diameter[inverse == index].mean() for index in range(len(unique))])
    if len(unique) < 2:
        return None
    return unique, averaged


def _intercept(curve: tuple[np.ndarray, np.ndarray] | None, depth: float) -> float:
    """Diameter of one axis at a depth measured on the other. No extrapolation."""
    if curve is None or not np.isfinite(depth):
        return np.nan
    stations, diameters = curve
    if depth < float(stations.min()) or depth > float(stations.max()):
        return np.nan
    return float(np.interp(depth, stations, diameters))


def _distance_from_top(oal: float | None, depth: float) -> float:
    if oal is None or not np.isfinite(oal) or not np.isfinite(depth):
        return np.nan
    return float(oal) - float(depth)


def intercept_area_equivalent(frame: pd.DataFrame, oal: float | None = None) -> pd.DataFrame:
    """Intercept the unknown axis at each round station and each oval station.

    At a round-probe depth the minor diameter is the probe diameter and the
    major diameter is read from the oval-probe curve. At an oval-probe depth
    the major diameter is the probe diameter and the minor diameter is read
    from the round-probe curve. D_eq is the square root of that product.
    """
    diameter = _numeric(frame, "Diameter").to_numpy(dtype=float)
    round_depth = _numeric(frame, "Round").to_numpy(dtype=float)
    oval_depth = _numeric(frame, "Oval").to_numpy(dtype=float)
    minor_curve = _axis_curve(round_depth, diameter)
    major_curve = _axis_curve(oval_depth, diameter)
    rows: list[dict[str, Any]] = []
    for probe, minor_depth, major_depth in zip(diameter, round_depth, oval_depth):
        if not np.isfinite(probe) or not (np.isfinite(minor_depth) or np.isfinite(major_depth)):
            continue
        notes: list[str] = []
        major_at_round = _intercept(major_curve, minor_depth) if np.isfinite(minor_depth) else np.nan
        minor_at_oval = _intercept(minor_curve, major_depth) if np.isfinite(major_depth) else np.nan
        if np.isfinite(minor_depth) and not np.isfinite(major_at_round):
            notes.append("Major diameter at the round-probe depth is outside the oval-probe depths.")
        if np.isfinite(major_depth) and not np.isfinite(minor_at_oval):
            notes.append("Minor diameter at the oval-probe depth is outside the round-probe depths.")
        if not np.isfinite(minor_depth):
            notes.append("No round-probe depth for this diameter.")
        if not np.isfinite(major_depth):
            notes.append("No oval-probe depth for this diameter.")
        eq_round = (
            float(np.sqrt(probe * major_at_round))
            if np.isfinite(minor_depth) and np.isfinite(major_at_round) and major_at_round >= 0
            else np.nan
        )
        eq_oval = (
            float(np.sqrt(minor_at_oval * probe))
            if np.isfinite(major_depth) and np.isfinite(minor_at_oval) and minor_at_oval >= 0
            else np.nan
        )
        rows.append(
            {
                "Probe diameter (in)": float(probe),
                "Round insertion depth (in)": float(minor_depth) if np.isfinite(minor_depth) else np.nan,
                "Oval insertion depth (in)": float(major_depth) if np.isfinite(major_depth) else np.nan,
                "Distance from top at round probe (in)": _distance_from_top(oal, minor_depth),
                "Distance from top at oval probe (in)": _distance_from_top(oal, major_depth),
                "Minor diameter at round station (in)": float(probe) if np.isfinite(minor_depth) else np.nan,
                "Intercepted major diameter at round station (in)": major_at_round,
                "D_eq at round station (in)": eq_round,
                "Intercepted minor diameter at oval station (in)": minor_at_oval,
                "Major diameter at oval station (in)": float(probe) if np.isfinite(major_depth) else np.nan,
                "D_eq at oval station (in)": eq_oval,
                "Note": " ".join(notes),
            }
        )
    calculated = pd.DataFrame(rows, columns=list(_CALC_COLUMNS))
    if calculated.empty:
        return calculated
    return calculated.sort_values("Probe diameter (in)").reset_index(drop=True)


def design_schedule(calculated: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """One design row per intercepted station, ordered by area-equivalent diameter."""
    rows: list[dict[str, Any]] = []
    if calculated.empty:
        return pd.DataFrame(columns=list(_DESIGN_COLUMNS))
    for _, row in calculated.iterrows():
        if np.isfinite(row["D_eq at round station (in)"]):
            rows.append(
                {
                    "Instrument": instrument,
                    "Design diameter D_eq (in)": row["D_eq at round station (in)"],
                    "Distance from top (in)": row["Distance from top at round probe (in)"],
                    "Insertion depth (in)": row["Round insertion depth (in)"],
                    "D_min (in)": row["Minor diameter at round station (in)"],
                    "D_maj (in)": row["Intercepted major diameter at round station (in)"],
                    "Station": "Round-probe depth",
                    "Source probe diameter (in)": row["Probe diameter (in)"],
                }
            )
        if np.isfinite(row["D_eq at oval station (in)"]):
            rows.append(
                {
                    "Instrument": instrument,
                    "Design diameter D_eq (in)": row["D_eq at oval station (in)"],
                    "Distance from top (in)": row["Distance from top at oval probe (in)"],
                    "Insertion depth (in)": row["Oval insertion depth (in)"],
                    "D_min (in)": row["Intercepted minor diameter at oval station (in)"],
                    "D_maj (in)": row["Major diameter at oval station (in)"],
                    "Station": "Oval-probe depth",
                    "Source probe diameter (in)": row["Probe diameter (in)"],
                }
            )
    schedule = pd.DataFrame(rows, columns=list(_DESIGN_COLUMNS))
    if schedule.empty:
        return schedule
    return schedule.sort_values("Design diameter D_eq (in)").reset_index(drop=True)


def _position_series(calculated: pd.DataFrame, round_column: str, oval_column: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Use distance from the top when the sheet has an overall length."""
    round_distance = calculated[round_column].to_numpy(dtype=float)
    if np.isfinite(round_distance).any():
        return (
            round_distance,
            calculated[oval_column].to_numpy(dtype=float),
            "Distance from top (in)",
        )
    return (
        calculated["Round insertion depth (in)"].to_numpy(dtype=float),
        calculated["Oval insertion depth (in)"].to_numpy(dtype=float),
        "Insertion depth (in)",
    )


def plot_area_equivalent_design(calculated: pd.DataFrame, instrument: str) -> bytes | None:
    """Measured axes, intercepted axes, and D_eq. Diameter is the horizontal axis."""
    if calculated.empty:
        return None
    round_position, oval_position, position_label = _position_series(
        calculated,
        "Distance from top at round probe (in)",
        "Distance from top at oval probe (in)",
    )
    probe = calculated["Probe diameter (in)"].to_numpy(dtype=float)
    major_at_round = calculated["Intercepted major diameter at round station (in)"].to_numpy(dtype=float)
    minor_at_oval = calculated["Intercepted minor diameter at oval station (in)"].to_numpy(dtype=float)
    eq_round = calculated["D_eq at round station (in)"].to_numpy(dtype=float)
    eq_oval = calculated["D_eq at oval station (in)"].to_numpy(dtype=float)
    design_diameter = np.concatenate([eq_round, eq_oval])
    design_position = np.concatenate([round_position, oval_position])
    design_keep = np.isfinite(design_diameter) & np.isfinite(design_position)
    buffer = BytesIO()
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        round_keep = np.isfinite(probe) & np.isfinite(round_position)
        oval_keep = np.isfinite(probe) & np.isfinite(oval_position)
        ax.scatter(
            probe[round_keep],
            round_position[round_keep],
            s=22,
            color="#c0392b",
            label="Round probe (minor axis)",
            zorder=3,
        )
        ax.scatter(
            probe[oval_keep],
            oval_position[oval_keep],
            s=22,
            marker="s",
            color="#1e8449",
            label="Oval probe (major axis)",
            zorder=3,
        )
        intercept_major = np.isfinite(major_at_round) & np.isfinite(round_position)
        intercept_minor = np.isfinite(minor_at_oval) & np.isfinite(oval_position)
        ax.scatter(
            major_at_round[intercept_major],
            round_position[intercept_major],
            s=28,
            marker="x",
            color="#196f3d",
            label="Intercepted major diameter",
            zorder=4,
        )
        ax.scatter(
            minor_at_oval[intercept_minor],
            oval_position[intercept_minor],
            s=28,
            marker="+",
            color="#922b21",
            label="Intercepted minor diameter",
            zorder=4,
        )
        if design_keep.any():
            ax.scatter(
                design_diameter[design_keep],
                design_position[design_keep],
                s=18,
                marker="D",
                color="#1a5276",
                label=r"Design $D_{eq}=\sqrt{D_{min}\times D_{maj}}$",
                zorder=5,
            )
        ax.set_xlabel("Bore diameter (in)")
        ax.set_ylabel(position_label)
        ax.set_title(f"{instrument} area-equivalent bore design")
        ax.legend(loc="upper left", frameon=True)
        ax.grid(True, color="#d0d0d0", linewidth=0.6)
        fig.tight_layout()
        fig.savefig(buffer, format="png", dpi=160)
        plt.close(fig)
    return buffer.getvalue()


def _sheet_title(name: str) -> str:
    cleaned = name.replace(" / ", " - ")
    cleaned = "".join("-" if char in "[]:*?/\\" else char for char in cleaned).strip()
    return (cleaned or "Bore")[:31]


def _write_number(cell, value: Any) -> None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        cell.value = None
        return
    if isinstance(value, str):
        cell.value = value
        return
    cell.value = float(value)
    cell.number_format = "0.000000"


def _write_sheet(workbook: Workbook, title: str, frame: pd.DataFrame, preface: str) -> None:
    sheet = workbook.create_sheet(_sheet_title(title))
    sheet["A1"] = preface
    sheet["A1"].font = Font(italic=True)
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(1, len(frame.columns)))
    for column, name in enumerate(frame.columns, start=1):
        cell = sheet.cell(2, column, name)
        cell.font = Font(bold=True)
    for row_index, row in enumerate(frame.itertuples(index=False), start=3):
        for column, value in enumerate(row, start=1):
            _write_number(sheet.cell(row_index, column), value)
    for column, name in enumerate(frame.columns, start=1):
        sheet.column_dimensions[get_column_letter(column)].width = min(42, max(14, len(str(name)) + 2))
    sheet.auto_filter.ref = f"A2:{get_column_letter(len(frame.columns))}{max(2, sheet.max_row)}"
    sheet.freeze_panes = "A3"


def _img_html(png: bytes, alt: str) -> str:
    encoded = base64.b64encode(png).decode("ascii")
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{encoded}"/>'


def _write_html(path: Path, source_name: str, sections: list[dict[str, Any]], written_at: str) -> None:
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>",
        f"<title>Area-equivalent bore design {html.escape(source_name)}</title>",
        "<style>body{font-family:Georgia,serif;margin:2rem;color:#222;max-width:980px;}"
        "img{max-width:100%;height:auto;}h1,h2{font-weight:normal;}</style></head><body>",
        f"<h1>Area-equivalent bore design: {html.escape(source_name)}</h1>",
        f"<p>Generated {html.escape(written_at)}. These values are not written to the database.</p>",
        "<p>A round probe and an oval probe of the same diameter seat at different depths. "
        "The oval probe is usually inserted further. At the oval-probe depth the minor diameter "
        "is intercepted from the round-probe curve. At the round-probe depth the major diameter "
        "is intercepted from the oval-probe curve. The design diameter is "
        "D<sub>eq</sub> = √(D<sub>min</sub> × D<sub>maj</sub>) at that station. "
        "Insertion depth locates the station. It is not the design quantity.</p>",
    ]
    for section in sections:
        parts.append(f"<h2>{html.escape(section['label'])}</h2>")
        if section.get("empty"):
            parts.append("<p>This section does not have both a round-probe curve and an oval-probe curve, so no area-equivalent diameter was intercepted.</p>")
            continue
        parts.append(
            f"<p>{section['design_count']} design diameters, "
            f"from {section['design_min']:.4f} in to {section['design_max']:.4f} in.</p>"
        )
        if section.get("png"):
            parts.append(_img_html(section["png"], f"{section['label']} area-equivalent design"))
    parts.append("</body></html>")
    path.write_text("".join(parts), encoding="utf-8")


def optimize_tables(tables: list[dict[str, Any]], source_name: str, destination: Path) -> list[Path]:
    """Write the area-equivalent workbook and HTML figure for bore tables."""
    destination.mkdir(parents=True, exist_ok=True)
    bore_tables = [table for table in tables if table.get("kind", "bore") == "bore"]
    bore_tables.sort(key=_report_sort_key)
    workbook = Workbook()
    default = workbook.active
    workbook.remove(default)
    design_frames: list[pd.DataFrame] = []
    sections: list[dict[str, Any]] = []
    for table in bore_tables:
        label = _overlay_label(table)
        calculated = intercept_area_equivalent(table["frame"], table.get("oal"))
        schedule = design_schedule(calculated, label)
        if calculated.empty or schedule.empty:
            sections.append({"label": label, "empty": True})
            continue
        _write_sheet(
            workbook,
            label,
            calculated,
            "Probe stations with the unknown axis intercepted. D_eq is the design diameter.",
        )
        design_frames.append(schedule)
        finite = schedule["Design diameter D_eq (in)"].to_numpy(dtype=float)
        sections.append(
            {
                "label": label,
                "empty": False,
                "design_count": int(np.isfinite(finite).sum()),
                "design_min": float(np.nanmin(finite)),
                "design_max": float(np.nanmax(finite)),
                "png": plot_area_equivalent_design(calculated, label),
            }
        )
    if design_frames:
        design = pd.concat(design_frames, ignore_index=True)
        _write_sheet(
            workbook,
            "Design",
            design,
            "Area-equivalent design diameters. Each bore is ordered by D_eq, not by insertion depth.",
        )
    else:
        sheet = workbook.create_sheet("Design")
        sheet["A1"] = "No bore had both a round-probe curve and an oval-probe curve."
    if "Design" in workbook.sheetnames:
        workbook.move_sheet("Design", offset=-workbook.sheetnames.index("Design"))
    stem = source_name.replace(" ", "_").replace("'", "")
    workbook_path = destination / f"{stem}_area_equivalent.xlsx"
    html_path = destination / f"{stem}_area_equivalent.html"
    workbook.save(workbook_path)
    written_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _write_html(html_path, source_name, sections, written_at)
    return [workbook_path, html_path]


def optimize_bore_source(source: Path, output_dir: Path | None = None) -> list[Path]:
    """Optimize a template-shaped workbook. Does not read or write PostgreSQL."""
    if not source.exists():
        raise FileNotFoundError(f"Bore optimizer source not found: {source}")
    tables = [table for table in load_bore_tables(source) if table.get("kind") == "bore"]
    if not tables:
        raise ValueError(f"No bore measurement tables were found in {source.name}.")
    destination = output_dir or (OUTPUT_DIR / source.stem)
    return optimize_tables(tables, source.stem, destination)


def optimize_bore_set(set_name: str, output_dir: Path | None = None) -> list[Path]:
    """Optimize one instrument set. Reads the set and does not write to PostgreSQL."""
    from db.sets import load_set_bore_tables

    set_row, tables = load_set_bore_tables(set_name)
    bore_tables = [table for table in tables if table.get("kind") == "bore"]
    if not bore_tables:
        raise ValueError(f"Set {set_name!r} has no bore rows to optimize.")
    label = str(set_row["name"]).replace(" ", "_").replace("'", "")
    destination = output_dir or (OUTPUT_DIR / label)
    return optimize_tables(bore_tables, str(set_row["name"]), destination)
