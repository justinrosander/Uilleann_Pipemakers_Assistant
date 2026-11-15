"""Execute the bore EDA plan: statistics, regressions, and plots.

Outputs an HTML report, a PDF report, or both. This module is invoked only
when the user runs the EDA agent against a workbook.
"""

from __future__ import annotations

import base64
import html
import warnings
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from agents.eda_agent import CORRELATION_THRESHOLD, load_brief
from agents.intake_agent import list_workbook_sheets, parse_workbook_sheet
from config import ROOT
from schema.template_schema import classify_sheet

EDA_OUTPUT_DIR = ROOT / "data" / "eda"


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def load_bore_tables(path: Path) -> list[dict[str, Any]]:
    """Load bore measurement tables from a template-shaped workbook."""
    tables: list[dict[str, Any]] = []
    for sheet_name in list_workbook_sheets(path):
        spec = classify_sheet(sheet_name)
        if spec is None or spec.kind == "toneholes":
            continue
        parsed = parse_workbook_sheet(path, sheet_name)
        for section in parsed.get("sections") or []:
            if section.get("kind") not in {None, "bore"}:
                continue
            frame = pd.DataFrame(section.get("rows") or [])
            if frame.empty or "Diameter" not in frame.columns:
                continue
            for column in ("Nominal", "Diameter", "Round", "Oval"):
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if frame["Diameter"].notna().sum() < 3:
                continue
            oal = None
            for label, value in (parsed.get("oals") or {}).items():
                if str(label).strip().lower() == "oal" and value is not None:
                    oal = float(value)
            tables.append(
                {
                    "sheet": sheet_name,
                    "section": section.get("name") or "main",
                    "family": parsed.get("instrument_family") or spec.instrument_family,
                    "title": parsed.get("instrument_name") or spec.instrument_name,
                    "units": parsed.get("units") or "",
                    "kind": "bore",
                    "oal": oal,
                    "frame": frame,
                }
            )
    tables.extend(load_tonehole_tables(path))
    return tables


_TONEHOLE_NUMERIC = (
    "d-bottom-bell",
    "d-top-reedseat",
    "d-ctr-bell",
    "d-axial",
    "d-xverse",
    "chimney",
    "ch_diam",
)


def load_tonehole_tables(path: Path) -> list[dict[str, Any]]:
    """Load tonehole rows from a template-shaped workbook."""
    tables: list[dict[str, Any]] = []
    for sheet_name in list_workbook_sheets(path):
        spec = classify_sheet(sheet_name)
        if spec is None or spec.kind != "toneholes":
            continue
        parsed = parse_workbook_sheet(path, sheet_name)
        for section in parsed.get("sections") or []:
            if section.get("kind") != "toneholes":
                continue
            frame = pd.DataFrame(section.get("rows") or [])
            if frame.empty:
                continue
            for column in _TONEHOLE_NUMERIC:
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if frame.select_dtypes(include="number").notna().any().any() is False:
                continue
            tables.append(
                {
                    "sheet": sheet_name,
                    "section": "toneholes",
                    "family": parsed.get("instrument_family") or spec.instrument_family,
                    "title": parsed.get("instrument_name") or spec.instrument_name,
                    "units": parsed.get("units") or "",
                    "kind": "toneholes",
                    "frame": frame,
                }
            )
    return tables


def univariate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.agg(["count", "mean", "std", "min", "median", "max"]).T
    summary["missing"] = numeric.isna().sum()
    return summary.reset_index(names="variable")


def strong_correlations(frame: pd.DataFrame, threshold: float = CORRELATION_THRESHOLD) -> list[dict[str, Any]]:
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] < 2 or len(numeric) < 3:
        return []
    corr = numeric.corr(method="pearson")
    pairs: list[dict[str, Any]] = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= threshold:
                pairs.append({"left": left, "right": right, "r": float(value)})
    pairs.sort(key=lambda item: abs(item["r"]), reverse=True)
    return pairs


def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    if total == 0:
        return None
    residual = float(np.sum((actual - predicted) ** 2))
    return 1.0 - residual / total


def _depth_series(frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Mean of Round and Oval when both exist; otherwise the available station."""
    round_values = _numeric(frame, "Round")
    oval_values = _numeric(frame, "Oval")
    if round_values.notna().any() and oval_values.notna().any():
        depth = pd.concat([round_values, oval_values], axis=1).mean(axis=1, skipna=True)
        return depth.to_numpy(dtype=float), "mean of Round and Oval"
    if round_values.notna().any():
        return round_values.to_numpy(dtype=float), "Round"
    if oval_values.notna().any():
        return oval_values.to_numpy(dtype=float), "Oval"
    nominal = _numeric(frame, "Nominal")
    return nominal.to_numpy(dtype=float), "Nominal"


def _exclude_above_throat(table: dict[str, Any]) -> bool:
    """Chanter, tenor, and baritone fits stop at the throat and continue toward the bell."""
    blob = f"{table.get('sheet', '')} {table.get('title', '')}".lower()
    if "drone" in blob or "bass" in blob:
        return False
    return any(name in blob for name in ("chanter", "tenor", "baritone"))


def _monotonic_profile(depth: np.ndarray, diameter: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average duplicate depths so diameter can be interpolated along depth."""
    order = np.argsort(depth)
    depth = depth[order]
    diameter = diameter[order]
    rounded = np.round(depth, 6)
    unique_depth, inverse = np.unique(rounded, return_inverse=True)
    unique_diameter = np.array(
        [diameter[inverse == index].mean() for index in range(len(unique_depth))]
    )
    return unique_depth, unique_diameter


def _ideal_diameter_profile(
    frame: pd.DataFrame, *, below_throat: bool
) -> tuple[np.ndarray, np.ndarray, str, int] | None:
    """Build the ideal circular diameter at common depths.

    Round and Oval are insertion depths for the same probe diameter: round probes
    follow the minor axis and oval probes the major axis. Both profiles are
    interpolated onto one depth grid, then combined as sqrt(D_minor * D_major),
    the area-equivalent circular diameter. That target lies between the two series.
    """
    diameter = _numeric(frame, "Diameter").to_numpy(dtype=float)
    round_depth = _numeric(frame, "Round").to_numpy(dtype=float)
    oval_depth = _numeric(frame, "Oval").to_numpy(dtype=float)
    both = np.isfinite(diameter) & np.isfinite(round_depth) & np.isfinite(oval_depth)
    excluded = 0
    if int(both.sum()) >= 4:
        probe = diameter[both]
        minor_depth = round_depth[both]
        major_depth = oval_depth[both]
        minor_z, minor_d = _monotonic_profile(minor_depth, probe)
        major_z, major_d = _monotonic_profile(major_depth, probe)
        if len(minor_z) < 2 or len(major_z) < 2:
            return None
        z0 = max(float(minor_z.min()), float(major_z.min()))
        z1 = min(float(minor_z.max()), float(major_z.max()))
        if z1 <= z0:
            return None
        if below_throat:
            throat = int(np.argmin(probe))
            throat_depth = float(min(minor_depth[throat], major_depth[throat]))
            z1 = min(z1, throat_depth)
            excluded = int(np.sum(np.maximum(minor_depth, major_depth) > throat_depth))
            if z1 <= z0:
                return None
        grid = np.linspace(z0, z1, num=max(20, int(both.sum())))
        minor_on_grid = np.interp(grid, minor_z, minor_d)
        major_on_grid = np.interp(grid, major_z, major_d)
        equivalent = np.sqrt(np.clip(minor_on_grid, 0, None) * np.clip(major_on_grid, 0, None))
        return grid, equivalent, "area-equivalent diameter at common depth", excluded

    depth, depth_label = _depth_series(frame)
    mask = np.isfinite(depth) & np.isfinite(diameter)
    x = depth[mask]
    y = diameter[mask]
    if below_throat and len(x) >= 1:
        throat = int(np.argmin(y))
        keep = x <= x[throat]
        excluded = int((~keep).sum())
        x = x[keep]
        y = y[keep]
    if len(x) < 4:
        return None
    return x, y, depth_label, excluded


def fit_ideal_profile(frame: pd.DataFrame, *, below_throat: bool = False) -> dict[str, Any] | None:
    """Fit a straight taper and several nonlinear curves to the ideal circular diameter."""
    profile = _ideal_diameter_profile(frame, below_throat=below_throat)
    if profile is None:
        return None
    x, y, depth_label, excluded = profile
    if len(x) < 4:
        return None
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    models: list[dict[str, Any]] = []

    def add_model(name: str, detail: str, pred: np.ndarray) -> None:
        if pred.shape != y.shape or not np.all(np.isfinite(pred)):
            return
        models.append(
            {
                "name": name,
                "detail": detail,
                "r2": _r_squared(y, pred),
                "pred": pred,
            }
        )

    linear_coef = np.polyfit(x, y, 1)
    linear_pred = np.polyval(linear_coef, x)
    add_model(
        "Linear",
        _coef_text("Linear", [float(item) for item in linear_coef], None).split(";")[0],
        linear_pred,
    )
    for degree, name in ((2, "Quadratic"), (3, "Cubic"), (4, "Quartic"), (5, "Quintic")):
        if len(x) < degree + 4:
            continue
        coef = np.polyfit(x, y, degree)
        add_model(
            name,
            _coef_text(name, [float(item) for item in coef], None).split(";")[0],
            np.polyval(coef, x),
        )
    if np.all(y > 0):
        slope, intercept = np.polyfit(x, np.log(y), 1)
        add_model(
            "Exponential",
            f"Exponential: diameter = {float(np.exp(intercept)):.6g} * exp({float(slope):.6g} * depth)",
            np.exp(intercept + slope * x),
        )
    if np.all(x > 0) and np.all(y > 0):
        exponent, log_scale = np.polyfit(np.log(x), np.log(y), 1)
        add_model(
            "Power",
            f"Power: diameter = {float(np.exp(log_scale)):.6g} * depth^{float(exponent):.6g}",
            np.exp(log_scale) * np.power(x, exponent),
        )
    if np.all(x > 0):
        log_slope, log_intercept = np.polyfit(np.log(x), y, 1)
        add_model(
            "Logarithmic",
            f"Logarithmic: diameter = {float(log_intercept):.6g} + {float(log_slope):.6g} * ln(depth)",
            log_intercept + log_slope * np.log(x),
        )
    if len(x) >= 8:
        try:
            from scipy.interpolate import UnivariateSpline

            spline = UnivariateSpline(x, y, k=3, s=max(len(x) * float(np.var(y)) * 0.05, 1e-8))
            add_model("Spline", "Smoothing spline of the area-equivalent diameter", spline(x))
        except Exception:
            pass
    if not models:
        return None
    return {
        "depth_label": depth_label,
        "n": int(len(x)),
        "excluded_above_throat": excluded,
        "depth": x,
        "diameter": y,
        "models": models,
        "linear_coef": [float(item) for item in linear_coef],
        "linear_r2": models[0]["r2"],
        "linear_pred": linear_pred,
    }


def _polynomial_prediction(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray | None:
    """In-sample least-squares polynomial prediction. None when the fit is unavailable."""
    if len(x) < degree + 1:
        return None
    try:
        coefficients, center, scale = _least_squares_curve(x, y, degree)
        predicted = _evaluate_curve(coefficients, center, scale, x)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if predicted.shape != y.shape or not np.all(np.isfinite(predicted)):
        return None
    return predicted


def compare_profile_models(position: np.ndarray, diameter: np.ndarray) -> list[dict[str, Any]]:
    """Fit the bore-profile models on one set of points so their R² values compare.

    The 5th-order row uses the same least-squares polynomial as the ideal-round figure.
    """
    order = np.argsort(position)
    x = np.asarray(position, dtype=float)[order]
    y = np.asarray(diameter, dtype=float)[order]
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    rows: list[dict[str, Any]] = []
    if len(x) < 4:
        return rows

    def add(name: str, predicted: np.ndarray | None) -> None:
        if predicted is None:
            return
        rows.append({"name": name, "r2": _r_squared(y, predicted), "n": int(len(x))})

    for degree, name in (
        (1, "Linear cone"),
        (2, "Quadratic"),
        (3, "Cubic"),
        (4, "Quartic"),
        (5, "5th-order polynomial"),
    ):
        add(name, _polynomial_prediction(x, y, degree))
    if np.all(y > 0):
        slope, intercept = np.polyfit(x, np.log(y), 1)
        add("Exponential", np.exp(intercept + slope * x))
    if np.all(x > 0) and np.all(y > 0):
        exponent, log_scale = np.polyfit(np.log(x), np.log(y), 1)
        add("Power", np.exp(log_scale) * np.power(x, exponent))
    if np.all(x > 0):
        log_slope, log_intercept = np.polyfit(np.log(x), y, 1)
        add("Logarithmic", log_intercept + log_slope * np.log(x))
    if len(x) >= 8:
        try:
            from scipy.interpolate import UnivariateSpline

            spline = UnivariateSpline(x, y, k=3, s=max(len(x) * float(np.var(y)) * 0.05, 1e-8))
            add("Spline", np.asarray(spline(x), dtype=float))
        except Exception:
            pass
    return rows


def _model_comparison_for_table(
    table: dict[str, Any], fit: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], str | None]:
    """Return comparable model rows and the basis used to fit them."""
    measured = _measured_equivalent_points(table["frame"], table.get("oal"))
    if measured is not None:
        diameter, distance, _top = measured
        return compare_profile_models(distance, diameter), "measured-equivalent"
    if not fit or not fit.get("models"):
        return [], None
    rows = []
    for model in fit["models"]:
        name = "5th-order polynomial" if model["name"] == "Quintic" else model["name"]
        if name == "Linear":
            name = "Linear cone"
        rows.append({"name": name, "r2": model["r2"], "n": fit["n"]})
    return rows, "probe-depth"


def _model_comparison_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": row["name"],
                "R²": "" if row["r2"] is None else f"{row['r2']:.4f}",
                "n": row["n"],
            }
            for row in rows
        ]
    )


def _label_for(table: dict[str, Any]) -> str:
    section = table["section"]
    if section in {None, "main"}:
        return str(table["title"])
    return f"{table['title']} / {section}"


def _tonehole_depth_column(frame: pd.DataFrame) -> str | None:
    for column in ("d-bottom-bell", "d-ctr-bell", "d-top-reedseat"):
        if column in frame.columns and pd.to_numeric(frame[column], errors="coerce").notna().any():
            return column
    return None


def _tonehole_size(frame: pd.DataFrame) -> pd.Series:
    if "ch_diam" in frame.columns and pd.to_numeric(frame["ch_diam"], errors="coerce").notna().any():
        return pd.to_numeric(frame["ch_diam"], errors="coerce")
    axial = pd.to_numeric(frame["d-axial"], errors="coerce") if "d-axial" in frame.columns else None
    transverse = (
        pd.to_numeric(frame["d-xverse"], errors="coerce") if "d-xverse" in frame.columns else None
    )
    if axial is not None and transverse is not None:
        return pd.concat([axial, transverse], axis=1).mean(axis=1, skipna=True)
    if axial is not None:
        return axial
    if transverse is not None:
        return transverse
    return pd.Series(dtype=float)


def plot_toneholes(table: dict[str, Any]) -> bytes | None:
    """Place each tonehole by diameter and depth. Depth is vertical."""
    frame = table["frame"]
    depth_column = _tonehole_depth_column(frame)
    size = _tonehole_size(frame)
    if depth_column is None or size.empty:
        return None
    depth = pd.to_numeric(frame[depth_column], errors="coerce")
    labels = frame["keyhole"] if "keyhole" in frame.columns else pd.Series([""] * len(frame))
    mask = depth.notna() & size.notna()
    if int(mask.sum()) < 1:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(size[mask], depth[mask], s=36, color="#1f4e79")
    for x_value, y_value, label in zip(size[mask], depth[mask], labels[mask]):
        ax.annotate(str(label), (x_value, y_value), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Tonehole diameter")
    ax.set_ylabel(depth_column)
    ax.set_title(f"{_label_for(table)} toneholes")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    return buffer.getvalue()


def tonehole_bore_links(
    tonehole_table: dict[str, Any], bore_tables: list[dict[str, Any]]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Match each tonehole depth to the nearest bore diameter on the same instrument."""
    frame = tonehole_table["frame"]
    depth_column = _tonehole_depth_column(frame)
    if depth_column is None:
        return pd.DataFrame(), []
    same_instrument = [
        table
        for table in bore_tables
        if table.get("kind", "bore") == "bore" and table.get("title") == tonehole_table.get("title")
    ]
    records = []
    for _, row in frame.iterrows():
        depth = pd.to_numeric(pd.Series([row.get(depth_column)]), errors="coerce").iloc[0]
        hole_size = _tonehole_size(pd.DataFrame([row])).iloc[0] if len(frame) else None
        nearest = None
        nearest_distance = None
        for bore in same_instrument:
            bore_depth, _ = _depth_series(bore["frame"])
            bore_diameter = _numeric(bore["frame"], "Diameter").to_numpy(dtype=float)
            mask = np.isfinite(bore_depth) & np.isfinite(bore_diameter)
            if not mask.any() or not np.isfinite(depth):
                continue
            distances = np.abs(bore_depth[mask] - float(depth))
            index = int(np.argmin(distances))
            distance = float(distances[index])
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest = float(bore_diameter[mask][index])
        records.append(
            {
                "keyhole": row.get("keyhole"),
                "hole_depth": depth,
                "depth_column": depth_column,
                "hole_diameter": hole_size,
                "nearest_bore_diameter": nearest,
            }
        )
    link_frame = pd.DataFrame(records)
    correlations: list[dict[str, Any]] = []
    numeric = link_frame[["hole_depth", "hole_diameter", "nearest_bore_diameter"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if len(numeric.dropna(how="any")) >= 3:
        corr = numeric.corr(method="pearson")
        for left, right in (
            ("hole_depth", "nearest_bore_diameter"),
            ("hole_diameter", "nearest_bore_diameter"),
            ("hole_depth", "hole_diameter"),
        ):
            value = corr.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= CORRELATION_THRESHOLD:
                correlations.append({"left": left, "right": right, "r": float(value)})
    return link_frame, correlations


def _scatter_round_oval(ax, frame: pd.DataFrame) -> None:
    diameter = _numeric(frame, "Diameter")
    round_values = _numeric(frame, "Round")
    oval_values = _numeric(frame, "Oval")
    if round_values.notna().any():
        ax.scatter(diameter, round_values, s=18, label="Round", color="#1f4e79")
    if oval_values.notna().any():
        ax.scatter(diameter, oval_values, s=18, label="Oval", color="#c45911")


def _save_figure(fig) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    return buffer.getvalue()


def _reconstructed_profile_instrument(table: dict[str, Any]) -> str | None:
    """Chanter and regulator bores drawn as reconstructed probe profiles."""
    blob = f"{table.get('sheet', '')} {table.get('title', '')}".lower()
    section = str(table.get("section") or "main").lower()
    if "drone" in blob or section in {"reed seat", "u-bend", "toneholes"}:
        return None
    if "chanter" in blob and section in {"main", "upper bore"}:
        return "chanter"
    if "tenor" in blob and section == "main":
        return "tenor regulator"
    if "baritone" in blob and section == "main":
        return "baritone regulator"
    if "bass" in blob and (
        section in {"upper bore", "lower bore"} or "upper" in blob or "lower" in blob
    ):
        return "bass regulator"
    return None


def _distance_axis_limit(length: float) -> float:
    """Inch limit just past the bore length. A 14.47 in chanter stays on 0–15."""
    limit = float(np.ceil(length - 1e-6))
    if length >= limit - 1e-6:
        limit += 1.0
    return limit


def _probe_axis_profiles(
    frame: pd.DataFrame, oal: float | None
) -> dict[str, Any] | None:
    """Minor and major diameters against distance from the top of the section."""
    diameter_in = _numeric(frame, "Diameter").to_numpy(dtype=float)
    round_depth = _numeric(frame, "Round").to_numpy(dtype=float)
    oval_depth = _numeric(frame, "Oval").to_numpy(dtype=float)
    both = np.isfinite(diameter_in) & np.isfinite(round_depth) & np.isfinite(oval_depth)
    if int(both.sum()) < 4:
        return None
    probe = diameter_in[both]
    minor_depth = round_depth[both]
    major_depth = oval_depth[both]
    top = float(oal) if oal is not None and np.isfinite(oal) else float(
        max(np.nanmax(minor_depth), np.nanmax(major_depth))
    )
    minor_distance, minor_d = _monotonic_profile(top - minor_depth, probe)
    major_distance, major_d = _monotonic_profile(top - major_depth, probe)
    throat = int(np.argmin(probe))
    throat_distance = float(min(top - minor_depth[throat], top - major_depth[throat]))
    bell_distance = float(min(minor_distance.max(), major_distance.max()))
    if not np.isfinite(throat_distance) or not np.isfinite(bell_distance):
        return None
    if bell_distance <= throat_distance:
        return None
    return {
        "minor_distance": minor_distance,
        "minor_diameter": minor_d,
        "major_distance": major_distance,
        "major_diameter": major_d,
        "throat_distance": throat_distance,
        "bell_distance": bell_distance,
        "top": top,
        "n_pairs": int(both.sum()),
    }


def _area_equivalent_curve(
    frame: pd.DataFrame, oal: float | None
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Throat-to-bell area-equivalent diameter and distance from the top.

    Returns diameter in inches, distance from the top in inches, and the
    section length used to place the probes. Round-probe depths are the minor
    axis and oval-probe depths are the major axis.
    """
    profiles = _probe_axis_profiles(frame, oal)
    if profiles is None:
        return None
    grid = np.linspace(
        profiles["throat_distance"],
        profiles["bell_distance"],
        num=max(40, profiles["n_pairs"]),
    )
    minor_on_grid = np.interp(grid, profiles["minor_distance"], profiles["minor_diameter"])
    major_on_grid = np.interp(grid, profiles["major_distance"], profiles["major_diameter"])
    equivalent = np.sqrt(np.clip(minor_on_grid, 0, None) * np.clip(major_on_grid, 0, None))
    return equivalent, grid, float(profiles["top"])


def _measured_equivalent_points(
    frame: pd.DataFrame, oal: float | None
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Area-equivalent diameter at each round-probe station from throat to bell."""
    profiles = _probe_axis_profiles(frame, oal)
    if profiles is None:
        return None
    distance = profiles["minor_distance"]
    lo = max(profiles["throat_distance"], float(profiles["major_distance"].min()))
    hi = min(profiles["bell_distance"], float(profiles["major_distance"].max()))
    keep = (distance >= lo) & (distance <= hi)
    if int(keep.sum()) < 6:
        return None
    station = distance[keep]
    minor_at = profiles["minor_diameter"][keep]
    major_at = np.interp(station, profiles["major_distance"], profiles["major_diameter"])
    equivalent = np.sqrt(np.clip(minor_at, 0, None) * np.clip(major_at, 0, None))
    return equivalent, station, float(profiles["top"])


def _least_squares_curve(x: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.ndarray, float, float]:
    """Least-squares polynomial of y on x, fit in a centered coordinate."""
    center = float(np.mean(x))
    scale = float(np.std(x))
    if scale < 1e-9:
        scale = 1.0
    rank_warning = getattr(getattr(np, "exceptions", np), "RankWarning", Warning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rank_warning)
        coefficients = np.polyfit((x - center) / scale, y, degree)
    return coefficients, center, scale


def _evaluate_curve(
    coefficients: np.ndarray, center: float, scale: float, x: np.ndarray
) -> np.ndarray:
    return np.polyval(coefficients, (x - center) / scale)


def plot_reconstructed_bore_profiles(
    frame: pd.DataFrame,
    oal: float | None = None,
    *,
    instrument: str = "chanter",
) -> bytes:
    """Round, oval, and area-equivalent profiles from throat to bell.

    Round-probe depths trace the minor axis. Oval-probe depths trace the major
    axis. Both are placed on a common distance from the top of the section.
    """
    curve = _area_equivalent_curve(frame, oal)
    if curve is None:
        raise ValueError("Not enough probe pairs to reconstruct the bore.")
    equivalent, grid, top = curve
    diameter_in = _numeric(frame, "Diameter").to_numpy(dtype=float)
    round_depth = _numeric(frame, "Round").to_numpy(dtype=float)
    oval_depth = _numeric(frame, "Oval").to_numpy(dtype=float)
    both = np.isfinite(diameter_in) & np.isfinite(round_depth) & np.isfinite(oval_depth)
    probe = diameter_in[both]
    minor_depth = round_depth[both]
    major_depth = oval_depth[both]
    minor_distance, minor_d = _monotonic_profile(top - minor_depth, probe)
    major_distance, major_d = _monotonic_profile(top - major_depth, probe)
    throat = int(np.argmin(probe))
    throat_distance = float(min(top - minor_depth[throat], top - major_depth[throat]))

    style = {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
    buffer = BytesIO()
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(10, 6))
        minor_show = minor_distance >= throat_distance
        major_show = major_distance >= throat_distance
        ax.plot(
            minor_d[minor_show],
            minor_distance[minor_show],
            color="#c0392b",
            linewidth=1,
            marker="o",
            markersize=4.5,
            label="Minor axis (round probes)",
        )
        ax.plot(
            major_d[major_show],
            major_distance[major_show],
            color="#1e8449",
            linewidth=1,
            marker="s",
            markersize=4.5,
            label="Major axis (oval probes)",
        )
        ax.plot(
            equivalent,
            grid,
            color="#1a5276",
            linewidth=2.4,
            marker="D",
            markersize=4.5,
            markevery=max(1, len(grid) // 18),
            label=r"Area-equivalent $D_{eq}=\sqrt{D_{min}\times D_{maj}}$",
        )
        ax.set_ylim(0, _distance_axis_limit(top))
        ax.set_xlabel("Bore diameter (in)")
        ax.set_ylabel(f"Distance from top of {instrument} (in)")
        ax.set_title("Reconstructed Bore Profiles from Probe Measurements")
        ax.legend(loc="upper left", frameon=True)
        ax.grid(True, color="#d0d0d0", linewidth=0.6)
        ax.annotate(
            r"$D_{eq}=\sqrt{D_{minor}\times D_{major}}$ is the area-equivalent circular diameter.",
            xy=(0.99, 0.02),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#333333",
        )
        fig.tight_layout()
        fig.savefig(buffer, format="png", dpi=220)
        plt.close(fig)
    return buffer.getvalue()


def plot_ideal_round_regression(
    frame: pd.DataFrame,
    oal: float | None = None,
    *,
    instrument: str = "chanter",
) -> bytes:
    """Measured area-equivalent points with a linear cone and a 5th-order fit.

    Diameter is fit against distance from the top by least squares. The linear
    fit is the straight taper. The degree-5 polynomial is the nonlinear profile.
    """
    measured = _measured_equivalent_points(frame, oal)
    if measured is None:
        raise ValueError("Not enough area-equivalent points to fit the ideal round profile.")
    diameter, distance, top = measured
    span = np.linspace(float(distance.min()), float(distance.max()), num=400)
    linear = _least_squares_curve(distance, diameter, 1)
    quintic = _least_squares_curve(distance, diameter, 5)
    style = {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
    buffer = BytesIO()
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            diameter,
            distance,
            s=28,
            color="#2471A3",
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
            label=r"Measured $D_{eq}$",
        )
        ax.plot(
            _evaluate_curve(*quintic, span),
            span,
            color="#6C3483",
            linestyle="-",
            linewidth=2.2,
            zorder=2,
            label="5th-order polynomial",
        )
        ax.plot(
            _evaluate_curve(*linear, span),
            span,
            color="#E67E22",
            linestyle="--",
            linewidth=2.0,
            zorder=2,
            label="Linear cone",
        )
        ax.set_ylim(0, _distance_axis_limit(top))
        ax.set_xlabel("Bore diameter (in)")
        ax.set_ylabel(f"Distance from top of {instrument} (in)")
        ax.set_title("Ideal Round Profile via Nonlinear Regression")
        ax.legend(loc="upper left", frameon=True)
        ax.grid(True, color="#d0d0d0", linewidth=0.6)
        fig.tight_layout()
        fig.savefig(buffer, format="png", dpi=220)
        plt.close(fig)
    return buffer.getvalue()


_EQUIVALENT_OVERLAY_STYLES = (
    ("#1a5276", "-"),
    ("#c0392b", "--"),
    ("#1e8449", "-."),
    ("#6c3483", ":"),
    ("#b9770e", (0, (5, 1))),
    ("#1a6f8a", (0, (3, 1, 1, 1))),
    ("#7b241c", (0, (1, 1))),
    ("#196f3d", (0, (5, 2))),
    ("#4a235a", "-"),
    ("#7e5109", "--"),
    ("#1b4f72", "-."),
    ("#922b21", ":"),
)


def _overlay_label(table: dict[str, Any]) -> str:
    """Name a bore on the overlay, including upper or lower when the section is unnamed."""
    label = _label_for(table)
    section = str(table.get("section") or "main").lower()
    sheet = str(table.get("sheet") or "").lower()
    if section in {"main", ""}:
        if "lower" in sheet and "lower" not in label.lower():
            return f"{table['title']} / Lower Bore"
        if "upper" in sheet and "upper" not in label.lower():
            return f"{table['title']} / Upper Bore"
    return label


def plot_area_equivalent_overlay(tables: list[dict[str, Any]]) -> bytes | None:
    """Overlay throat-to-bell area-equivalent profiles. Round and oval stay off the figure."""
    curves: list[tuple[dict[str, Any], np.ndarray, np.ndarray, float]] = []
    for table in tables:
        if table.get("kind", "bore") != "bore":
            continue
        curve = _area_equivalent_curve(table["frame"], table.get("oal"))
        if curve is None:
            continue
        diameter, distance, top = curve
        curves.append((table, diameter, distance, top))
    if len(curves) < 2:
        return None
    curves.sort(key=lambda item: _report_sort_key(item[0]))
    y_limit = max(_distance_axis_limit(top) for _, _, _, top in curves)
    style = {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    }
    buffer = BytesIO()
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(10, 6))
        for index, (table, diameter, distance, _) in enumerate(curves):
            color, linestyle = _EQUIVALENT_OVERLAY_STYLES[
                index % len(_EQUIVALENT_OVERLAY_STYLES)
            ]
            ax.plot(
                diameter,
                distance,
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=_overlay_label(table),
            )
        ax.set_ylim(0, y_limit)
        ax.set_xlabel("Bore diameter (in)")
        ax.set_ylabel("Distance from top (in)")
        ax.set_title("Reconstructed area-equivalent bore profiles")
        ax.legend(loc="upper left", frameon=True)
        ax.grid(True, color="#d0d0d0", linewidth=0.6)
        ax.annotate(
            r"Each curve is $D_{eq}=\sqrt{D_{minor}\times D_{major}}$ from throat to bell.",
            xy=(0.99, 0.02),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#333333",
        )
        fig.tight_layout()
        fig.savefig(buffer, format="png", dpi=220)
        plt.close(fig)
    return buffer.getvalue()


def plot_profile_curves(table: dict[str, Any], fit: dict[str, Any] | None) -> list[dict[str, Any]]:
    """One figure per regression, each with the round and oval measurements."""
    styles = {
        "Linear": ("#548235", "-"),
        "Quadratic": ("#833c0c", "--"),
        "Cubic": ("#1f4e79", ":"),
        "Quartic": ("#5b2c6f", "-."),
        "Quintic": ("#1a5276", (0, (1, 1))),
        "Exponential": ("#c45911", (0, (3, 1, 1, 1))),
        "Power": ("#7030a0", (0, (5, 1))),
        "Logarithmic": ("#385723", (0, (5, 2))),
        "Spline": ("#000000", "-"),
    }
    figures: list[dict[str, Any]] = []
    models = fit.get("models", []) if fit else []
    if not models:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        _scatter_round_oval(ax, table["frame"])
        ax.set_xlabel("Diameter")
        ax.set_ylabel("Depth")
        ax.set_title(_label_for(table))
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figures.append({"name": "Measurements", "png": _save_figure(fig)})
        return figures
    for model in models:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        _scatter_round_oval(ax, table["frame"])
        color, linestyle = styles.get(model["name"], ("#666666", "--"))
        label = "Linear (straight taper)" if model["name"] == "Linear" else model["name"]
        ax.plot(model["pred"], fit["depth"], color=color, linestyle=linestyle, label=label)
        ax.set_xlabel("Diameter")
        ax.set_ylabel("Depth")
        ax.set_title(f"{_label_for(table)} — {label}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figures.append({"name": model["name"], "png": _save_figure(fig)})
    return figures


def plot_chanter_vs_regulators(tables: list[dict[str, Any]]) -> bytes | None:
    selected = [
        table
        for table in tables
        if table["family"] in {"chanter", "regulator"} and table["section"] in {"main", "Upper Bore"}
    ]
    if len(selected) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for table in selected:
        depth, _ = _depth_series(table["frame"])
        diameter = _numeric(table["frame"], "Diameter").to_numpy(dtype=float)
        mask = np.isfinite(depth) & np.isfinite(diameter)
        ax.plot(diameter[mask], depth[mask], marker="o", markersize=3, label=_label_for(table))
    ax.set_xlabel("Diameter")
    ax.set_ylabel("Depth")
    ax.set_title("Chanter and regulator bore profiles")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    return buffer.getvalue()


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _table_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>No numeric summary.</p>"
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = "".join(f"<td>{html.escape(_fmt(value))}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _img_html(png: bytes, alt: str) -> str:
    encoded = base64.b64encode(png).decode("ascii")
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{encoded}"/>'


def _append_reamer_html(parts: list[str], reamer: dict[str, Any] | None) -> None:
    """Cross-bore PCA, residual signatures, and the major-axis hypothesis checks."""
    if not reamer:
        return
    parts.append("<h2>Area-equivalent and major-axis profiles</h2>")
    parts.append(_img_html(reamer["figure"], "Area-equivalent and major-axis profiles"))
    for section in reamer.get("sections") or []:
        if section.get("heading"):
            parts.append(f"<h3>{html.escape(section['heading'])}</h3>")
        for paragraph in section.get("paragraphs") or []:
            parts.append(f"<p>{html.escape(paragraph)}</p>")
    comparison = reamer.get("comparison")
    if comparison is not None and not comparison.empty:
        parts.append(_table_html(comparison))


def _append_shape_html(parts: list[str], shape: dict[str, Any] | None) -> None:
    """Add the derived-feature univariate, bivariate, and multivariate sections."""
    if not shape:
        return
    parts.append("<h3>Derived bore features</h3>")
    for note in shape["notes"]:
        parts.append(f"<p>{html.escape(note)}</p>")
    parts.append("<h4>Summary statistics</h4>")
    parts.append(_table_html(shape["summary"]))
    if shape.get("area") is not None and not shape["area"].empty:
        parts.append("<h4>Area change from ovality</h4>")
        parts.append(_table_html(shape["area"]))
    if shape.get("taper_choice") is not None and not shape["taper_choice"].empty:
        parts.append("<h4>Ideal-diameter choices</h4>")
        parts.append(_table_html(shape["taper_choice"]))
    for figure in shape["figures"]:
        parts.append(f"<h4>{html.escape(figure['name'])}</h4>")
        parts.append(_img_html(figure["png"], figure["name"]))
    if shape.get("recommendations"):
        parts.append("<h4>Reamer notes</h4><ul>")
        for note in shape["recommendations"]:
            parts.append(f"<li>{html.escape(note)}</li>")
        parts.append("</ul>")


def _coef_text(name: str, coefficients: list[float], r2: float | None) -> str:
    terms = ", ".join(f"{value:.6g}" for value in coefficients)
    fit = "n/a" if r2 is None else f"{r2:.4f}"
    return f"{name} coefficients (highest degree first): {terms}; R² = {fit}"


def _prepare_formats(formats: set[str]) -> set[str]:
    requested = {item.lower() for item in formats}
    unknown = requested - {"html", "pdf"}
    if unknown or not requested:
        raise ValueError("Format must be html, pdf, or both.")
    return requested


def execute_eda(
    source: Path,
    formats: set[str],
    output_dir: Path | None = None,
    observations: list[str] | None = None,
) -> list[Path]:
    """Run the EDA from a workbook and write the requested report formats."""
    if not source.exists():
        raise FileNotFoundError(f"EDA source not found: {source}")
    tables = load_bore_tables(source)
    if not tables:
        raise ValueError(f"No bore measurement tables were found in {source.name}.")
    return _write_eda_report(
        source,
        tables,
        _prepare_formats(formats),
        output_dir or (EDA_OUTPUT_DIR / source.stem),
        observations,
    )


def execute_eda_set(
    set_name: str,
    formats: set[str],
    output_dir: Path | None = None,
    observations: list[str] | None = None,
) -> list[Path]:
    """Run the EDA from one instrument set stored in PostgreSQL."""
    from db.sets import load_set_bore_tables

    set_row, tables = load_set_bore_tables(set_name)
    if not tables:
        raise ValueError(f"Set {set_name!r} has no bore rows to analyze.")
    label = set_row["name"].replace(" ", "_")
    destination = output_dir or (EDA_OUTPUT_DIR / label)
    return _write_eda_report(
        Path(set_row["name"]),
        tables,
        _prepare_formats(formats),
        destination,
        observations,
    )


def _report_sort_key(table: dict[str, Any]) -> tuple[int, int, str]:
    """Chanter, tenor, baritone, bass, then the drones in that same order."""
    blob = f"{table.get('sheet', '')} {table.get('title', '')}".lower()
    sheet = str(table.get("sheet") or "").lower()
    section = str(table.get("section") or "main").lower()
    drone = "drone" in blob
    if drone and "tenor" in blob:
        rank = 4
    elif drone and "baritone" in blob:
        rank = 5
    elif drone and "bass" in blob:
        rank = 6
    elif "chanter" in blob:
        rank = 0
    elif "tenor" in blob:
        rank = 1
    elif "baritone" in blob:
        rank = 2
    elif "bass" in blob:
        rank = 3
    else:
        rank = 9
    if "reed" in sheet or section == "reed seat":
        section_rank = 0
    elif "upper" in sheet or section == "upper bore":
        section_rank = 2
    elif "lower" in sheet or section == "lower bore":
        section_rank = 4
    elif section == "j1":
        section_rank = 5
    elif section == "j2":
        section_rank = 6
    elif section == "j3":
        section_rank = 7
    elif section == "slide":
        section_rank = 8
    elif section == "toneholes" or "tonehole" in sheet:
        section_rank = 15
    else:
        section_rank = 1
    return (rank, section_rank, sheet)


def _write_eda_report(
    source: Path,
    tables: list[dict[str, Any]],
    requested: set[str],
    destination: Path,
    observations: list[str] | None = None,
) -> list[Path]:
    """Run the EDA and write the requested report formats."""
    context, objectives = load_brief()
    tables = sorted(tables, key=_report_sort_key)
    bore_tables = [table for table in tables if table.get("kind", "bore") == "bore"]
    analyses: list[dict[str, Any]] = []
    for table in tables:
        if table.get("kind") == "toneholes":
            links, link_correlations = tonehole_bore_links(table, bore_tables)
            analyses.append(
                {
                    "table": table,
                    "kind": "toneholes",
                    "univariate": univariate_summary(table["frame"]),
                    "correlations": strong_correlations(table["frame"]),
                    "fit": None,
                    "plot": plot_toneholes(table),
                    "bore_links": links,
                    "link_correlations": link_correlations,
                }
            )
            continue
        fit = fit_ideal_profile(
            table["frame"], below_throat=_exclude_above_throat(table)
        )
        reconstructed = None
        ideal_round = None
        instrument = _reconstructed_profile_instrument(table)
        if instrument:
            try:
                reconstructed = plot_reconstructed_bore_profiles(
                    table["frame"], table.get("oal"), instrument=instrument
                )
            except ValueError:
                reconstructed = None
            try:
                ideal_round = plot_ideal_round_regression(
                    table["frame"], table.get("oal"), instrument=instrument
                )
            except ValueError:
                ideal_round = None
        model_comparison, model_basis = _model_comparison_for_table(table, fit)
        from agents.eda_shape import analyze_bore_shape

        shape = analyze_bore_shape(table)
        analyses.append(
            {
                "table": table,
                "kind": "bore",
                "univariate": univariate_summary(table["frame"]),
                "correlations": strong_correlations(table["frame"]),
                "fit": fit,
                "plots": plot_profile_curves(table, fit),
                "reconstructed": reconstructed,
                "ideal_round": ideal_round,
                "model_comparison": model_comparison,
                "model_basis": model_basis,
                "shape": shape,
            }
        )
    comparison = plot_chanter_vs_regulators(bore_tables)
    equivalent_overlay = plot_area_equivalent_overlay(bore_tables)
    from agents.eda_shape import compare_reamer_hypotheses

    reamer = compare_reamer_hypotheses(bore_tables)
    written_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if reamer and reamer.get("markdown"):
        notes_path = destination / f"{source.stem}_reamer_notes.md"
        notes_path.write_text(reamer["markdown"], encoding="utf-8")
        paths.append(notes_path)
    if "html" in requested:
        html_path = destination / f"{source.stem}_eda.html"
        html_path.write_text(
            _render_html(
                source,
                context,
                objectives,
                analyses,
                comparison,
                equivalent_overlay,
                reamer,
                written_at,
                observations,
            ),
            encoding="utf-8",
        )
        paths.append(html_path)
    if "pdf" in requested:
        pdf_path = destination / f"{source.stem}_eda.pdf"
        _render_pdf(
            pdf_path,
            source,
            context,
            objectives,
            analyses,
            comparison,
            equivalent_overlay,
            reamer,
            written_at,
            observations,
        )
        paths.append(pdf_path)
    return paths


def _observation_lines(observations: list[str] | None) -> list[str]:
    notes = [item.strip() for item in (observations or []) if item and item.strip()]
    if not notes:
        return []
    lines = ["Operator observations", "These notes were supplied for this report only."]
    lines.extend(f"- {item}" for item in notes)
    lines.append("")
    return lines


def _observation_html(observations: list[str] | None) -> list[str]:
    notes = [item.strip() for item in (observations or []) if item and item.strip()]
    if not notes:
        return []
    items = "".join(f"<li>{html.escape(item)}</li>" for item in notes)
    return [
        "<h2>Operator observations</h2>",
        "<p>These notes were supplied for this report only.</p>",
        f"<ul>{items}</ul>",
    ]


def _render_html(
    source: Path,
    context: str,
    objectives: list[str],
    analyses: list[dict[str, Any]],
    comparison: bytes | None,
    equivalent_overlay: bytes | None,
    reamer: dict[str, Any] | None,
    written_at: str,
    observations: list[str] | None = None,
) -> str:
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>",
        f"<title>EDA {html.escape(source.name)}</title>",
        "<style>body{font-family:Georgia,serif;margin:2rem;color:#222;}"
        "table{border-collapse:collapse;margin:1rem 0;}td,th{border:1px solid #ccc;padding:0.3rem 0.5rem;}"
        "img{max-width:100%;height:auto;}h1,h2{font-weight:normal;}</style></head><body>",
        f"<h1>Bore EDA: {html.escape(source.name)}</h1>",
        f"<p>Generated {html.escape(written_at)}</p>",
        f"<h2>Business context</h2><p>{html.escape(context)}</p>",
        *_observation_html(observations),
        "<h2>Analytical objectives</h2><ol>",
    ]
    parts.extend(f"<li>{html.escape(item)}</li>" for item in objectives)
    parts.append("</ol>")
    if comparison:
        parts.append("<h2>Chanter and regulator comparison</h2>")
        parts.append(_img_html(comparison, "Chanter and regulator profiles"))
    if equivalent_overlay:
        parts.append("<h2>Reconstructed area-equivalent bores</h2>")
        parts.append(
            "<p>Each curve is the area-equivalent diameter from the throat to the bell. "
            "Distance is measured from the top of that bore.</p>"
        )
        parts.append(
            _img_html(
                equivalent_overlay,
                "Reconstructed area-equivalent bore profiles",
            )
        )
    _append_reamer_html(parts, reamer)
    for item in analyses:
        table = item["table"]
        parts.append(f"<h2>{html.escape(_label_for(table))}</h2>")
        parts.append(
            f"<p>Family: {html.escape(str(table['family']))}; "
            f"section: {html.escape(str(table['section']))}; "
            f"units: {html.escape(str(table['units']) or 'not indicated')}</p>"
        )
        if item.get("reconstructed"):
            parts.append("<h3>Reconstructed bore profiles from probe measurements</h3>")
            parts.append(
                _img_html(
                    item["reconstructed"],
                    "Reconstructed Bore Profiles from Probe Measurements",
                )
            )
        if item.get("ideal_round"):
            parts.append("<h3>Ideal round profile via nonlinear regression</h3>")
            parts.append(
                _img_html(
                    item["ideal_round"],
                    "Ideal Round Profile via Nonlinear Regression",
                )
            )
        parts.append("<h3>Univariate summary</h3>")
        parts.append(_table_html(item["univariate"]))
        comparison_rows = item.get("model_comparison") or []
        if comparison_rows:
            parts.append("<h3>Model comparison</h3>")
            has_fifth = any(row["name"] == "5th-order polynomial" for row in comparison_rows)
            if item.get("model_basis") == "measured-equivalent":
                parts.append(
                    "<p>Each model is fit to the measured area-equivalent diameter "
                    "from the throat to the bell, including the 5th-order polynomial "
                    "shown above. R² values in this table use those same points.</p>"
                )
            elif has_fifth:
                parts.append(
                    "<p>Each model is fit to bore diameter against probe depth. "
                    "The 5th-order polynomial is included with the other curves.</p>"
                )
            else:
                parts.append(
                    "<p>Each model is fit to bore diameter against probe depth. "
                    "This section does not have enough stations for a 5th-order polynomial.</p>"
                )
            parts.append(_table_html(_model_comparison_frame(comparison_rows)))
        _append_shape_html(parts, item.get("shape"))
        parts.append("<h3>Strong correlations</h3>")
        pairs = item["correlations"]
        if pairs:
            parts.append("<ul>")
            for pair in pairs:
                parts.append(
                    f"<li>{html.escape(pair['left'])} and {html.escape(pair['right'])}: "
                    f"r = {pair['r']:.3f}</li>"
                )
            parts.append("</ul>")
        else:
            parts.append(f"<p>No pairs with |r| &gt;= {CORRELATION_THRESHOLD}.</p>")
        if item.get("kind") == "toneholes":
            parts.append("<h3>Tonehole placement</h3>")
            if item.get("plot") is not None:
                parts.append(_img_html(item["plot"], f"{_label_for(table)} toneholes"))
            else:
                parts.append("<p>No tonehole depth and diameter were available to plot.</p>")
            parts.append("<h3>Tonehole placement and bore diameter</h3>")
            links = item.get("bore_links")
            parts.append(_table_html(links if isinstance(links, pd.DataFrame) else pd.DataFrame()))
            link_pairs = item.get("link_correlations") or []
            if link_pairs:
                parts.append("<p>Strong relationships between tonehole placement and bore size:</p><ul>")
                for pair in link_pairs:
                    parts.append(
                        f"<li>{html.escape(pair['left'])} and {html.escape(pair['right'])}: "
                        f"r = {pair['r']:.3f}</li>"
                    )
                parts.append("</ul>")
            else:
                parts.append(
                    "<p>No strong correlation (|r| &gt;= "
                    f"{CORRELATION_THRESHOLD}) between tonehole placement and the nearest bore diameter.</p>"
                )
            continue
        fit = item["fit"]
        parts.append("<h3>Ideal round profile</h3>")
        if fit is None:
            parts.append("<p>Not enough paired depth and diameter values to fit a curve.</p>")
        else:
            parts.append(
                f"<p>Depth variable: {html.escape(fit['depth_label'])}. "
                f"n = {fit['n']}.</p>"
            )
            if fit.get("excluded_above_throat"):
                parts.append(
                    "<p>The linear and quadratic fits omit the bore above the throat, "
                    "the smallest measured diameter. The plotted points still show the full bore.</p>"
                )
            parts.append(
                "<p>The target is the area-equivalent diameter, the square root of the "
                "round-probe and oval-probe diameters at the same depth, so the curve "
                "lies between those two profiles. Linear is the straight-taper reference. "
                "Polynomials through degree 5, plus exponential, power, logarithmic, and "
                "spline curves, show where the bore leaves that line.</p><ul>"
            )
            for model in fit.get("models", []):
                r2 = "n/a" if model["r2"] is None else f"{model['r2']:.4f}"
                parts.append(
                    "<li>"
                    + html.escape(f"{model['detail']}; R² = {r2}")
                    + "</li>"
                )
            parts.append("</ul>")
        for figure in item.get("plots") or []:
            parts.append(f"<h3>{html.escape(str(figure['name']))}</h3>")
            parts.append(_img_html(figure["png"], f"{_label_for(table)} {figure['name']}"))
    parts.append("</body></html>")
    return "".join(parts)


def _render_pdf(
    path: Path,
    source: Path,
    context: str,
    objectives: list[str],
    analyses: list[dict[str, Any]],
    comparison: bytes | None,
    equivalent_overlay: bytes | None,
    reamer: dict[str, Any] | None,
    written_at: str,
    observations: list[str] | None = None,
) -> None:
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        text = [
            f"Bore EDA: {source.name}",
            f"Generated {written_at}",
            "",
            "Business context",
            context,
            "",
            *_observation_lines(observations),
            "Analytical objectives",
        ]
        text.extend(f"- {item}" for item in objectives)
        fig.text(0.08, 0.95, "\n".join(text), va="top", ha="left", fontsize=9, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)
        if comparison:
            _add_png_page(pdf, comparison, "Chanter and regulator bore profiles")
        if equivalent_overlay:
            _add_png_page(
                pdf,
                equivalent_overlay,
                "Reconstructed area-equivalent bore profiles",
            )
        if reamer and reamer.get("figure"):
            _add_png_page(
                pdf,
                reamer["figure"],
                "Area-equivalent and major-axis profiles",
            )
            note_lines = ["Reamer notes"]
            for section in reamer.get("sections") or []:
                if section.get("heading"):
                    note_lines.append(section["heading"])
                note_lines.extend(section.get("paragraphs") or [])
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(
                0.08,
                0.95,
                "\n".join(note_lines),
                va="top",
                ha="left",
                fontsize=8,
                wrap=True,
            )
            pdf.savefig(fig)
            plt.close(fig)
        for item in analyses:
            table = item["table"]
            fit = item["fit"]
            lines = [f"{_label_for(table)}", "Univariate summary:"]
            summary = item["univariate"]
            if not summary.empty:
                lines.append(summary.to_string(index=False))
            comparison_rows = item.get("model_comparison") or []
            if comparison_rows:
                lines.append("Model comparison:")
                if item.get("model_basis") == "measured-equivalent":
                    lines.append("  Target: measured area-equivalent diameter, throat to bell.")
                else:
                    lines.append("  Target: bore diameter against probe depth.")
                if not any(row["name"] == "5th-order polynomial" for row in comparison_rows):
                    lines.append("  Not enough stations for a 5th-order polynomial.")
                for row in comparison_rows:
                    r2 = "n/a" if row["r2"] is None else f"{row['r2']:.4f}"
                    lines.append(f"  {row['name']}: R² = {r2}; n = {row['n']}")
            shape = item.get("shape")
            if shape:
                lines.append("Derived bore features:")
                lines.extend(f"  {note}" for note in shape["notes"])
                lines.extend(f"  {note}" for note in shape["recommendations"])
            lines.append("Strong correlations:")
            if item["correlations"]:
                for pair in item["correlations"]:
                    lines.append(f"  {pair['left']} ~ {pair['right']}: r = {pair['r']:.3f}")
            else:
                lines.append(f"  none with |r| >= {CORRELATION_THRESHOLD}")
            if item.get("kind") == "toneholes":
                lines.append("Tonehole placement and nearest bore diameter:")
                links = item.get("bore_links")
                if links is not None and not links.empty:
                    lines.append(links.to_string(index=False))
                if item.get("link_correlations"):
                    for pair in item["link_correlations"]:
                        lines.append(f"  {pair['left']} ~ {pair['right']}: r = {pair['r']:.3f}")
            elif fit is None:
                lines.append("Ideal profile: not enough points.")
            else:
                lines.append(f"Depth: {fit['depth_label']}; n = {fit['n']}")
                if fit.get("excluded_above_throat"):
                    lines.append(
                        f"Excluded {fit['excluded_above_throat']} points above the throat."
                    )
                lines.append("Linear is the straight-taper reference.")
                for model in fit.get("models", []):
                    r2 = "n/a" if model["r2"] is None else f"{model['r2']:.4f}"
                    lines.append(f"{model['detail']}; R² = {r2}")
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.08, 0.95, "\n".join(lines), va="top", ha="left", fontsize=8, family="monospace")
            pdf.savefig(fig)
            plt.close(fig)
            if item.get("reconstructed"):
                _add_png_page(
                    pdf,
                    item["reconstructed"],
                    f"{_label_for(table)} — Reconstructed bore profiles",
                )
            if item.get("ideal_round"):
                _add_png_page(
                    pdf,
                    item["ideal_round"],
                    f"{_label_for(table)} — Ideal round profile via nonlinear regression",
                )
            shape = item.get("shape") or {}
            for figure in shape.get("figures") or []:
                _add_png_page(pdf, figure["png"], f"{_label_for(table)} — {figure['name']}")
            if item.get("kind") == "toneholes" and item.get("plot") is not None:
                _add_png_page(pdf, item["plot"], _label_for(table))
            for figure in item.get("plots") or []:
                _add_png_page(pdf, figure["png"], f"{_label_for(table)} — {figure['name']}")


def _add_png_page(pdf: PdfPages, png: bytes, title: str) -> None:
    image = plt.imread(BytesIO(png), format="png")
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title)
    pdf.savefig(fig)
    plt.close(fig)
