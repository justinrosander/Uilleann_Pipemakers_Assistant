"""Tonehole agent: sparse-data learning and three acoustic analyses.

The learned models are small networks trained hole by hole. They are
exploratory on a sparse set. Benade, Nederveen, and Lefebvre are kept
as separate physics calculations and are not mixed into the network loss.

Nothing is written to PostgreSQL.
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
from matplotlib.backends.backend_pdf import PdfPages

from agents.eda_execute import (
    _label_for,
    _numeric,
    _report_sort_key,
    _tonehole_depth_column,
    _tonehole_size,
    load_bore_tables,
    load_tonehole_tables,
    tonehole_bore_links,
)
from config import ROOT

OUTPUT_DIR = ROOT / "data" / "tonehole_model"
SOUND_SPEED_M_S = 346.2  # air at 23 C, as used with Benade's lattice examples
INCH_TO_M = 0.0254

_STYLE = {
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
}


def _acoustic_hole_diameter(frame: pd.DataFrame) -> pd.Series:
    """Area-equivalent tonehole diameter from the axial and transverse measurements.

    ch_diam is kept out of this value. On the collected sheets it is larger than
    the bore, while d-axial and d-xverse are the hole opening.
    """
    axial = _numeric(frame, "d-axial")
    transverse = _numeric(frame, "d-xverse")
    both = axial.notna() & transverse.notna() & (axial > 0) & (transverse > 0)
    diameter = pd.Series(np.nan, index=frame.index, dtype=float)
    diameter.loc[both] = np.sqrt(axial.loc[both].to_numpy(dtype=float) * transverse.loc[both].to_numpy(dtype=float))
    one_axis = diameter.isna() & axial.notna()
    diameter.loc[one_axis] = axial.loc[one_axis]
    one_transverse = diameter.isna() & transverse.notna()
    diameter.loc[one_transverse] = transverse.loc[one_transverse]
    if diameter.notna().any():
        return diameter
    return _tonehole_size(frame)


def _records_for_table(
    tonehole_table: dict[str, Any], bore_tables: list[dict[str, Any]]
) -> pd.DataFrame:
    """One row per tonehole, with the nearest bore diameter from the EDA link."""
    frame = tonehole_table["frame"]
    links, _pairs = tonehole_bore_links(tonehole_table, bore_tables)
    if links.empty:
        return pd.DataFrame()
    depth_column = _tonehole_depth_column(frame) or ""
    size = _acoustic_hole_diameter(frame)
    chimney = _numeric(frame, "chimney")
    position = pd.to_numeric(frame[depth_column], errors="coerce") if depth_column else pd.Series(np.nan, index=frame.index)
    rows = []
    for index, link in links.iterrows():
        rows.append(
            {
                "instrument": _label_for(tonehole_table),
                "keyhole": link.get("keyhole"),
                "position_in": float(position.iloc[index]) if index < len(position) and pd.notna(position.iloc[index]) else np.nan,
                "position_column": depth_column,
                "hole_diameter_in": float(size.iloc[index]) if index < len(size) and pd.notna(size.iloc[index]) else np.nan,
                "chimney_in": float(chimney.iloc[index]) if index < len(chimney) and pd.notna(chimney.iloc[index]) else np.nan,
                "bore_diameter_in": link.get("nearest_bore_diameter"),
            }
        )
    data = pd.DataFrame(rows)
    data["bore_diameter_in"] = pd.to_numeric(data["bore_diameter_in"], errors="coerce")
    return data


def _correlations(data: pd.DataFrame) -> list[dict[str, Any]]:
    columns = ["position_in", "hole_diameter_in", "chimney_in", "bore_diameter_in"]
    numeric = data[columns].apply(pd.to_numeric, errors="coerce")
    if numeric.dropna(how="any").shape[0] < 3:
        numeric = data[["position_in", "hole_diameter_in", "bore_diameter_in"]].apply(pd.to_numeric, errors="coerce")
    pairs = []
    corr = numeric.corr(method="pearson")
    names = list(corr.columns)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value):
                pairs.append({"left": left, "right": right, "r": float(value), "n": int(numeric[[left, right]].dropna().shape[0])})
    pairs.sort(key=lambda item: abs(item["r"]), reverse=True)
    return pairs


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    scale[scale < 1e-9] = 1.0
    return (train - center) / scale, (test - center) / scale, center, scale


def _fit_ann(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> float:
    """One-hidden-layer network. Returns the prediction in original units."""
    x_train_s, x_test_s, _center, _scale = _standardize(x_train, x_test)
    y_center = float(y_train.mean())
    y_scale = float(y_train.std()) or 1.0
    target = (y_train - y_center) / y_scale
    rng = np.random.default_rng(7)
    weights_1 = rng.normal(0, 0.3, size=(x_train_s.shape[1], 4))
    bias_1 = np.zeros(4)
    weights_2 = rng.normal(0, 0.3, size=4)
    bias_2 = 0.0
    for _step in range(600):
        hidden = np.tanh(x_train_s @ weights_1 + bias_1)
        residual = hidden @ weights_2 + bias_2 - target
        weights_2 -= 0.08 * (hidden.T @ residual / len(target) + 0.02 * weights_2)
        bias_2 -= 0.08 * float(residual.mean())
        back = (residual[:, None] * weights_2) * (1 - hidden**2)
        weights_1 -= 0.08 * (x_train_s.T @ back / len(target) + 0.02 * weights_1)
        bias_1 -= 0.08 * back.mean(axis=0)
    hidden_test = np.tanh(x_test_s @ weights_1 + bias_1)
    predicted = (hidden_test @ weights_2 + bias_2) * y_scale + y_center
    return float(np.asarray(predicted).reshape(-1)[0])


def _fit_mlp(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> float:
    """Two-hidden-layer perceptron, same training budget as the single-layer net."""
    x_train_s, x_test_s, _center, _scale = _standardize(x_train, x_test)
    y_center = float(y_train.mean())
    y_scale = float(y_train.std()) or 1.0
    target = (y_train - y_center) / y_scale
    rng = np.random.default_rng(11)
    weights = [
        rng.normal(0, 0.3, size=(x_train_s.shape[1], 6)),
        rng.normal(0, 0.3, size=(6, 4)),
    ]
    biases = [np.zeros(6), np.zeros(4)]
    weights_out = rng.normal(0, 0.3, size=4)
    bias_out = 0.0
    for _step in range(600):
        first = np.tanh(x_train_s @ weights[0] + biases[0])
        second = np.tanh(first @ weights[1] + biases[1])
        residual = second @ weights_out + bias_out - target
        weights_out -= 0.08 * (second.T @ residual / len(target) + 0.02 * weights_out)
        bias_out -= 0.08 * float(residual.mean())
        back_2 = (residual[:, None] * weights_out) * (1 - second**2)
        weights[1] -= 0.08 * (first.T @ back_2 / len(target) + 0.02 * weights[1])
        biases[1] -= 0.08 * back_2.mean(axis=0)
        back_1 = (back_2 @ weights[1].T) * (1 - first**2)
        weights[0] -= 0.08 * (x_train_s.T @ back_1 / len(target) + 0.02 * weights[0])
        biases[0] -= 0.08 * back_1.mean(axis=0)
    first = np.tanh(x_test_s @ weights[0] + biases[0])
    second = np.tanh(first @ weights[1] + biases[1])
    predicted = (second @ weights_out + bias_out) * y_scale + y_center
    return float(np.asarray(predicted).reshape(-1)[0])


def _fit_cnn(full_x: np.ndarray, full_y: np.ndarray, test_index: int) -> float:
    """1D convolution along the hole sequence. Neighbors stay visible; the held-out target does not."""
    order = np.argsort(full_x[:, 0])
    sequence = full_x[order]
    y_sorted = full_y[order]
    test_sorted = int(np.flatnonzero(order == test_index)[0])
    train_sorted = np.array([index for index in range(len(order)) if index != test_sorted])
    center = sequence[train_sorted].mean(axis=0)
    scale = sequence[train_sorted].std(axis=0)
    scale[scale < 1e-9] = 1.0
    norm = (sequence - center) / scale
    y_center = float(y_sorted[train_sorted].mean())
    y_scale = float(y_sorted[train_sorted].std()) or 1.0
    target = np.zeros(len(order))
    target[train_sorted] = (y_sorted[train_sorted] - y_center) / y_scale
    rng = np.random.default_rng(19)
    kernel = rng.normal(0, 0.2, size=(3, norm.shape[1], 2))
    bias = np.zeros(2)
    readout = rng.normal(0, 0.2, size=2)
    read_bias = 0.0
    padded = np.pad(norm, ((1, 1), (0, 0)), mode="edge")
    for _step in range(500):
        windows = np.stack([padded[i : i + 3] for i in range(len(norm))])
        hidden = np.tanh(np.einsum("nkf,kfc->nc", windows, kernel) + bias)
        predicted = hidden @ readout + read_bias
        residual = predicted[train_sorted] - target[train_sorted]
        readout -= 0.05 * (hidden[train_sorted].T @ residual / len(train_sorted) + 0.02 * readout)
        read_bias -= 0.05 * float(residual.mean())
        back = np.zeros_like(hidden)
        back[train_sorted] = residual[:, None] * readout * (1 - hidden[train_sorted] ** 2)
        grad = np.einsum("nkf,nc->kfc", windows, back) / len(train_sorted)
        kernel -= 0.05 * (grad + 0.02 * kernel)
        bias -= 0.05 * back[train_sorted].mean(axis=0)
    windows = np.stack([padded[i : i + 3] for i in range(len(norm))])
    hidden = np.tanh(np.einsum("nkf,kfc->nc", windows, kernel) + bias)
    predicted = hidden @ readout + read_bias
    return float(predicted[test_sorted] * y_scale + y_center)


def _leave_one_out(features: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    predictions = {name: np.full(len(target), np.nan) for name in ("ANN", "MLP", "CNN")}
    if len(target) < 5:
        return predictions
    for index in range(len(target)):
        train = np.ones(len(target), dtype=bool)
        train[index] = False
        predictions["ANN"][index] = _fit_ann(features[train], target[train], features[index : index + 1])
        predictions["MLP"][index] = _fit_mlp(features[train], target[train], features[index : index + 1])
        predictions["CNN"][index] = _fit_cnn(features, target, index)
    return predictions


def _scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float] | None:
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if int(mask.sum()) < 3:
        return None
    error = predicted[mask] - actual[mask]
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((actual[mask] - actual[mask].mean()) ** 2))
    return {
        "n": float(mask.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def _benade_cutoff(spacing_m: float, bore_radius_m: float, hole_radius_m: float, effective_height_m: float) -> float:
    """Open-lattice cutoff. 2*ell is the hole spacing. Benade's t_e is the effective height.

    Moers and Kergomard (2011), eq. (6), the low-frequency limit of Benade's lattice:
    f_c = (c / 2π) (b / a) / sqrt(2 ell h_t).
    """
    if min(spacing_m, bore_radius_m, hole_radius_m, effective_height_m) <= 0:
        return np.nan
    ell = spacing_m / 2.0
    return float(
        (SOUND_SPEED_M_S / (2 * np.pi))
        * (hole_radius_m / bore_radius_m)
        / np.sqrt(2 * ell * effective_height_m)
    )


def _nederveen(hole_radius_m: float, bore_radius_m: float, chimney_m: float) -> dict[str, float]:
    """Open-hole length corrections from Nederveen and from Nederveen, Jansen, and van Hassel 1998.

    Matching volume t_m = (b δ / 8) (1 + 0.207 δ^3).
    Inner correction t_i = (0.82 - 1.4 δ^2 + 0.75 δ^2.7) b.
    Radiation correction for an unflanged open end, 0.6133 b.
    The inner correction was obtained for tall chimneys, t > b.
    """
    delta = hole_radius_m / bore_radius_m
    if not 0 < delta <= 1.0:
        return {
            "delta": float(delta),
            "matching_m": np.nan,
            "inner_m": np.nan,
            "radiation_m": np.nan,
            "effective_m": np.nan,
            "series_m": np.nan,
            "tall_chimney": bool(chimney_m >= hole_radius_m),
        }
    matching = (hole_radius_m * delta / 8.0) * (1.0 + 0.207 * delta**3)
    inner = (0.82 - 1.4 * delta**2 + 0.75 * delta**2.7) * hole_radius_m
    radiation = 0.6133 * hole_radius_m
    series = -0.28 * hole_radius_m * delta**4
    return {
        "delta": float(delta),
        "matching_m": float(matching),
        "inner_m": float(inner),
        "radiation_m": float(radiation),
        "effective_m": float(chimney_m + matching + inner + radiation),
        "series_m": float(series),
        "tall_chimney": bool(chimney_m >= hole_radius_m),
    }


def _lefebvre(hole_radius_m: float, bore_radius_m: float, chimney_m: float) -> dict[str, float]:
    """Lefebvre and Scavone (2010) low-frequency fits for an unflanged open tonehole.

    Shunt length is their equation (34). Series length is their equation (37).
    The fit was built for δ from 0.2 to 1, with the thin-wall and small-hole limits added.
    """
    delta = hole_radius_m / bore_radius_m
    if not 0.2 <= delta <= 1.0:
        return {"effective_m": np.nan, "series_m": np.nan, "delta_in_fit_range": False}
    ratio = chimney_m / hole_radius_m if hole_radius_m else np.nan
    f_delta = (
        0.095
        - 0.422 * delta
        + 1.168 * delta**2
        - 1.808 * delta**3
        + 1.398 * delta**4
        - 0.416 * delta**5
    )
    g_height = 1.0 - np.tanh(0.778 * ratio)
    h_delta = (
        1.435
        + 0.030 * delta
        - 1.566 * delta**2
        + 2.138 * delta**3
        - 1.614 * delta**4
        + 0.502 * delta**5
    )
    effective = chimney_m + hole_radius_m * (1.0 + f_delta * g_height) * h_delta
    f_series = 1.0 + (0.333 - 0.138 * delta) * (1.0 - np.tanh(2.666 * ratio))
    g_series = 0.307 - 0.022 * delta - 0.002 * delta**2
    series = -f_series * g_series * hole_radius_m * delta**4
    return {
        "effective_m": float(effective),
        "series_m": float(series),
        "delta_in_fit_range": bool(0.2 <= delta <= 1.0),
    }


def _physics_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    holes = data.dropna(subset=["position_in", "hole_diameter_in", "bore_diameter_in", "chimney_in"]).copy()
    holes = holes.sort_values("position_in").reset_index(drop=True)
    hole_rows = []
    for _, row in holes.iterrows():
        bore_radius = float(row["bore_diameter_in"]) / 2.0 * INCH_TO_M
        hole_radius = float(row["hole_diameter_in"]) / 2.0 * INCH_TO_M
        chimney = float(row["chimney_in"]) * INCH_TO_M
        if min(bore_radius, hole_radius, chimney) <= 0:
            continue
        nederveen = _nederveen(hole_radius, bore_radius, chimney)
        lefebvre = _lefebvre(hole_radius, bore_radius, chimney)
        benade_height = chimney + 1.5 * hole_radius
        hole_rows.append(
            {
                "keyhole": row["keyhole"],
                "position_in": row["position_in"],
                "bore_diameter_in": row["bore_diameter_in"],
                "hole_diameter_in": row["hole_diameter_in"],
                "chimney_in": row["chimney_in"],
                "delta": nederveen["delta"],
                "benade_te_in": benade_height / INCH_TO_M,
                "nederveen_te_in": nederveen["effective_m"] / INCH_TO_M,
                "lefebvre_te_in": lefebvre["effective_m"] / INCH_TO_M,
                "nederveen_series_in": nederveen["series_m"] / INCH_TO_M,
                "lefebvre_series_in": lefebvre["series_m"] / INCH_TO_M,
                "nederveen_tall": nederveen["tall_chimney"],
                "lefebvre_fit_range": lefebvre["delta_in_fit_range"],
            }
        )
    holes_out = pd.DataFrame(hole_rows)
    cells = []
    if len(holes_out) >= 2:
        for index in range(len(holes_out) - 1):
            left = holes_out.iloc[index]
            right = holes_out.iloc[index + 1]
            spacing_m = abs(float(right["position_in"]) - float(left["position_in"])) * INCH_TO_M
            bore_radius = 0.5 * (left["bore_diameter_in"] + right["bore_diameter_in"]) / 2.0 * INCH_TO_M
            hole_radius = 0.5 * (left["hole_diameter_in"] + right["hole_diameter_in"]) / 2.0 * INCH_TO_M
            height = 0.5 * (left["benade_te_in"] + right["benade_te_in"]) * INCH_TO_M
            cells.append(
                {
                    "from_hole": left["keyhole"],
                    "to_hole": right["keyhole"],
                    "spacing_in": spacing_m / INCH_TO_M,
                    "cutoff_hz": _benade_cutoff(spacing_m, bore_radius, hole_radius, height),
                }
            )
    return holes_out, pd.DataFrame(cells)


def _bore_half_angle_deg(bore_tables: list[dict[str, Any]], instrument: str) -> float | None:
    from agents.eda_execute import _depth_series

    selected = [table for table in bore_tables if _label_for(table) == instrument or table.get("title") == instrument]
    if not selected:
        return None
    table = selected[0]
    depth, _label = _depth_series(table["frame"])
    diameter = _numeric(table["frame"], "Diameter").to_numpy(dtype=float)
    mask = np.isfinite(depth) & np.isfinite(diameter)
    if int(mask.sum()) < 4:
        return None
    slope = float(np.polyfit(depth[mask], diameter[mask] / 2.0, 1)[0])
    return float(np.degrees(np.arctan(abs(slope))))


def _prediction_figure(data: pd.DataFrame, predictions: dict[str, np.ndarray]) -> bytes:
    actual = data["hole_diameter_in"].to_numpy(dtype=float)
    buffer = BytesIO()
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
        for ax, name, color in zip(axes, ("ANN", "MLP", "CNN"), ("#1a5276", "#6c3483", "#b9770e")):
            predicted = predictions[name]
            ax.scatter(actual, predicted, s=28, color=color)
            finite = actual[np.isfinite(predicted)]
            if len(finite):
                limits = [float(np.nanmin(actual)), float(np.nanmax(actual))]
                ax.plot(limits, limits, color="#888888", linewidth=1)
            ax.set_title(name)
            ax.set_xlabel("Measured hole diameter (in)")
            ax.grid(True, color="#e6e6e6", linewidth=0.6)
        axes[0].set_ylabel("Leave-one-out prediction (in)")
        fig.tight_layout()
        fig.savefig(buffer, format="png", dpi=140)
        plt.close(fig)
    return buffer.getvalue()


def _physics_figure(holes: pd.DataFrame, cells: pd.DataFrame) -> bytes | None:
    if holes.empty:
        return None
    buffer = BytesIO()
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        position = holes["position_in"].to_numpy(dtype=float)
        axes[0].plot(position, holes["nederveen_te_in"], marker="o", color="#1a5276", label="Nederveen")
        axes[0].plot(position, holes["lefebvre_te_in"], marker="s", color="#b9770e", label="Lefebvre")
        axes[0].plot(position, holes["benade_te_in"], marker="^", color="#1e8449", label="Benade t + 1.5b")
        axes[0].set_xlabel(str(holes.attrs.get("position_column", "Position (in)")))
        axes[0].set_ylabel("Effective chimney length (in)")
        axes[0].set_title("Effective open-hole length")
        axes[0].legend(loc="best", fontsize=7)
        axes[0].grid(True, color="#e6e6e6", linewidth=0.6)
        if not cells.empty:
            labels = [f"{row['from_hole']}–{row['to_hole']}" for _, row in cells.iterrows()]
            axes[1].bar(range(len(cells)), cells["cutoff_hz"], color="#6c3483")
            axes[1].set_xticks(range(len(cells)), labels, rotation=30, ha="right", fontsize=7)
            axes[1].set_ylabel("Cutoff (Hz)")
        axes[1].set_title("Benade local lattice cutoff")
        axes[1].grid(True, axis="y", color="#e6e6e6", linewidth=0.6)
        fig.tight_layout()
        fig.savefig(buffer, format="png", dpi=140)
        plt.close(fig)
    return buffer.getvalue()


def _table_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>No rows.</p>"
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in shown.columns)
    body = []
    for _, row in shown.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _img(png: bytes, alt: str) -> str:
    encoded = base64.b64encode(png).decode("ascii")
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{encoded}"/>'


def analyze_instrument(
    tonehole_table: dict[str, Any], bore_tables: list[dict[str, Any]]
) -> dict[str, Any] | None:
    data = _records_for_table(tonehole_table, bore_tables)
    if data.empty:
        return None
    data = data.dropna(subset=["position_in", "hole_diameter_in", "bore_diameter_in"]).reset_index(drop=True)
    if data.empty:
        return None
    features = data[["position_in", "bore_diameter_in", "chimney_in"]].to_numpy(dtype=float)
    chimney_known = np.isfinite(features[:, 2])
    if chimney_known.any():
        fill = float(np.nanmedian(features[chimney_known, 2]))
        features[~chimney_known, 2] = fill
    else:
        features[:, 2] = 0.0
    target = data["hole_diameter_in"].to_numpy(dtype=float)
    predictions = _leave_one_out(features, target)
    score_rows = []
    for name, predicted in predictions.items():
        score = _scores(target, predicted)
        if score:
            score_rows.append({"model": name, **score})
    holes, cells = _physics_tables(data)
    holes.attrs["position_column"] = data["position_column"].iloc[0] if len(data) else "Position (in)"
    return {
        "label": _label_for(tonehole_table),
        "data": data,
        "correlations": _correlations(data),
        "predictions": predictions,
        "scores": pd.DataFrame(score_rows),
        "holes": holes,
        "cells": cells,
        "half_angle_deg": _bore_half_angle_deg(bore_tables, str(tonehole_table.get("title"))),
        "prediction_png": _prediction_figure(data, predictions) if score_rows else None,
        "physics_png": _physics_figure(holes, cells),
        "n": len(data),
    }


def _summary_lines(result: dict[str, Any]) -> list[str]:
    lines = [
        f"{result['label']}: {result['n']} toneholes linked to a bore diameter.",
        "The learned models predict tonehole diameter from position, nearest bore diameter, and chimney height. Each prediction is leave-one-out, so the reported error is not the training error.",
    ]
    if result["scores"].empty:
        lines.append("Fewer than five complete holes, so the networks were not trained.")
    else:
        for _, row in result["scores"].iterrows():
            r2 = "n/a" if not np.isfinite(row["r2"]) else f"{row['r2']:.3f}"
            lines.append(
                f"{row['model']}: leave-one-out MAE {row['mae']:.4f} in, RMSE {row['rmse']:.4f} in, R² {r2}."
            )
        best = result["scores"].sort_values("mae").iloc[0]
        lines.append(
            f"On this sparse sample the lowest leave-one-out MAE is the {best['model']} model, "
            f"{best['mae']:.4f} in. Hole size on one instrument changes smoothly along the bore, "
            "so a high score here is that trend, not a design law trained across sets."
        )
    if result["correlations"]:
        strongest = result["correlations"][0]
        lines.append(
            f"Strongest measured relationship: {strongest['left']} and {strongest['right']}, r = {strongest['r']:.3f} (n = {strongest['n']})."
        )
    if result["half_angle_deg"] is not None:
        lines.append(
            f"The matched bore half-angle is about {result['half_angle_deg']:.2f} degrees. "
            "Lefebvre and Scavone found the cylindrical tonehole corrections unchanged on cones up to 6 degrees."
        )
    if not result["cells"].empty:
        cutoffs = result["cells"]["cutoff_hz"].to_numpy(dtype=float)
        lines.append(
            f"Benade local cutoffs run from {np.nanmin(cutoffs):.0f} Hz to {np.nanmax(cutoffs):.0f} Hz. "
            "A regular lattice would keep those values close together. Speed of sound is 346.2 m/s."
        )
    if not result["holes"].empty:
        short = int((~result["holes"]["nederveen_tall"]).sum())
        outside = int((~result["holes"]["lefebvre_fit_range"]).sum())
        lines.append(
            f"Nederveen's inner correction assumes a tall chimney (t at least b). {short} holes are shorter than that. "
            f"Lefebvre's fit range is a radius ratio from 0.2 to 1. {outside} holes fall outside it."
        )
    lines.append(
        "Use the learned diameter only as a check against the measured holes. "
        "Use Lefebvre's effective length when the chimney is short, and Nederveen's split into matching volume, inner correction, and radiation when the chimney is tall. "
        "Benade's cutoff is a property of the open-hole row, not of one hole."
    )
    return lines


def _html_report(path: Path, source_name: str, results: list[dict[str, Any]], written_at: str) -> None:
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>",
        f"<title>Tonehole model {html.escape(source_name)}</title>",
        "<style>body{font-family:Georgia,serif;margin:2rem;color:#222;max-width:1000px;}"
        "table{border-collapse:collapse;margin:1rem 0;}td,th{border:1px solid #ccc;padding:0.3rem 0.5rem;}"
        "img{max-width:100%;height:auto;}h1,h2{font-weight:normal;}</style></head><body>",
        f"<h1>Tonehole placement and dimensions: {html.escape(source_name)}</h1>",
        f"<p>Generated {html.escape(written_at)}. Sparse-data models and three separate acoustic analyses. Nothing is written to the database.</p>",
        "<p>The networks are an ANN with one hidden layer, an MLP with two hidden layers, and a 1D CNN along the hole sequence. "
        "Benade's analysis is the open-hole lattice cutoff. Nederveen's analysis splits the chimney correction into matching volume, inner length, and radiation. "
        "Lefebvre and Scavone's analysis is their 2010 low-frequency fit for short unflanged holes.</p>",
    ]
    for result in results:
        parts.append(f"<h2>{html.escape(result['label'])}</h2>")
        for line in _summary_lines(result):
            parts.append(f"<p>{html.escape(line)}</p>")
        parts.append("<h3>Measured holes and nearest bore</h3>")
        parts.append(_table_html(result["data"]))
        parts.append("<h3>Relationships</h3>")
        parts.append(_table_html(pd.DataFrame(result["correlations"])))
        parts.append("<h3>Leave-one-out model scores</h3>")
        parts.append(_table_html(result["scores"]))
        if result.get("prediction_png"):
            parts.append(_img(result["prediction_png"], f"{result['label']} model predictions"))
        parts.append("<h3>Benade lattice</h3>")
        parts.append("<p>Local cutoff between successive holes, using t<sub>e</sub> = t + 1.5b and the low-frequency lattice formula of Moers and Kergomard (2011), equation (6).</p>")
        parts.append(_table_html(result["cells"]))
        parts.append("<h3>Nederveen and Lefebvre</h3>")
        show = result["holes"].drop(columns=[column for column in ("nederveen_tall", "lefebvre_fit_range") if column in result["holes"].columns], errors="ignore")
        parts.append(_table_html(show))
        if result.get("physics_png"):
            parts.append(_img(result["physics_png"], f"{result['label']} acoustic lengths and cutoff"))
    parts.append("</body></html>")
    path.write_text("".join(parts), encoding="utf-8")


def _pdf_report(path: Path, source_name: str, results: list[dict[str, Any]], written_at: str) -> None:
    with PdfPages(path) as pdf:
        for result in results:
            lines = [f"Tonehole model: {source_name}", written_at, result["label"], ""]
            lines.extend(_summary_lines(result))
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.08, 0.95, "\n".join(lines), va="top", ha="left", fontsize=9, family="DejaVu Sans")
            pdf.savefig(fig)
            plt.close(fig)
            for key, title in (
                ("prediction_png", "Leave-one-out predictions"),
                ("physics_png", "Acoustic lengths and Benade cutoff"),
            ):
                if not result.get(key):
                    continue
                image = plt.imread(BytesIO(result[key]), format="png")
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.imshow(image)
                ax.axis("off")
                ax.set_title(f"{result['label']} — {title}")
                pdf.savefig(fig)
                plt.close(fig)


def run_tonehole_agent(
    tonehole_tables: list[dict[str, Any]],
    bore_tables: list[dict[str, Any]],
    source_name: str,
    destination: Path,
    formats: set[str],
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    ordered = sorted(tonehole_tables, key=_report_sort_key)
    results = []
    for table in ordered:
        result = analyze_instrument(table, bore_tables)
        if result is not None:
            results.append(result)
    if not results:
        raise ValueError(f"No toneholes with a matched bore diameter were found for {source_name}.")
    written_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stem = source_name.replace(" ", "_").replace("'", "")
    paths: list[Path] = []
    if "html" in formats:
        html_path = destination / f"{stem}_toneholes.html"
        _html_report(html_path, source_name, results, written_at)
        paths.append(html_path)
    if "pdf" in formats:
        pdf_path = destination / f"{stem}_toneholes.pdf"
        _pdf_report(pdf_path, source_name, results, written_at)
        paths.append(pdf_path)
    return paths


def tonehole_agent_source(source: Path, formats: set[str], output_dir: Path | None = None) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Tonehole agent source not found: {source}")
    bore_tables = [table for table in load_bore_tables(source) if table.get("kind") == "bore"]
    tonehole_tables = load_tonehole_tables(source)
    destination = output_dir or (OUTPUT_DIR / source.stem)
    return run_tonehole_agent(tonehole_tables, bore_tables, source.stem, destination, formats)


def tonehole_agent_set(set_name: str, formats: set[str], output_dir: Path | None = None) -> list[Path]:
    from db.sets import load_set_bore_tables

    set_row, tables = load_set_bore_tables(set_name)
    bore_tables = [table for table in tables if table.get("kind") == "bore"]
    tonehole_tables = [table for table in tables if table.get("kind") == "toneholes"]
    label = str(set_row["name"]).replace(" ", "_").replace("'", "")
    destination = output_dir or (OUTPUT_DIR / label)
    return run_tonehole_agent(tonehole_tables, bore_tables, str(set_row["name"]), destination, formats)
