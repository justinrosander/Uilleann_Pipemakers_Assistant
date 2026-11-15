"""Derived bore-shape EDA: ovality, taper, cone residuals, and multivariate checks.

These steps come from the sections added to ML EDA Problem Definition.md.
Axial position is distance from the top when an overall length is known,
and probe depth otherwise. Diagnostic plots put that position on the
horizontal axis. The reconstructed profile figures keep diameter horizontal.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2, whiten
from scipy.stats import gaussian_kde, ttest_1samp

from agents.eda_execute import _depth_series, _numeric, _probe_axis_profiles

_STYLE = {
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
}

_LABELS = {
    "axial_position": "Axial position (in)",
    "minor_diameter": "Minor diameter (in)",
    "major_diameter": "Major diameter (in)",
    "equivalent_diameter": "Area-equivalent diameter (in)",
    "diameter": "Bore diameter (in)",
    "absolute_ovality": "Absolute ovality (in)",
    "relative_ovality": "Relative ovality",
    "taper_rate": "Local taper rate (in/in)",
    "cone_residual": "Linear-cone residual (in)",
    "major_minus_equivalent": "Major minus area-equivalent (in)",
}


def _png(fig) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=130)
    plt.close(fig)
    return buffer.getvalue()


def _finite_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return numeric.reset_index(drop=True)


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column].dropna()
        if series.empty:
            continue
        skew = float(series.skew()) if len(series) >= 3 else np.nan
        kurtosis = float(series.kurtosis()) if len(series) >= 4 else np.nan
        rows.append(
            {
                "variable": _LABELS.get(column, column),
                "count": int(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std(ddof=1)) if len(series) > 1 else np.nan,
                "min": float(series.min()),
                "max": float(series.max()),
                "skewness": skew,
                "excess_kurtosis": kurtosis,
            }
        )
    return pd.DataFrame(rows)


def _smooth(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 4:
        return None
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    degree = 3 if len(x) >= 8 else 1
    coefficients = np.polyfit(x, y, degree)
    grid = np.linspace(float(x.min()), float(x.max()), 120)
    return grid, np.polyval(coefficients, grid)


def _hist_kde(ax, values: np.ndarray, color: str, title: str, xlabel: str) -> None:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        ax.set_axis_off()
        return
    bins = min(24, max(8, int(np.sqrt(len(values)))))
    ax.hist(values, bins=bins, density=True, color=color, alpha=0.35, edgecolor="white")
    if len(values) >= 5 and float(np.std(values)) > 0:
        kde = gaussian_kde(values)
        grid = np.linspace(float(values.min()), float(values.max()), 200)
        ax.plot(grid, kde(grid), color=color, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, color="#e6e6e6", linewidth=0.5)


def _dual_features(frame: pd.DataFrame, oal: float | None) -> pd.DataFrame | None:
    profiles = _probe_axis_profiles(frame, oal)
    if profiles is None:
        return None
    lo = max(float(profiles["minor_distance"].min()), float(profiles["major_distance"].min()))
    hi = min(float(profiles["minor_distance"].max()), float(profiles["major_distance"].max()))
    if hi <= lo:
        return None
    grid = np.linspace(lo, hi, num=max(40, int(profiles["n_pairs"])))
    minor = np.interp(grid, profiles["minor_distance"], profiles["minor_diameter"])
    major = np.interp(grid, profiles["major_distance"], profiles["major_diameter"])
    equivalent = np.sqrt(np.clip(minor, 0, None) * np.clip(major, 0, None))
    safe_minor = np.where(minor > 1e-9, minor, np.nan)
    taper = np.gradient(equivalent, grid)
    slope, intercept = np.polyfit(grid, equivalent, 1)
    residual = equivalent - (slope * grid + intercept)
    features = pd.DataFrame(
        {
            "axial_position": grid,
            "minor_diameter": minor,
            "major_diameter": major,
            "equivalent_diameter": equivalent,
            "absolute_ovality": major - minor,
            "relative_ovality": major / safe_minor,
            "taper_rate": taper,
            "cone_residual": residual,
        }
    )
    return _finite_frame(features)


def _single_features(frame: pd.DataFrame, oal: float | None) -> tuple[pd.DataFrame, str] | None:
    diameter = _numeric(frame, "Diameter").to_numpy(dtype=float)
    depth, _depth_label = _depth_series(frame)
    mask = np.isfinite(diameter) & np.isfinite(depth)
    if int(mask.sum()) < 4:
        return None
    depth = depth[mask]
    diameter = diameter[mask]
    if oal is not None and np.isfinite(oal):
        position = float(oal) - depth
        label = "Distance from top (in)"
    else:
        position = depth
        label = "Probe depth (in)"
    order = np.argsort(position)
    position = position[order]
    diameter = diameter[order]
    rounded = np.round(position, 6)
    unique, inverse = np.unique(rounded, return_inverse=True)
    diameter = np.array([diameter[inverse == index].mean() for index in range(len(unique))])
    if len(unique) < 4:
        return None
    taper = np.gradient(diameter, unique)
    slope, intercept = np.polyfit(unique, diameter, 1)
    residual = diameter - (slope * unique + intercept)
    features = _finite_frame(
        pd.DataFrame(
            {
                "axial_position": unique,
                "diameter": diameter,
                "taper_rate": taper,
                "cone_residual": residual,
            }
        )
    )
    if len(features) < 4:
        return None
    return features, label


def _probe_depth_gap(frame: pd.DataFrame) -> dict[str, float] | None:
    round_depth = _numeric(frame, "Round").to_numpy(dtype=float)
    oval_depth = _numeric(frame, "Oval").to_numpy(dtype=float)
    both = np.isfinite(round_depth) & np.isfinite(oval_depth)
    if int(both.sum()) < 4:
        return None
    gap = np.abs(round_depth[both] - oval_depth[both])
    correlation = float(np.corrcoef(round_depth[both], oval_depth[both])[0, 1])
    return {
        "n": float(both.sum()),
        "mean_abs_gap": float(np.mean(gap)),
        "median_abs_gap": float(np.median(gap)),
        "correlation": correlation,
    }


def _region_means(features: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in features or len(features) < 6:
        return None
    labels = pd.qcut(
        features["axial_position"],
        3,
        labels=["near the top", "middle", "toward the bell"],
        duplicates="drop",
    )
    grouped = features.groupby(labels, observed=True)[column].mean()
    if grouped.empty:
        return None
    return grouped


def _outliers(features: pd.DataFrame, column: str) -> list[str]:
    series = features[column]
    deviation = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    if deviation == 0 or not np.isfinite(deviation):
        return []
    score = (series - series.mean()).abs() / deviation
    flagged = features.loc[score > 3, ["axial_position", column]]
    notes = []
    for _, row in flagged.head(6).iterrows():
        notes.append(
            f"{_LABELS.get(column, column)} {row[column]:.4g} at "
            f"{row['axial_position']:.3g} in"
        )
    if len(flagged) > 6:
        notes.append(f"{len(flagged) - 6} more beyond 3 standard deviations")
    return notes


def _area_table(features: pd.DataFrame) -> pd.DataFrame:
    minor = features["minor_diameter"].to_numpy(dtype=float)
    major = features["major_diameter"].to_numpy(dtype=float)
    versus_minor = (major / minor - 1.0) * 100.0
    versus_major = (minor / major - 1.0) * 100.0
    return pd.DataFrame(
        [
            {
                "comparison": "Equivalent circle versus minor-axis circle",
                "mean_area_change_percent": float(np.mean(versus_minor)),
                "median_area_change_percent": float(np.median(versus_minor)),
            },
            {
                "comparison": "Equivalent circle versus major-axis circle",
                "mean_area_change_percent": float(np.mean(versus_major)),
                "median_area_change_percent": float(np.median(versus_major)),
            },
        ]
    )


def _taper_choice(features: pd.DataFrame) -> pd.DataFrame:
    position = features["axial_position"].to_numpy(dtype=float)
    rows = []
    reference = float(np.polyfit(position, features["equivalent_diameter"], 1)[0])
    for column, name in (
        ("minor_diameter", "Minor axis"),
        ("major_diameter", "Major axis"),
        ("equivalent_diameter", "Area-equivalent"),
    ):
        slope = float(np.polyfit(position, features[column], 1)[0])
        gap = float(np.mean(np.abs(features[column] - features["equivalent_diameter"])))
        rows.append(
            {
                "profile": name,
                "taper_in_per_in": slope,
                "taper_change_from_equivalent": slope - reference,
                "mean_abs_gap_from_equivalent_in": gap,
            }
        )
    return pd.DataFrame(rows)


def _pca(features: pd.DataFrame) -> dict[str, Any] | None:
    columns = [
        column
        for column in features.columns
        if float(features[column].std(ddof=1) or 0) > 0
    ]
    if len(columns) < 2 or len(features) < 5:
        return None
    matrix = features[columns].to_numpy(dtype=float)
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=1)
    scale[scale == 0] = 1.0
    standardized = (matrix - center) / scale
    _u, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    explained = singular**2
    explained = explained / explained.sum()
    scores = standardized @ vt[:2].T
    loadings = pd.DataFrame(vt[:2].T, index=columns, columns=["PC1", "PC2"])
    return {
        "explained": explained,
        "scores": scores,
        "position": features["axial_position"].to_numpy(dtype=float),
        "loadings": loadings,
    }


def _segments(features: pd.DataFrame) -> pd.Series | None:
    columns = [
        column
        for column in ("taper_rate", "absolute_ovality")
        if column in features and float(features[column].std(ddof=1) or 0) > 0
    ]
    if not columns or len(features) < 9:
        return None
    data = features[columns].to_numpy(dtype=float)
    try:
        _centroids, labels = kmeans2(whiten(data), 3, minit="points", seed=7, iter=20)
    except Exception:
        return None
    labeled = pd.Series(labels, index=features.index, name="segment")
    rank_column = "absolute_ovality" if "absolute_ovality" in features else "taper_rate"
    order = (
        features.assign(segment=labeled)
        .groupby("segment")[rank_column]
        .mean()
        .sort_values()
    )
    if rank_column == "absolute_ovality":
        names = ["lower ovality", "moderate ovality", "higher ovality"]
    else:
        names = ["shallower taper", "moderate taper", "steeper taper"]
    mapping = {int(segment): names[index] for index, segment in enumerate(order.index)}
    return labeled.map(mapping)


def _distribution_figure(features: pd.DataFrame, dual: bool) -> bytes:
    panels = (
        [
            ("minor_diameter", "#2471A3", "Minor diameter"),
            ("major_diameter", "#1E8449", "Major diameter"),
            ("equivalent_diameter", "#6C3483", "Area-equivalent diameter"),
            ("absolute_ovality", "#B9770E", "Absolute ovality"),
            ("taper_rate", "#1A5276", "Local taper rate"),
        ]
        if dual
        else [
            ("diameter", "#2471A3", "Bore diameter"),
            ("taper_rate", "#1A5276", "Local taper rate"),
            ("cone_residual", "#C0392B", "Linear-cone residual"),
        ]
    )
    columns = 2
    rows = int(np.ceil(len(panels) / columns))
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(rows, columns, figsize=(10, 3.2 * rows))
        flat = np.atleast_1d(axes).ravel()
        for ax, (column, color, title) in zip(flat, panels):
            _hist_kde(ax, features[column].to_numpy(dtype=float), color, title, _LABELS[column])
        for ax in flat[len(panels) :]:
            ax.set_axis_off()
        fig.suptitle("Distributions of bore features", fontsize=14)
        fig.tight_layout()
        return _png(fig)


def _diameter_comparison_figure(features: pd.DataFrame) -> bytes:
    columns = ["minor_diameter", "major_diameter", "equivalent_diameter"]
    labels = ["Minor", "Major", "Area-equivalent"]
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.violinplot(
            [features[column].to_numpy(dtype=float) for column in columns],
            showmedians=True,
        )
        ax.set_xticks([1, 2, 3], labels)
        ax.set_ylabel("Diameter (in)")
        ax.set_title("Minor, major, and area-equivalent diameters")
        ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.5)
        fig.tight_layout()
        return _png(fig)


def _ovality_figure(features: pd.DataFrame) -> bytes:
    ovality = features["absolute_ovality"].to_numpy(dtype=float)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        _hist_kde(axes[0], ovality, "#B9770E", "Absolute ovality", "Major − minor (in)")
        ordered = np.sort(ovality)
        axes[1].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), color="#6C3483")
        axes[1].set_title("Cumulative distribution of ovality")
        axes[1].set_xlabel("Major − minor (in)")
        axes[1].set_ylabel("Fraction of stations")
        axes[1].grid(True, color="#e6e6e6", linewidth=0.5)
        fig.tight_layout()
        return _png(fig)


def _scatter_smooth(ax, x, y, color: str, label: str | None = None) -> None:
    ax.scatter(x, y, s=12, color=color, alpha=0.75, label=label)
    smoothed = _smooth(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if smoothed is not None:
        ax.plot(smoothed[0], smoothed[1], color="#1a1a1a", linewidth=1.3, label="Polynomial smooth")
    ax.grid(True, color="#e6e6e6", linewidth=0.5)


def _bivariate_figure(features: pd.DataFrame, position_label: str, dual: bool) -> bytes:
    position = features["axial_position"].to_numpy(dtype=float)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
        if dual:
            for column, color, name in (
                ("minor_diameter", "#2471A3", "Minor"),
                ("major_diameter", "#1E8449", "Major"),
                ("equivalent_diameter", "#6C3483", "Area-equivalent"),
            ):
                axes[0, 0].plot(position, features[column], color=color, linewidth=1.4, label=name)
            axes[0, 0].legend(loc="best", fontsize=7)
            axes[0, 0].set_ylabel("Diameter (in)")
        else:
            _scatter_smooth(axes[0, 0], position, features["diameter"], "#2471A3")
            axes[0, 0].set_ylabel("Diameter (in)")
        axes[0, 0].set_title("Diameter and axial position")
        axes[0, 0].set_xlabel(position_label)
        axes[0, 0].grid(True, color="#e6e6e6", linewidth=0.5)
        if dual:
            _scatter_smooth(axes[0, 1], position, features["absolute_ovality"], "#B9770E")
            axes[0, 1].set_title("Ovality and axial position")
            axes[0, 1].set_ylabel("Major − minor (in)")
        else:
            _scatter_smooth(axes[0, 1], position, features["cone_residual"], "#C0392B")
            axes[0, 1].set_title("Linear-cone residual")
            axes[0, 1].axhline(0, color="#888888", linewidth=0.8)
            axes[0, 1].set_ylabel("Residual (in)")
        axes[0, 1].set_xlabel(position_label)
        _scatter_smooth(axes[1, 0], position, features["taper_rate"], "#1A5276")
        axes[1, 0].set_title("Local taper rate")
        axes[1, 0].set_xlabel(position_label)
        axes[1, 0].set_ylabel("Change in diameter per inch")
        if dual:
            _scatter_smooth(axes[1, 1], position, features["cone_residual"], "#C0392B")
            axes[1, 1].axhline(0, color="#888888", linewidth=0.8)
            axes[1, 1].set_title("Residual from a linear cone")
            axes[1, 1].set_ylabel("Residual (in)")
        else:
            axes[1, 1].set_axis_off()
        if dual:
            axes[1, 1].set_xlabel(position_label)
        fig.tight_layout()
        return _png(fig)


def _method_figure(features: pd.DataFrame) -> bytes:
    minor = features["minor_diameter"].to_numpy(dtype=float)
    major = features["major_diameter"].to_numpy(dtype=float)
    equivalent = features["equivalent_diameter"].to_numpy(dtype=float)
    position = features["axial_position"].to_numpy(dtype=float)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        _scatter_smooth(axes[0], minor, major, "#1E8449")
        limits = [
            min(float(minor.min()), float(major.min())),
            max(float(minor.max()), float(major.max())),
        ]
        axes[0].plot(limits, limits, color="#888888", linestyle="--", linewidth=1, label="Equal axes")
        axes[0].set_xlabel("Minor diameter (in)")
        axes[0].set_ylabel("Major diameter (in)")
        axes[0].set_title("Round probe and oval probe")
        axes[0].legend(loc="best", fontsize=7)
        _scatter_smooth(axes[1], minor, equivalent, "#2471A3", "Versus minor")
        axes[1].scatter(major, equivalent, s=12, color="#1E8449", alpha=0.7, label="Versus major")
        axes[1].set_xlabel("Probe diameter (in)")
        axes[1].set_ylabel("Area-equivalent diameter (in)")
        axes[1].set_title("Area-equivalent and each axis")
        axes[1].legend(loc="best", fontsize=7)
        axes[2].plot(position, major - equivalent, color="#1E8449", label="Major − equivalent")
        axes[2].plot(position, equivalent - minor, color="#2471A3", label="Equivalent − minor")
        axes[2].set_xlabel("Axial position (in)")
        axes[2].set_ylabel("Diameter difference (in)")
        axes[2].set_title("Choice of ideal diameter")
        axes[2].legend(loc="best", fontsize=7)
        axes[2].grid(True, color="#e6e6e6", linewidth=0.5)
        fig.tight_layout()
        return _png(fig)


def _heatmap_figure(features: pd.DataFrame) -> bytes:
    corr = features.corr(numeric_only=True)
    labels = [_LABELS.get(column, column) for column in corr.columns]
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8.2, 6.4))
        image = ax.imshow(corr.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for row in range(corr.shape[0]):
            for col in range(corr.shape[1]):
                ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title("Correlations among derived bore features")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return _png(fig)


def _pca_figure(result: dict[str, Any]) -> bytes:
    scores = result["scores"]
    position = result["position"]
    explained = result["explained"]
    loadings = result["loadings"]
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
        plot = axes[0].scatter(scores[:, 0], scores[:, 1], c=position, cmap="viridis", s=16)
        axes[0].set_xlabel(f"PC1 ({explained[0] * 100:.0f}%)")
        axes[0].set_ylabel(f"PC2 ({explained[1] * 100:.0f}%)")
        axes[0].set_title("Principal components, colored by position")
        fig.colorbar(plot, ax=axes[0], fraction=0.046, pad=0.04, label="Axial position (in)")
        ypos = np.arange(len(loadings))
        axes[1].barh(ypos - 0.15, loadings["PC1"], height=0.3, color="#2471A3", label="PC1")
        axes[1].barh(ypos + 0.15, loadings["PC2"], height=0.3, color="#B9770E", label="PC2")
        axes[1].set_yticks(ypos, [_LABELS.get(name, name) for name in loadings.index])
        axes[1].axvline(0, color="#888888", linewidth=0.8)
        axes[1].legend(loc="best", fontsize=7)
        axes[1].set_title("Loadings")
        fig.tight_layout()
        return _png(fig)


def _segment_figure(features: pd.DataFrame, segments: pd.Series, position_label: str) -> bytes:
    position = features["axial_position"].to_numpy(dtype=float)
    colors = {
        "lower ovality": "#2471A3",
        "moderate ovality": "#B9770E",
        "higher ovality": "#1E8449",
        "shallower taper": "#2471A3",
        "moderate taper": "#B9770E",
        "steeper taper": "#1E8449",
    }
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for name, color in colors.items():
            mask = segments.to_numpy() == name
            if not mask.any():
                continue
            y_column = "absolute_ovality" if "absolute_ovality" in features else "taper_rate"
            axes[0].scatter(position[mask], features[y_column].to_numpy()[mask], s=14, color=color, label=name)
        axes[0].set_xlabel(position_label)
        axes[0].set_ylabel("Ovality (in)" if "absolute_ovality" in features else "Taper rate")
        axes[0].set_title("Bore segments")
        axes[0].legend(loc="best", fontsize=7)
        axes[0].grid(True, color="#e6e6e6", linewidth=0.5)
        score = features["cone_residual"].abs()
        if "absolute_ovality" in features:
            ovality = features["absolute_ovality"].abs()
            score = score / (score.std(ddof=1) or 1) + ovality / (ovality.std(ddof=1) or 1)
        axes[1].plot(position, score, color="#6C3483", linewidth=1.4)
        axes[1].set_xlabel(position_label)
        axes[1].set_ylabel("Irregularity")
        axes[1].set_title("Combined residual and ovality")
        axes[1].grid(True, color="#e6e6e6", linewidth=0.5)
        fig.tight_layout()
        return _png(fig)


def _notes_for_dual(
    features: pd.DataFrame,
    position_label: str,
    depth_gap: dict[str, float] | None,
    pca: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    recommendations: list[str] = []
    regions = _region_means(features, "absolute_ovality")
    if regions is not None:
        peak = str(regions.idxmax())
        notes.append(
            f"Mean absolute ovality is highest {peak} "
            f"({regions.max():.4g} in). Position is {position_label.lower()}."
        )
    outliers = _outliers(features, "absolute_ovality")
    if outliers:
        notes.append("Ovality beyond 3 standard deviations: " + "; ".join(outliers) + ".")
    else:
        notes.append("No absolute-ovality station is beyond 3 standard deviations.")
    taper_regions = _region_means(features, "taper_rate")
    if taper_regions is not None:
        notes.append(
            "Mean local taper rate by third of the bore: "
            + "; ".join(f"{name} {value:.4g}" for name, value in taper_regions.items())
            + " in/in."
        )
        if float(taper_regions.iloc[0]) < 0:
            notes.append(
                "A negative taper rate means the diameter decreases as distance from the top "
                "increases, which is the contraction into the throat."
            )
    residual = features["cone_residual"].abs()
    position = features["axial_position"]
    early = residual[position <= position.quantile(0.2)].mean()
    late = residual[position >= position.quantile(0.8)].mean()
    side = "near the top" if early >= late else "toward the bell"
    notes.append(
        f"The linear cone residual is larger {side} "
        f"(mean absolute residual {max(early, late):.4g} in)."
    )
    if depth_gap is not None:
        notes.append(
            "Round-probe and oval-probe insertion depths differ by "
            f"{depth_gap['mean_abs_gap']:.4g} in on average "
            f"(correlation {depth_gap['correlation']:.3f})."
        )
    area = _area_table(features)
    minor_change = float(area.iloc[0]["mean_area_change_percent"])
    major_change = float(area.iloc[1]["mean_area_change_percent"])
    recommendations.append(
        "The area-equivalent diameter matches the area of the measured ellipse. "
        f"That circle has about {minor_change:.1f}% more area than a circle of the minor diameter, "
        f"and about {abs(major_change):.1f}% less area than a circle of the major diameter."
    )
    choice = _taper_choice(features)
    ranked = choice.iloc[:2].copy()
    ranked["abs_change"] = ranked["taper_change_from_equivalent"].abs()
    farther = ranked.sort_values("abs_change").iloc[-1]
    recommendations.append(
        f"Copying the {farther['profile'].lower()} instead of the area-equivalent profile "
        f"changes the straight-taper slope by {farther['taper_change_from_equivalent']:.4g} in/in."
    )
    if pca is not None:
        top = pca["loadings"]["PC1"].abs().sort_values(ascending=False)
        lead = _LABELS.get(str(top.index[0]), str(top.index[0]))
        recommendations.append(
            f"The first principal component accounts for {pca['explained'][0] * 100:.0f}% of the "
            f"standardized feature variation. {lead} has the largest loading on that component."
        )
    recommendations.append(
        "Where ovality is small, a minor-axis, major-axis, or area-equivalent reamer nearly agree. "
        "Where the area gap is large, the area-equivalent profile keeps the measured cross-section "
        "area, and the major-axis profile cuts the wider clearance."
    )
    return notes, recommendations


def _notes_for_single(features: pd.DataFrame, position_label: str) -> tuple[list[str], list[str]]:
    residual = features["cone_residual"].abs()
    position = features["axial_position"]
    early = residual[position <= position.quantile(0.2)].mean()
    late = residual[position >= position.quantile(0.8)].mean()
    side = "near the start of this position axis" if early >= late else "toward the far end"
    notes = [
        "This section has one probe diameter, so minor and major axes are not both available "
        "and ovality is not defined.",
        f"Axial position is {position_label.lower()}.",
        f"The linear cone residual is larger {side} "
        f"(mean absolute residual {max(early, late):.4g} in).",
    ]
    outliers = _outliers(features, "taper_rate")
    if outliers:
        notes.append("Taper-rate stations beyond 3 standard deviations: " + "; ".join(outliers) + ".")
    recommendations = [
        "Use the diameter and taper-rate plots to see where this section leaves a straight cone. "
        "An area-equivalent comparison needs both a round-probe depth and an oval-probe depth."
    ]
    return notes, recommendations


def analyze_bore_shape(table: dict[str, Any]) -> dict[str, Any] | None:
    """Univariate, bivariate, and multivariate shape checks for one bore section."""
    frame = table["frame"]
    oal = table.get("oal")
    dual_frame = _dual_features(frame, oal)
    position_label = "Distance from top (in)" if oal is not None else "Axial position (in)"
    if dual_frame is not None:
        features = dual_frame
        dual = True
    else:
        single = _single_features(frame, oal)
        if single is None:
            return None
        features, position_label = single
        dual = False
    summary = _summarize(features)
    area = _area_table(features) if dual else None
    taper = _taper_choice(features) if dual else None
    depth_gap = _probe_depth_gap(frame) if dual else None
    pca = _pca(features)
    segments = _segments(features)
    if dual:
        notes, recommendations = _notes_for_dual(features, position_label, depth_gap, pca)
    else:
        notes, recommendations = _notes_for_single(features, position_label)
    figures: list[dict[str, Any]] = [
        {"name": "Feature distributions", "png": _distribution_figure(features, dual)},
        {
            "name": "Features along the bore",
            "png": _bivariate_figure(features, position_label, dual),
        },
        {"name": "Derived-feature correlations", "png": _heatmap_figure(features)},
    ]
    if dual:
        figures.insert(1, {"name": "Diameter comparison", "png": _diameter_comparison_figure(features)})
        figures.insert(2, {"name": "Ovality distribution", "png": _ovality_figure(features)})
        figures.insert(4, {"name": "Probe axes and ideal diameter", "png": _method_figure(features)})
    if pca is not None:
        figures.append({"name": "Principal components", "png": _pca_figure(pca)})
    if segments is not None:
        figures.append(
            {
                "name": "Bore segments and irregularity",
                "png": _segment_figure(features, segments, position_label),
            }
        )
    return {
        "dual": dual,
        "summary": summary,
        "area": area,
        "taper_choice": taper,
        "notes": notes,
        "recommendations": recommendations,
        "figures": figures,
    }


def _binned_means(position: np.ndarray, values: np.ndarray, bins: int = 8) -> np.ndarray:
    """Mean of values in equal-count axial bins. Bins limit the dependence of neighboring stations."""
    order = np.argsort(position)
    ordered = np.asarray(values, dtype=float)[order]
    count = max(3, min(bins, len(ordered) // 3))
    chunks = [chunk for chunk in np.array_split(ordered, count) if len(chunk)]
    return np.array([float(np.mean(chunk)) for chunk in chunks])


def _one_sample_check(samples: np.ndarray, null: float = 0.0) -> dict[str, float] | None:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) < 3 or float(np.std(samples)) == 0:
        return None
    result = ttest_1samp(samples, null)
    return {
        "n": float(len(samples)),
        "mean": float(np.mean(samples)),
        "statistic": float(result.statistic),
        "p": float(result.pvalue),
    }


def _residual_signature(position: np.ndarray, residual: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    span = float(position.max() - position.min())
    if span <= 0 or len(position) < 4:
        return None
    fraction = (position - float(position.min())) / span
    order = np.argsort(fraction)
    return np.interp(grid, fraction[order], residual[order])


def _shift_correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Pearson correlation and a circular-shift p-value for two residual signatures."""
    observed = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(observed):
        return np.nan, np.nan
    shifts = np.unique(np.linspace(1, len(right) - 1, num=min(80, len(right) - 1), dtype=int))
    null = np.array([float(np.corrcoef(left, np.roll(right, int(shift)))[0, 1]) for shift in shifts])
    null = null[np.isfinite(null)]
    if len(null) == 0:
        return observed, np.nan
    p_value = (float(np.sum(np.abs(null) >= abs(observed))) + 1.0) / (len(null) + 1.0)
    return observed, p_value


def _profile_record(table: dict[str, Any]) -> dict[str, Any] | None:
    from agents.eda_execute import _overlay_label

    features = _dual_features(table["frame"], table.get("oal"))
    if features is None or len(features) < 8:
        return None
    position = features["axial_position"].to_numpy(dtype=float)
    major = features["major_diameter"].to_numpy(dtype=float)
    equivalent = features["equivalent_diameter"].to_numpy(dtype=float)
    minor = features["minor_diameter"].to_numpy(dtype=float)
    slope_eq, _intercept_eq = np.polyfit(position, equivalent, 1)
    slope_major, _intercept_major = np.polyfit(position, major, 1)
    residual_eq = equivalent - (slope_eq * position + _intercept_eq)
    residual_major = major - (slope_major * position + _intercept_major)
    area_excess = np.where(minor > 1e-9, (major / minor - 1.0) * 100.0, np.nan)
    return {
        "name": _overlay_label(table),
        "table": table,
        "features": features,
        "position": position,
        "major": major,
        "equivalent": equivalent,
        "slope_eq": float(slope_eq),
        "slope_major": float(slope_major),
        "residual_eq": residual_eq,
        "residual_major": residual_major,
        "area_excess": area_excess,
    }


def _pooled_shape_pca(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """PCA on within-bore centered shape features, so instrument size does not set the first mode."""
    columns = (
        "absolute_ovality",
        "relative_ovality",
        "taper_rate",
        "cone_residual",
    )
    frames = []
    for record in records:
        frame = record["features"][list(columns)].astype(float).copy()
        frame["major_minus_equivalent"] = record["major"] - record["equivalent"]
        frame = frame - frame.mean()
        frame["bore"] = record["name"]
        frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True)
    feature_columns = [column for column in pooled.columns if column != "bore"]
    usable = [column for column in feature_columns if float(pooled[column].std(ddof=1) or 0) > 0]
    if len(usable) < 2 or len(pooled) < 8:
        return None
    matrix = pooled[usable].to_numpy(dtype=float)
    scale = matrix.std(axis=0, ddof=1)
    scale[scale == 0] = 1.0
    standardized = (matrix - matrix.mean(axis=0)) / scale
    _u, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    explained = singular**2
    explained = explained / explained.sum()
    scores = standardized @ vt[:2].T
    loadings = pd.DataFrame(vt[:2].T, index=usable, columns=["PC1", "PC2"])
    return {
        "explained": explained,
        "scores": scores,
        "bores": pooled["bore"].to_numpy(),
        "loadings": loadings,
    }


def _comparison_rows(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        position = record["position"]
        gap = _binned_means(position, record["major"] - record["equivalent"])
        area = _binned_means(position, record["area_excess"])
        squared_gap = _binned_means(
            position, record["residual_major"] ** 2 - record["residual_eq"] ** 2
        )
        gap_test = _one_sample_check(gap)
        area_test = _one_sample_check(area)
        variance_test = _one_sample_check(squared_gap)
        var_eq = float(np.var(record["residual_eq"], ddof=1))
        var_major = float(np.var(record["residual_major"], ddof=1))
        rows.append(
            {
                "bore": record["name"],
                "taper_equivalent": record["slope_eq"],
                "taper_major": record["slope_major"],
                "taper_difference": record["slope_major"] - record["slope_eq"],
                "residual_var_equivalent": var_eq,
                "residual_var_major": var_major,
                "residual_var_ratio": var_major / var_eq if var_eq > 0 else np.nan,
                "mean_area_excess_percent": float(np.nanmean(record["area_excess"])),
                "diameter_gap_p": None if gap_test is None else gap_test["p"],
                "area_excess_p": None if area_test is None else area_test["p"],
                "residual_variance_p": None if variance_test is None else variance_test["p"],
            }
        )
    return pd.DataFrame(rows)


def _signature_pairs(
    records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[tuple[str, np.ndarray]], np.ndarray]:
    grid = np.linspace(0.0, 1.0, 50)
    signatures = []
    for record in records:
        signature = _residual_signature(record["position"], record["residual_eq"], grid)
        if signature is not None:
            signatures.append((record["name"], signature))
    rows = []
    for index, (left_name, left) in enumerate(signatures):
        for right_name, right in signatures[index + 1 :]:
            correlation, p_value = _shift_correlation(left, right)
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "correlation": correlation,
                    "shift_p": p_value,
                }
            )
    return pd.DataFrame(rows), signatures, grid


def _format_p(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    if value < 0.001:
        return "< 0.001"
    return f"{value:.3f}"


def _maker_markdown(
    comparison: pd.DataFrame,
    pairs: pd.DataFrame,
    pca: dict[str, Any] | None,
) -> str:
    lines = [
        "# Reamer notes from the shape comparison",
        "",
        "The major-axis profile is the oval-probe diameter. The area-equivalent profile is the circle with the same area as that oval, the square root of minor times major. A reamer follows one of those circles. Each check below uses eight axial bins so neighboring stations are not treated as independent measurements.",
        "",
        "## Dominant modes of variation",
        "",
    ]
    if pca is None:
        lines.append("There were not enough shape features for a principal component summary.")
    else:
        loadings = pca["loadings"]["PC1"].abs().sort_values(ascending=False)
        lead = _LABELS.get(str(loadings.index[0]), str(loadings.index[0]).replace("_", " "))
        second = loadings.index[1] if len(loadings) > 1 else loadings.index[0]
        second_name = _LABELS.get(str(second), str(second).replace("_", " "))
        lines.append(
            f"After removing each bore's own average, the first component accounts for "
            f"{pca['explained'][0] * 100:.0f}% of the remaining shape variation and the second for "
            f"{pca['explained'][1] * 100:.0f}%. The strongest loading on the first component is "
            f"{lead}. The next is {second_name}. These are the dominant ways the bores leave a constant oval and a constant taper."
        )
    lines.extend(["", "## Area-equivalent profile versus the major axis", ""])
    lines.append(
        "For each bore the straight taper is the slope of diameter against distance from the top. "
        "Residual variance is the scatter left after that straight cone. Area excess is how much larger a circle of the major diameter is than the ellipse."
    )
    lines.append("")
    for _, row in comparison.iterrows():
        lines.append(
            f"- **{row['bore']}.** Area-equivalent taper {row['taper_equivalent']:.4f} in/in; "
            f"major-axis taper {row['taper_major']:.4f} in/in; difference "
            f"{row['taper_difference']:+.4f} in/in. Residual variance is "
            f"{row['residual_var_equivalent']:.3g} for the area-equivalent cone and "
            f"{row['residual_var_major']:.3g} for the major axis "
            f"(ratio {row['residual_var_ratio']:.2f}). "
            f"The major-axis circle is {row['mean_area_excess_percent']:.2f}% larger in area "
            f"(bin test p {_format_p(row['area_excess_p'])}). "
            f"The diameter gap test p is {_format_p(row['diameter_gap_p'])}."
        )
    if len(comparison) >= 3:
        taper_test = _one_sample_check(comparison["taper_difference"].to_numpy(dtype=float))
        if taper_test is not None:
            lines.append("")
            lines.append(
                f"Across these bores, the mean major-axis taper minus the area-equivalent taper is "
                f"{taper_test['mean']:+.4f} in/in (p {_format_p(taper_test['p'])}). "
                "A small difference means the two profiles open at nearly the same rate."
            )
    lines.extend(["", "## Do the bores share a residual signature?", ""])
    lines.append(
        "Each residual signature is what remains after a straight cone is subtracted from the area-equivalent diameter, stretched onto a 0–1 fraction of that bore. A positive correlation means two bores leave the cone in the same places."
    )
    if pairs.empty:
        lines.append("")
        lines.append("At least two oval bores are needed before residual signatures can be compared.")
    else:
        lines.append("")
        for _, row in pairs.iterrows():
            lines.append(
                f"- {row['left']} and {row['right']}: r = {row['correlation']:.2f} "
                f"(circular-shift p {_format_p(row['shift_p'])})."
            )
        mean_r = float(pairs["correlation"].mean())
        strong = int(((pairs["correlation"] >= 0.5) & (pairs["shift_p"] < 0.1)).sum())
        lines.append("")
        if mean_r >= 0.5 and strong >= max(1, len(pairs) // 2):
            verdict = (
                "The residual wiggles line up across most of these bores. That is consistent with a shared departure from a straight cone, such as a similar reamer or a similar making habit. It does not by itself prove that one reamer was used."
            )
        elif mean_r >= 0.3:
            verdict = (
                "Some pairs share a residual pattern, and others do not. A single reamer curve would capture only part of the departure from a cone."
            )
        else:
            verdict = (
                "The residual patterns do not line up. The departures from a straight cone are particular to each bore, so one shared reamer curve would not reproduce them."
            )
        lines.append(
            f"The mean pairwise correlation is {mean_r:.2f}. {strong} of {len(pairs)} pairs stay at or above 0.5 after allowing for a circular shift of the pattern. {verdict}"
        )
    lines.extend(
        [
            "",
            "## For the maker",
            "",
            "Use the area-equivalent profile when the goal is to keep the measured cross-section area. Use the major-axis profile when the goal is the wider clearance of the oval. On these bores the two tapers are close, and the area gap is the clearer difference. Match a reamer to the residual signature only where the curves above actually agree; elsewhere cut the cone and then the local departure for that instrument.",
            "",
        ]
    )
    return "\n".join(lines)


def _hypothesis_figure(
    records: list[dict[str, Any]],
    comparison: pd.DataFrame,
    pairs: pd.DataFrame,
    signatures: list[tuple[str, np.ndarray]],
    grid: np.ndarray,
    pca: dict[str, Any] | None,
) -> bytes:
    colors = (
        "#1a5276",
        "#c0392b",
        "#1e8449",
        "#6c3483",
        "#b9770e",
        "#1a6f8a",
        "#7b241c",
    )
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.0))
        if pca is not None:
            names = list(dict.fromkeys(pca["bores"]))
            for index, name in enumerate(names):
                mask = pca["bores"] == name
                axes[0, 0].scatter(
                    pca["scores"][mask, 0],
                    pca["scores"][mask, 1],
                    s=12,
                    color=colors[index % len(colors)],
                    alpha=0.75,
                    label=name,
                )
            axes[0, 0].legend(loc="best", fontsize=6)
            axes[0, 0].set_xlabel(f"PC1 ({pca['explained'][0] * 100:.0f}%)")
            axes[0, 0].set_ylabel(f"PC2 ({pca['explained'][1] * 100:.0f}%)")
            loadings = pca["loadings"]
            ypos = np.arange(len(loadings))
            axes[0, 1].barh(ypos - 0.15, loadings["PC1"], height=0.3, color="#2471A3", label="PC1")
            axes[0, 1].barh(ypos + 0.15, loadings["PC2"], height=0.3, color="#B9770E", label="PC2")
            axes[0, 1].set_yticks(
                ypos,
                [_LABELS.get(name, name.replace("_", " ")) for name in loadings.index],
            )
            axes[0, 1].axvline(0, color="#888888", linewidth=0.8)
            axes[0, 1].legend(loc="best", fontsize=7)
        else:
            axes[0, 0].set_axis_off()
            axes[0, 1].set_axis_off()
        axes[0, 0].set_title("Shape features, first two components")
        axes[0, 1].set_title("What drives those components")
        axes[0, 0].grid(True, color="#e6e6e6", linewidth=0.5)
        for index, (name, signature) in enumerate(signatures):
            axes[0, 2].plot(
                grid,
                signature,
                color=colors[index % len(colors)],
                linewidth=1.5,
                label=name,
            )
        axes[0, 2].axhline(0, color="#888888", linewidth=0.8)
        axes[0, 2].set_xlabel("Fraction of bore length")
        axes[0, 2].set_ylabel("Cone residual (in)")
        axes[0, 2].set_title("Residual signature")
        axes[0, 2].legend(loc="best", fontsize=6)
        axes[0, 2].grid(True, color="#e6e6e6", linewidth=0.5)
        if not pairs.empty:
            names = list(dict.fromkeys([*pairs["left"], *pairs["right"]]))
            matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
            for _, row in pairs.iterrows():
                matrix.loc[row["left"], row["right"]] = row["correlation"]
                matrix.loc[row["right"], row["left"]] = row["correlation"]
            image = axes[1, 0].imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
            axes[1, 0].set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=6)
            axes[1, 0].set_yticks(range(len(names)), names, fontsize=6)
            fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)
        axes[1, 0].set_title("Residual-signature correlation")
        xpos = np.arange(len(comparison))
        width = 0.36
        axes[1, 1].bar(
            xpos - width / 2,
            comparison["taper_equivalent"],
            width=width,
            color="#6C3483",
            label="Area-equivalent",
        )
        axes[1, 1].bar(
            xpos + width / 2,
            comparison["taper_major"],
            width=width,
            color="#1E8449",
            label="Major axis",
        )
        axes[1, 1].set_xticks(xpos, comparison["bore"], rotation=25, ha="right", fontsize=6)
        axes[1, 1].set_ylabel("Taper (in/in)")
        axes[1, 1].set_title("Taper rate")
        axes[1, 1].legend(loc="best", fontsize=7)
        axes[1, 1].grid(True, axis="y", color="#e6e6e6", linewidth=0.5)
        axes[1, 2].bar(xpos, comparison["mean_area_excess_percent"], color="#B9770E")
        axes[1, 2].axhline(0, color="#888888", linewidth=0.8)
        axes[1, 2].set_xticks(xpos, comparison["bore"], rotation=25, ha="right", fontsize=6)
        axes[1, 2].set_ylabel("Area excess of major axis (%)")
        axes[1, 2].set_title("Area deviation")
        axes[1, 2].grid(True, axis="y", color="#e6e6e6", linewidth=0.5)
        fig.suptitle("Area-equivalent and major-axis profiles", fontsize=14)
        fig.tight_layout()
        return _png(fig)


def compare_reamer_hypotheses(tables: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Cross-bore PCA, cone-residual signatures, and area-equivalent versus major-axis checks."""
    from agents.eda_execute import _report_sort_key

    bore_tables = [table for table in tables if table.get("kind", "bore") == "bore"]
    bore_tables.sort(key=_report_sort_key)
    records = [record for table in bore_tables if (record := _profile_record(table)) is not None]
    if len(records) < 1:
        return None
    comparison = _comparison_rows(records)
    pairs, signatures, grid = _signature_pairs(records)
    pca = _pooled_shape_pca(records)
    markdown = _maker_markdown(comparison, pairs, pca)
    figure = _hypothesis_figure(records, comparison, pairs, signatures, grid, pca)
    sections = _markdown_sections(markdown)
    return {
        "markdown": markdown,
        "sections": sections,
        "figure": figure,
        "comparison": comparison,
        "pairs": pairs,
    }


def _markdown_sections(markdown: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    heading = ""
    paragraphs: list[str] = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            if heading or paragraphs:
                sections.append({"heading": heading, "paragraphs": paragraphs})
            heading = line.lstrip("#").strip()
            paragraphs = []
        elif line.startswith("- "):
            paragraphs.append(line[2:])
        elif line:
            paragraphs.append(line)
    if heading or paragraphs:
        sections.append({"heading": heading, "paragraphs": paragraphs})
    return sections
