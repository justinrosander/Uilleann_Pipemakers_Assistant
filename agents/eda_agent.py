"""EDA plan for bore and reamer design.

The steps follow 'ML EDA Problem Definition.md'. Running the EDA agent
computes them in agents.eda_execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from config import ROOT, TEMPLATE_PATH
from schema.template_schema import SHEET_SPECS, classify_sheet

BRIEF_PATH = ROOT / "ML EDA Problem Definition.md"

# Columns the planned bore analyses read. Drones may omit Round.
BORE_LENGTH_FIELDS = ("Nominal", "Round", "Oval")
BORE_DIAMETER_FIELD = "Diameter"
CORRELATION_THRESHOLD = 0.7


@dataclass(frozen=True)
class EdaStep:
    """One planned analysis. status is planned until an execute path exists."""

    id: str
    title: str
    question: str
    required_fields: tuple[str, ...]
    applies_to: str
    method: str
    status: str = "planned"


@dataclass
class SheetReadiness:
    sheet_name: str
    instrument_title: str
    family: str
    kind: str
    headers_found: list[str]
    missing_for_round_oval: list[str]
    notes: str = ""


@dataclass
class EdaPlan:
    source: str | None
    brief_path: str
    business_context: str
    objectives: list[str]
    steps: list[EdaStep]
    sheets: list[SheetReadiness] = field(default_factory=list)
    executed: bool = False

    def as_lines(self) -> list[str]:
        lines = [
            "EDA framework (plan only; no analysis was run)",
            f"Brief: {self.brief_path}",
            f"Source inspected: {self.source or '(none)'}",
            "",
            "Business context:",
            self.business_context,
            "",
            "Analytical objectives:",
        ]
        for index, objective in enumerate(self.objectives, start=1):
            lines.append(f"  {index}. {objective}")
        lines.append("")
        lines.append("Planned steps:")
        for step in self.steps:
            lines.append(f"  [{step.status}] {step.id}: {step.title}")
            lines.append(f"      Question: {step.question}")
            lines.append(f"      Applies to: {step.applies_to}")
            lines.append(f"      Method: {step.method}")
            lines.append(f"      Fields: {', '.join(step.required_fields)}")
        if self.sheets:
            lines.append("")
            lines.append("Template readiness:")
            for sheet in self.sheets:
                missing = ", ".join(sheet.missing_for_round_oval) or "none"
                lines.append(
                    f"  {sheet.sheet_name} | {sheet.instrument_title} | "
                    f"{sheet.family} | headers: {', '.join(sheet.headers_found) or '(none)'} "
                    f"| missing for round/oval profile: {missing}"
                )
                if sheet.notes:
                    lines.append(f"      {sheet.notes}")
        lines.append("")
        lines.append(
            "This plan is what --eda runs when a workbook is passed with --source."
        )
        return lines


def load_brief(path: Path | None = None) -> tuple[str, list[str]]:
    """Read the business context and the analytical objective bullets."""
    brief_path = path or BRIEF_PATH
    text = brief_path.read_text(encoding="utf-8")
    context_lines: list[str] = []
    objectives: list[str] = []
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("business context"):
            section = "context"
            continue
        if line.lower().startswith("analytical objective"):
            section = "objectives"
            continue
        if section == "context":
            context_lines.append(line)
        elif section == "objectives" and line.startswith("*"):
            objectives.append(line.lstrip("*").strip())
    context = " ".join(context_lines).strip()
    if not context or not objectives:
        raise ValueError(f"Could not read EDA objectives from {brief_path}")
    return context, objectives


def planned_steps() -> list[EdaStep]:
    """The analysis sequence from the problem definition. Nothing here is executed."""
    return [
        EdaStep(
            id="bore_visualizations",
            title="Visualize each instrument bore",
            question="What do the measured bore profiles look like for the chanter, regulators, and drones?",
            required_fields=("Diameter", "Round", "Oval"),
            applies_to="chanter, regulator, and drone bore sheets",
            method="Plot diameter on the horizontal axis and depth on the vertical axis.",
        ),
        EdaStep(
            id="chanter_vs_regulators",
            title="Compare chanter and regulator bores",
            question="How do the chanter and regulator bore dimensions relate?",
            required_fields=("Diameter", "Round", "Oval"),
            applies_to="chanter bore plus tenor, baritone, and bass regulator bores",
            method="Overlay the measured profiles and report strong correlations.",
        ),
        EdaStep(
            id="round_vs_oval",
            title="Evaluate round and oval measurements",
            question="Where do round and oval readings differ, and what ideal round profile lies between them?",
            required_fields=("Diameter", "Round", "Oval"),
            applies_to="instruments with both round and oval stations",
            method=(
                "Interpolate round and oval probes onto one depth grid. "
                "The ideal diameter is the square root of their product. "
                "Fit a straight taper, polynomials through degree 5, exponential, "
                "power, logarithmic, and spline curves to that diameter. "
                "Chanter, tenor, and baritone fits use only the throat and the bore below it."
            ),
        ),
        EdaStep(
            id="tonehole_placement",
            title="Visualize tonehole placement",
            question="Where are the toneholes on each instrument that has them?",
            required_fields=("keyhole", "d-bottom-bell", "d-ctr-bell", "ch_diam"),
            applies_to="chanter, tenor, baritone, and bass tonehole sheets",
            method="Plot hole position against hole diameter and label each keyhole.",
        ),
        EdaStep(
            id="relationships",
            title="Check dependent and independent relationships",
            question="Do bore and tonehole variables move together, including tonehole placement and bore size?",
            required_fields=("Diameter", "Round", "Oval", "d-bottom-bell", "ch_diam"),
            applies_to="each instrument with bore rows, and toneholes when present",
            method=(
                "Treat diameter as dependent on depth. "
                f"Report pairs with |r| >= {CORRELATION_THRESHOLD}. "
                "Match each tonehole depth to the nearest bore diameter."
            ),
        ),
        EdaStep(
            id="univariate",
            title="Summarize univariate behavior",
            question="How does each measured variable behave on its own?",
            required_fields=("Diameter", "Round", "Oval", "Nominal", "ch_diam"),
            applies_to="all numeric bore and tonehole fields that are present",
            method=(
                "Report count, mean, median, spread, min, max, skewness, and excess kurtosis "
                "for the measured columns and for the derived bore features: axial position, "
                "minor diameter, major diameter, area-equivalent diameter, absolute and relative "
                "ovality, and local taper rate. Include histograms with kernel density, a "
                "box or violin comparison of the diameters, the taper-rate distribution, "
                "ovality outliers, and the axial region where ovality is highest. "
                "Compare the 5th-order polynomial with the other bore-profile models "
                "on the same area-equivalent points."
            ),
        ),
        EdaStep(
            id="ovality",
            title="Summarize bore ovality",
            question="How oval is the bore, where is that ovality largest, and how does it change the cross-section area?",
            required_fields=("Diameter", "Round", "Oval"),
            applies_to="bores with both a round-probe and an oval-probe profile",
            method=(
                "At common axial stations, absolute ovality is major minus minor and relative "
                "ovality is major divided by minor. Plot the histogram, kernel density, and "
                "cumulative distribution. Compare the elliptical area with a circle of the "
                "minor diameter and a circle of the major diameter."
            ),
        ),
        EdaStep(
            id="bivariate",
            title="Highlight strong bivariate relationships",
            question=f"Which variable pairs are strongly correlated (|r| >= {CORRELATION_THRESHOLD}), and how do shape features move with axial position?",
            required_fields=("Diameter", "Round", "Oval", "Nominal", "ch_diam"),
            applies_to="numeric pairs within each instrument section",
            method=(
                f"Pearson correlations on the measured columns, keeping pairs at or above {CORRELATION_THRESHOLD}. "
                "Also plot minor, major, and area-equivalent diameter, ovality, local taper rate, "
                "and linear-cone residuals against axial position, with a polynomial smoother. "
                "Plot minor against major and compare insertion depths of the two probes."
            ),
        ),
        EdaStep(
            id="multivariate",
            title="Multivariate bore shape",
            question="Which combination of position, diameter, ovality, and taper accounts for departure from a straight cone?",
            required_fields=("Diameter", "Round", "Oval"),
            applies_to="each bore section",
            method=(
                "Build a correlation heatmap of the derived features. "
                "Run a principal component analysis on the standardized features. "
                "Segment the bore from taper rate and ovality, and mark the axial zone "
                "where cone residual and ovality are largest. Rank the minor-axis, "
                "major-axis, and area-equivalent profiles by how much the choice changes "
                "the mean diameter and the effective taper. Across bores, compare "
                "area-equivalent and major-axis taper, residual variance, and area, "
                "and test whether linear-cone residual signatures match."
            ),
        ),
    ]


def _header_row_values(ws) -> list[str]:
    """Return the first recognized bore or tonehole header row."""
    wanted = {
        "nominal",
        "diameter",
        "round",
        "oval",
        "notes",
        "keyhole",
        "id",
    }
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 40), values_only=True):
        labels = [str(value).strip() for value in row if value is not None and str(value).strip()]
        lowered = {label.lower() for label in labels}
        if "diameter" in lowered or "keyhole" in lowered or lowered & wanted:
            if len(lowered & wanted) >= 2:
                return labels
    return []


def inspect_workbook(path: Path) -> list[SheetReadiness]:
    """Read sheet titles and headers only. No measurements are aggregated."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    readiness: list[SheetReadiness] = []
    try:
        for sheet_name in workbook.sheetnames:
            spec = classify_sheet(sheet_name)
            if spec is None:
                continue
            ws = workbook[sheet_name]
            title = ws["A1"].value
            headers = _header_row_values(ws)
            header_set = {header.lower() for header in headers}
            missing: list[str] = []
            if spec.kind != "toneholes":
                if "diameter" not in header_set:
                    missing.append("Diameter")
                if "oval" not in header_set and spec.kind != "bass_drone_bore":
                    # U-bend blocks use different headers; bore blocks still need Oval or Round.
                    if "round" not in header_set:
                        missing.append("Round or Oval")
            note = ""
            if spec.kind == "drone_bore":
                note = "Drone bore length is stored in Oval; Round may be absent."
            elif spec.kind == "toneholes":
                note = "Toneholes are out of scope for the reamer-profile regression."
            elif spec.kind == "bass_drone_bore":
                note = "Bass drone mixes bore sections and U-bend sections."
            readiness.append(
                SheetReadiness(
                    sheet_name=sheet_name,
                    instrument_title=str(title).strip() if title else spec.instrument_name,
                    family=spec.instrument_family,
                    kind=spec.kind,
                    headers_found=headers,
                    missing_for_round_oval=missing,
                    notes=note,
                )
            )
    finally:
        workbook.close()
    return readiness


def build_eda_plan(source: Path | None = None, brief_path: Path | None = None) -> EdaPlan:
    """Build the EDA plan and, if a workbook is given, its structural readiness."""
    context, objectives = load_brief(brief_path)
    sheets: list[SheetReadiness] = []
    source_label = None
    if source is not None:
        if not source.exists():
            raise FileNotFoundError(f"EDA source not found: {source}")
        sheets = inspect_workbook(source)
        source_label = str(source)
    return EdaPlan(
        source=source_label,
        brief_path=str(brief_path or BRIEF_PATH),
        business_context=context,
        objectives=objectives,
        steps=planned_steps(),
        sheets=sheets,
        executed=False,
    )


def run_eda_framework(source: Path | None = None) -> EdaPlan:
    """Return the plan. This function never executes the analyses."""
    plan = build_eda_plan(source)
    if plan.executed:
        raise RuntimeError("EDA execution is not available in this framework.")
    return plan


def framework_status() -> dict[str, Any]:
    """Small status payload for tests and CLI checks."""
    context, objectives = load_brief()
    return {
        "brief": str(BRIEF_PATH),
        "template": str(TEMPLATE_PATH),
        "objective_count": len(objectives),
        "step_ids": [step.id for step in planned_steps()],
        "template_sheets": list(SHEET_SPECS),
        "context_present": bool(context),
        "execution": "disabled",
    }
