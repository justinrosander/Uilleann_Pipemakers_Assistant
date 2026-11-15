"""Canonical sheet/column expectations derived from UP_measurement_template.xlsx."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SheetKind = Literal[
    "chanter_regulator_bore",
    "toneholes",
    "drone_bore",
    "bass_drone_bore",
]


@dataclass(frozen=True)
class SheetSpec:
    name: str
    kind: SheetKind
    instrument_family: str
    instrument_name: str
    expected_bore_headers: tuple[str, ...] = ()
    expected_tonehole_headers: tuple[str, ...] = (
        "keyhole",
        "d-bottom-bell",
        "d-top-reedseat",
        "d-ctr-bell",
        "d-axial",
        "d-xverse",
        "chimney",
        "ch_diam",
        "notes",
    )
    oal_labels: tuple[str, ...] = ()
    # Present on the template for folded builds, but a source file may omit them.
    optional_oal_labels: tuple[str, ...] = ()
    optional_sections: tuple[str, ...] = ()
    multi_section: bool = False


CHANTER_REG_BORE_HEADERS = ("Nominal", "Diameter", "Round", "Oval", "Notes")
DRONE_BORE_HEADERS = ("Nominal", "Diameter", "Oval", "Notes")
U_BEND_HEADERS = (
    "ID",
    "OD x-axis",
    "OD y-axis",
    "Wall",
    "Diameter OD",
    "Radius",
    "Length",
)
# Folded bass-upper U-bend. OD is the maker's estimate of roundness from the
# two OD axes. Notes and any taper/length fields may be partial.
BASS_UPPER_U_BEND_HEADERS = (
    "ID",
    "OD x-axis",
    "OD y-axis",
    "OD",
    "Wall",
    "Diameter OD",
    "Radius",
    "Length",
    "Notes",
)
SLIDE_BRASS_HEADERS = ("ID", "OD", "Wall", "Length")

SHEET_SPECS: dict[str, SheetSpec] = {
    "Chanter Bore": SheetSpec(
        name="Chanter Bore",
        kind="chanter_regulator_bore",
        instrument_family="chanter",
        instrument_name="Chanter",
        expected_bore_headers=CHANTER_REG_BORE_HEADERS,
        oal_labels=("OAL",),
    ),
    "Chanter Toneholes": SheetSpec(
        name="Chanter Toneholes",
        kind="toneholes",
        instrument_family="chanter",
        instrument_name="Chanter",
        oal_labels=("OAL",),
    ),
    "Tenor Bore": SheetSpec(
        name="Tenor Bore",
        kind="chanter_regulator_bore",
        instrument_family="regulator",
        instrument_name="Tenor Regulator",
        expected_bore_headers=CHANTER_REG_BORE_HEADERS,
        oal_labels=("OAL",),
    ),
    "Tenor Toneholes": SheetSpec(
        name="Tenor Toneholes",
        kind="toneholes",
        instrument_family="regulator",
        instrument_name="Tenor Regulator",
        oal_labels=("OAL",),
    ),
    "Baritone Bore": SheetSpec(
        name="Baritone Bore",
        kind="chanter_regulator_bore",
        instrument_family="regulator",
        instrument_name="Baritone Regulator",
        expected_bore_headers=CHANTER_REG_BORE_HEADERS,
        oal_labels=("OAL",),
    ),
    "Baritone Toneholes": SheetSpec(
        name="Baritone Toneholes",
        kind="toneholes",
        instrument_family="regulator",
        instrument_name="Baritone Regulator",
        oal_labels=("OAL",),
    ),
    "Bass Upper Bore": SheetSpec(
        name="Bass Upper Bore",
        kind="chanter_regulator_bore",
        instrument_family="regulator",
        instrument_name="Bass Regulator",
        expected_bore_headers=CHANTER_REG_BORE_HEADERS,
        oal_labels=("OAL",),
        optional_oal_labels=("Reed Seat OAL", "U-bend OAL"),
        optional_sections=("Reed Seat", "U-bend"),
        multi_section=True,
    ),
    "Bass Lower Bore": SheetSpec(
        name="Bass Lower Bore",
        kind="chanter_regulator_bore",
        instrument_family="regulator",
        instrument_name="Bass Regulator",
        expected_bore_headers=CHANTER_REG_BORE_HEADERS,
        oal_labels=("OAL",),
    ),
    "Bass Toneholes": SheetSpec(
        name="Bass Toneholes",
        kind="toneholes",
        instrument_family="regulator",
        instrument_name="Bass Regulator",
        oal_labels=("OAL",),
    ),
    "Tenor Drone Bore": SheetSpec(
        name="Tenor Drone Bore",
        kind="drone_bore",
        instrument_family="drone",
        instrument_name="Tenor Drone",
        expected_bore_headers=DRONE_BORE_HEADERS,
        oal_labels=("Upper Bore OAL", "Lower Bore OAL"),
        multi_section=True,
    ),
    "Baritone Drone Bore": SheetSpec(
        name="Baritone Drone Bore",
        kind="drone_bore",
        instrument_family="drone",
        instrument_name="Baritone Drone",
        expected_bore_headers=DRONE_BORE_HEADERS,
        oal_labels=("Upper Bore OAL", "Lower Bore OAL"),
        multi_section=True,
    ),
    "Bass Drone Bore": SheetSpec(
        name="Bass Drone Bore",
        kind="bass_drone_bore",
        instrument_family="drone",
        instrument_name="Bass Drone",
        expected_bore_headers=DRONE_BORE_HEADERS,
        oal_labels=(
            "J1 OAL",
            "J2 OAL",
            "J3 OAL",
            "Slide OAL",
            "U1 OAL",
            "U2 OAL",
            "U3 OAL",
        ),
        multi_section=True,
    ),
}


def classify_sheet(sheet_name: str) -> SheetSpec | None:
    if sheet_name in SHEET_SPECS:
        return SHEET_SPECS[sheet_name]
    lowered = sheet_name.strip().lower()
    for name, spec in SHEET_SPECS.items():
        if name.lower() == lowered:
            return spec
    return None


def normalize_header(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()
