"""Group committed measurement batches into a pipe set.

A set is one maker's chanter, regulators, and drones that belong together,
usually in one musical key. EDA loads the latest committed batch for each
sheet in the set.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from db.connection import get_connection


def ensure_set_schema() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS instrument_sets (
                id            BIGSERIAL PRIMARY KEY,
                name          TEXT NOT NULL UNIQUE,
                pipemaker     TEXT,
                musical_key   TEXT,
                notes         TEXT,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            ALTER TABLE measurement_batches
                ADD COLUMN IF NOT EXISTS set_id BIGINT
                REFERENCES instrument_sets (id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS measurement_batches_set_idx
                ON measurement_batches (set_id)
            """
        )


def list_sets() -> list[dict[str, Any]]:
    ensure_set_schema()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.name, s.pipemaker, s.musical_key,
                COUNT(b.id) AS batches
            FROM instrument_sets s
            LEFT JOIN measurement_batches b
                ON b.set_id = s.id AND b.status = 'committed'
            GROUP BY s.id
            ORDER BY s.name
            """
        ).fetchall()
    return [dict(row) for row in rows]


def create_set(name: str, pipemaker: str | None = None, musical_key: str | None = None, notes: str | None = None) -> int:
    ensure_set_schema()
    with get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO instrument_sets (name, pipemaker, musical_key, notes)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET
                pipemaker = COALESCE(EXCLUDED.pipemaker, instrument_sets.pipemaker),
                musical_key = COALESCE(EXCLUDED.musical_key, instrument_sets.musical_key),
                notes = COALESCE(EXCLUDED.notes, instrument_sets.notes)
            RETURNING id
            """,
            (name, pipemaker, musical_key, notes),
        ).fetchone()
    return int(row["id"])


def attach_latest_batches(set_id: int, pipemaker: str, musical_key: str) -> list[dict[str, Any]]:
    """Attach the newest committed batch for each sheet of this maker and key.

    Older duplicate commits of the same sheet stay unattached so a set does
    not contain the same bore twice.
    """
    ensure_set_schema()
    with get_connection() as conn:
        rows = conn.execute(
            """
            WITH ranked AS (
                SELECT b.id, b.source_sheet,
                    ROW_NUMBER() OVER (
                        PARTITION BY b.source_sheet
                        ORDER BY b.id DESC
                    ) AS rn
                FROM measurement_batches b
                JOIN instruments i ON i.id = b.instrument_id
                WHERE b.status = 'committed'
                    AND i.pipemaker = %s
                    AND i.musical_key = %s
            )
            UPDATE measurement_batches AS b
            SET set_id = %s
            FROM ranked
            WHERE b.id = ranked.id
                AND ranked.rn = 1
            RETURNING b.id, b.source_sheet
            """,
            (pipemaker, musical_key, set_id),
        ).fetchall()
    return [dict(row) for row in rows]


def assign_batch_to_set(batch_id: int, set_name: str, pipemaker: str | None = None, musical_key: str | None = None) -> int:
    set_id = create_set(set_name, pipemaker, musical_key)
    with get_connection() as conn:
        conn.execute(
            "UPDATE measurement_batches SET set_id = %s WHERE id = %s",
            (set_id, batch_id),
        )
    return set_id


def load_set_bore_tables(set_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the set record and bore tables shaped for the EDA agent."""
    ensure_set_schema()
    with get_connection() as conn:
        set_row = conn.execute(
            """
            SELECT id, name, pipemaker, musical_key, notes
            FROM instrument_sets
            WHERE name = %s
            """,
            (set_name,),
        ).fetchone()
        if set_row is None:
            known = conn.execute("SELECT name FROM instrument_sets ORDER BY name").fetchall()
            names = ", ".join(row["name"] for row in known) or "(none)"
            raise ValueError(f"No instrument set named {set_name!r}. Known sets: {names}")
        batches = conn.execute(
            """
            SELECT b.id, b.source_sheet, i.instrument_name, i.instrument_family, i.units
            FROM measurement_batches b
            JOIN instruments i ON i.id = b.instrument_id
            WHERE b.set_id = %s AND b.status = 'committed'
            ORDER BY b.source_sheet
            """,
            (set_row["id"],),
        ).fetchall()
        tables: list[dict[str, Any]] = []
        for batch in batches:
            measurements = conn.execute(
                """
                SELECT section_name, nominal, diameter, round_val, oval
                FROM bore_measurements
                WHERE batch_id = %s
                ORDER BY section_name, row_index
                """,
                (batch["id"],),
            ).fetchall()
            if not measurements:
                continue
            oal_row = conn.execute(
                """
                SELECT oal FROM section_oals
                WHERE batch_id = %s AND lower(section_name) = 'oal'
                """,
                (batch["id"],),
            ).fetchone()
            oal = float(oal_row["oal"]) if oal_row and oal_row["oal"] is not None else None
            frame = pd.DataFrame(measurements)
            frame = frame.rename(columns={"round_val": "Round", "oval": "Oval", "nominal": "Nominal", "diameter": "Diameter"})
            for column in ("Nominal", "Diameter", "Round", "Oval"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            for section_name, section_frame in frame.groupby("section_name", sort=False):
                section = section_frame.drop(columns=["section_name"])
                if section["Diameter"].notna().sum() < 3:
                    continue
                tables.append(
                    {
                        "sheet": batch["source_sheet"],
                        "section": section_name,
                        "family": batch["instrument_family"],
                        "title": batch["instrument_name"],
                        "units": batch["units"] or "",
                        "kind": "bore",
                        "oal": oal,
                        "frame": section.reset_index(drop=True),
                    }
                )
        toneholes = conn.execute(
            """
            SELECT b.source_sheet, i.instrument_name, i.instrument_family, i.units,
                t.keyhole, t.d_bottom_bell, t.d_top_reedseat, t.d_ctr_bell,
                t.d_axial, t.d_xverse, t.chimney, t.ch_diam
            FROM tonehole_measurements t
            JOIN measurement_batches b ON b.id = t.batch_id
            JOIN instruments i ON i.id = b.instrument_id
            WHERE b.set_id = %s AND b.status = 'committed'
            ORDER BY b.source_sheet, t.row_index
            """,
            (set_row["id"],),
        ).fetchall()
        if toneholes:
            tone_frame = pd.DataFrame(toneholes)
            rename = {
                "d_bottom_bell": "d-bottom-bell",
                "d_top_reedseat": "d-top-reedseat",
                "d_ctr_bell": "d-ctr-bell",
                "d_axial": "d-axial",
                "d_xverse": "d-xverse",
                "ch_diam": "ch_diam",
            }
            tone_frame = tone_frame.rename(columns=rename)
            for column in rename.values():
                if column in tone_frame.columns:
                    tone_frame[column] = pd.to_numeric(tone_frame[column], errors="coerce")
            for sheet_name, section in tone_frame.groupby("source_sheet", sort=False):
                sample = section.iloc[0]
                tables.append(
                    {
                        "sheet": sheet_name,
                        "section": "toneholes",
                        "family": sample["instrument_family"],
                        "title": sample["instrument_name"],
                        "units": sample["units"] or "",
                        "kind": "toneholes",
                        "frame": section.drop(
                            columns=["source_sheet", "instrument_name", "instrument_family", "units"]
                        ).reset_index(drop=True),
                    }
                )
    return dict(set_row), tables
