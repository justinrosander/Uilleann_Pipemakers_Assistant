"""Remember accepted preprocess column mappings and retrieve them later.

This is a feedback harness, not model fine-tuning. Each accepted mapping is
appended to data/preprocess_memory/mappings.jsonl. Later suggestions load the
closest examples and pass them to the heuristic and, when configured, the LLM.
"""

from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import MAPPING_MEMORY_PATH, PROCESSED_DIR


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _tokens(value: object) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value).lower()))


def _sample_values(series: pd.Series, limit: int = 4) -> list[str]:
    samples: list[str] = []
    for value in series.tolist():
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        samples.append(text[:40])
        if len(samples) >= limit:
            break
    return samples


def load_examples(path: Path | None = None) -> list[dict[str, Any]]:
    memory_path = path or MAPPING_MEMORY_PATH
    if not memory_path.exists():
        return []
    examples: list[dict[str, Any]] = []
    for line in memory_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("source_column") and item.get("target_column"):
            examples.append(item)
    return examples


def record_mappings(
    *,
    frame: pd.DataFrame,
    column_map: dict[str, str],
    sheet_name: str,
    section_name: str | None,
    source_block: str,
    pipemaker: str = "",
    path: Path | None = None,
) -> int:
    """Append one memory row per accepted source-to-target mapping."""
    memory_path = path or MAPPING_MEMORY_PATH
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    seen_targets: set[str] = set()
    written = 0
    stamp = datetime.now(timezone.utc).isoformat()
    with memory_path.open("a", encoding="utf-8") as handle:
        for source_column, target_column in column_map.items():
            if source_column.startswith("__"):
                continue
            append = target_column in seen_targets
            seen_targets.add(target_column)
            if source_column not in frame.columns:
                samples: list[str] = []
            else:
                samples = _sample_values(frame[source_column])
            record = {
                "recorded_at": stamp,
                "source_column": source_column,
                "target_column": target_column,
                "sheet_name": sheet_name,
                "section_name": section_name or "main",
                "source_block": source_block,
                "pipemaker": pipemaker,
                "sample_values": samples,
                "append": append,
                "accepted": True,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def _score(example: dict[str, Any], source_column: str, sheet_name: str | None) -> float:
    score = 0.0
    if _normalize(example.get("source_column")) == _normalize(source_column):
        score += 10.0
    left = _tokens(source_column)
    right = _tokens(example.get("source_column"))
    if left and right:
        score += 5.0 * (len(left & right) / len(left | right))
    if sheet_name and example.get("sheet_name") == sheet_name:
        score += 2.0
    return score


def retrieve_examples(
    source_columns: list[str],
    *,
    sheet_name: str | None = None,
    targets: tuple[str, ...] | list[str] | None = None,
    limit: int = 12,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the closest accepted mappings for these source columns."""
    examples = load_examples(path)
    if not examples:
        return []
    allowed = set(targets) if targets else None
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for index, example in enumerate(examples):
        target = example.get("target_column")
        if allowed is not None and target not in allowed:
            continue
        best = 0.0
        for column in source_columns:
            best = max(best, _score(example, column, sheet_name))
        if best <= 0:
            continue
        ranked.append((best, index, example))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [example for _, _, example in ranked[:limit]]


def memory_column_map(
    source_columns: list[str],
    targets: tuple[str, ...],
    *,
    sheet_name: str | None = None,
    path: Path | None = None,
) -> dict[str, str]:
    """Suggest mappings from prior accepted choices. Later examples win ties."""
    examples = load_examples(path)
    if not examples:
        return {}
    suggestions: dict[str, str] = {}
    used_targets: set[str] = set()
    # Walk oldest to newest so a newer exact match replaces an older one.
    for example in examples:
        target = example.get("target_column")
        if target not in targets or target in used_targets:
            continue
        source_norm = _normalize(example.get("source_column"))
        for column in source_columns:
            if column in suggestions:
                continue
            if _normalize(column) == source_norm or _score(example, column, sheet_name) >= 8:
                suggestions[column] = str(target)
                used_targets.add(str(target))
                break
    return suggestions


def format_examples_for_prompt(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "No previous accepted mappings."
    lines = []
    for example in examples:
        samples = ", ".join(example.get("sample_values") or [])
        append = " (appended below existing values)" if example.get("append") else ""
        lines.append(
            f"- source '{example.get('source_column')}' -> "
            f"'{example.get('target_column')}' on {example.get('sheet_name')}"
            f"/{example.get('section_name')}{append}"
            + (f"; samples: {samples}" if samples else "")
        )
    return "\n".join(lines)


def import_mapping_logs(processed_dir: Path | None = None, path: Path | None = None) -> int:
    """One-time bootstrap from existing .mapping.txt files when memory is empty."""
    memory_path = path or MAPPING_MEMORY_PATH
    if load_examples(memory_path):
        return 0
    folder = processed_dir or PROCESSED_DIR
    if not folder.exists():
        return 0
    imported = 0
    pattern = re.compile(
        r"^\s*(?P<sheet>.+?)(?:\s+\(A1=.*?\))?\s*/\s*(?P<section>\S+)\s*:\s*(?P<mapping>\{.*\})\s*$"
    )
    for log_path in sorted(folder.glob("*.mapping.txt")):
        for line in log_path.read_text(encoding="utf-8").splitlines():
            match = pattern.match(line)
            if not match:
                continue
            try:
                column_map = ast.literal_eval(match.group("mapping"))
            except (SyntaxError, ValueError):
                continue
            if not isinstance(column_map, dict):
                continue
            frame = pd.DataFrame({str(key): [] for key in column_map})
            imported += record_mappings(
                frame=frame,
                column_map={str(key): str(value) for key, value in column_map.items()},
                sheet_name=match.group("sheet").strip(),
                section_name=match.group("section").strip(),
                source_block=log_path.name,
                path=memory_path,
            )
    return imported
