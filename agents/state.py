"""Shared LangGraph state for the intake pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class Issue(TypedDict, total=False):
    stage: str
    code: str
    message: str
    sheet: str
    section: str
    row: int
    field: str
    value: Any
    expected: Any


class IntakeState(TypedDict, total=False):
    source_path: str
    sheet_name: str | None
    set_name: str | None
    # Workbook mode: template sheets in order, and the one currently being processed.
    all_sheets: bool
    sheet_queue: list[str]
    sheet_index: int
    template_path: str
    parsed: dict[str, Any]
    validation_report: dict[str, Any]
    sanity_report: dict[str, Any]
    # Replaced each node (not appended) so re-validation can clear prior failures
    issues: list[Issue]
    needs_human_review: bool
    human_decision: str  # approve | reject | correct
    human_notes: str
    human_corrections: dict[str, Any]
    commit_approved: bool
    post_commit_verified: bool
    batch_id: int | None
    review_id: int | None
    status: str
    messages: Annotated[list[str], operator.add]
