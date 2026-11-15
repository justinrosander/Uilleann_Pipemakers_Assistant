"""Human-in-the-loop gates for failed checks and commit approval."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from agents.state import IntakeState
from db.writer import queue_human_review


def _sheet_progress(state: IntakeState) -> str:
    queue = state.get("sheet_queue") or []
    if not queue:
        return ""
    index = int(state.get("sheet_index") or 0)
    name = state.get("sheet_name") or ""
    return f"Sheet {index + 1} of {len(queue)}: {name}. "


def human_review_node(state: IntakeState) -> dict[str, Any]:
    """Pause for operator review when validation/sanity fails."""
    issues = list(state.get("issues") or [])
    parsed = state.get("parsed") or {}
    source = parsed.get("source_filename") or state.get("source_path") or ""
    sheet = parsed.get("source_sheet") or state.get("sheet_name") or ""

    review_id = queue_human_review(
        batch_id=state.get("batch_id"),
        issue_stage="validation_or_sanity",
        issues=issues,
        source_filename=str(source),
        source_sheet=str(sheet),
    )

    decision = interrupt(
        {
            "prompt": (
                _sheet_progress(state)
                + "Automated checks failed or need confirmation. "
                "Inspect the issues, optionally correct the source sheet, then resume."
            ),
            "sheet_progress": _sheet_progress(state),
            "remaining_sheets": (state.get("sheet_queue") or [])[
                int(state.get("sheet_index") or 0) + 1 :
            ],
            "review_id": review_id,
            "issues": issues,
            "sheet": sheet,
            "source": source,
            "actions": {
                "approve": "Accept data as-is and continue toward DB commit",
                "reject": "Reject this intake batch",
                "correct": (
                    "I corrected the source workbook on disk; "
                    "re-run intake/validation/sanity"
                ),
            },
            "resume_format": {
                "human_decision": "approve | reject | correct",
                "human_notes": "optional string",
                "human_corrections": "optional object merged into parsed payload",
            },
            "source_path": state.get("source_path"),
        }
    )

    # decision is whatever the operator supplies via Command(resume=...)
    if not isinstance(decision, dict):
        decision = {"human_decision": str(decision)}

    human_decision = str(decision.get("human_decision", "reject")).lower()
    human_notes = decision.get("human_notes") or ""
    corrections = decision.get("human_corrections") or {}

    updates: dict[str, Any] = {
        "review_id": review_id,
        "human_decision": human_decision,
        "human_notes": human_notes,
        "messages": [f"Human review decision: {human_decision}"],
    }

    if human_decision == "correct":
        # Prefer re-reading the corrected file from source_path.
        # Optional human_corrections can still patch the previous parse.
        if corrections and parsed:
            merged = dict(parsed)
            merged.update(corrections)
            updates["parsed"] = merged
        updates["needs_human_review"] = False
        updates["status"] = "revalidate"
        updates["issues"] = []
        updates["validation_report"] = {}
        updates["sanity_report"] = {}
        updates["messages"] = [
            "Human marked corrections; re-running intake from source file."
        ]
    elif human_decision == "approve":
        updates["needs_human_review"] = False
        updates["status"] = "human_approved"
    else:
        updates["needs_human_review"] = False
        updates["status"] = "rejected"
        updates["messages"] = [f"Human rejected intake. Notes: {human_notes}"]

    return updates


def commit_approval_node(state: IntakeState) -> dict[str, Any]:
    """Human must approve before PostgreSQL commit."""
    parsed = state.get("parsed") or {}
    decision = interrupt(
        {
            "prompt": (
                _sheet_progress(state)
                + "Validation and sanity checks passed. "
                "Approve committing this sheet into up_db?"
            ),
            "sheet_progress": _sheet_progress(state),
            "remaining_sheets": (state.get("sheet_queue") or [])[
                int(state.get("sheet_index") or 0) + 1 :
            ],
            "sheet": parsed.get("source_sheet"),
            "source": parsed.get("source_filename"),
            "row_counts": {
                s.get("name"): len(s.get("rows") or [])
                for s in (parsed.get("sections") or [])
            },
            "validation_report": state.get("validation_report"),
            "sanity_report": state.get("sanity_report"),
            "resume_format": {
                "commit_approved": True,
                "human_notes": "optional string",
            },
        }
    )
    if not isinstance(decision, dict):
        decision = {"commit_approved": bool(decision)}

    decision_name = str(decision.get("human_decision", "")).lower()
    # Accept either flag. --decision approve is the same intent as --commit-approved.
    approved = bool(decision.get("commit_approved", False)) or decision_name == "approve"
    if decision_name == "reject":
        approved = False
    notes = decision.get("human_notes") or state.get("human_notes") or ""
    return {
        "commit_approved": approved,
        "human_notes": notes,
        "status": "commit_approved" if approved else "commit_rejected",
        "messages": [
            "Human approved DB commit."
            if approved
            else (
                "Human declined DB commit. "
                "To approve, resume with --commit-approved or --decision approve."
            )
        ],
    }


def post_commit_verify_node(state: IntakeState) -> dict[str, Any]:
    """Human verifies rows after commit."""
    decision = interrupt(
        {
            "prompt": (
                _sheet_progress(state)
                + "Batch was written to PostgreSQL. "
                "Verify the committed rows in up_db and confirm."
            ),
            "sheet_progress": _sheet_progress(state),
            "remaining_sheets": (state.get("sheet_queue") or [])[
                int(state.get("sheet_index") or 0) + 1 :
            ],
            "batch_id": state.get("batch_id"),
            "resume_format": {
                "post_commit_verified": True,
                "human_notes": "optional string",
            },
        }
    )
    if not isinstance(decision, dict):
        decision = {"post_commit_verified": bool(decision)}

    verified = bool(decision.get("post_commit_verified", False))
    notes = decision.get("human_notes") or ""
    return {
        "post_commit_verified": verified,
        "human_notes": notes or state.get("human_notes"),
        "status": "complete" if verified else "post_commit_unverified",
        "messages": [
            f"Post-commit verification {'confirmed' if verified else 'not confirmed'} "
            f"for batch_id={state.get('batch_id')}."
        ],
    }
