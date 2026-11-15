"""Database agent: commit validated sheets into up_db."""

from __future__ import annotations

from typing import Any

from agents.state import IntakeState
from db.writer import commit_batch


def db_commit_node(state: IntakeState) -> dict[str, Any]:
    parsed = state.get("parsed")
    if not parsed:
        return {
            "status": "failed",
            "messages": ["DB commit aborted: no parsed payload."],
        }
    if not state.get("commit_approved"):
        return {
            "status": "commit_rejected",
            "messages": ["DB commit aborted: human did not approve."],
        }

    reports = {
        "validation": state.get("validation_report") or {},
        "sanity": state.get("sanity_report") or {},
        "human_notes": state.get("human_notes"),
    }
    batch_id = commit_batch(parsed, reports)
    messages = [f"Committed sheet to up_db as measurement_batches.id={batch_id}"]
    set_name = state.get("set_name")
    if set_name:
        from db.sets import assign_batch_to_set

        set_id = assign_batch_to_set(
            batch_id,
            str(set_name),
            parsed.get("pipemaker"),
            parsed.get("musical_key"),
        )
        messages.append(f"Assigned batch {batch_id} to instrument set {set_name!r} (id={set_id}).")
    return {
        "batch_id": batch_id,
        "status": "committed",
        "messages": messages,
    }
