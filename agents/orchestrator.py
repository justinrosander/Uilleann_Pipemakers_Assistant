"""LangGraph orchestrator for UP measurement intake.

Flow:
  intake -> validate -> (human_review)? -> sanity -> (human_review)?
         -> commit_approval -> db_commit -> post_commit_verify -> END

EDA / ML / DL agents are intentionally not wired in this phase.
Hallucination tolerance is zero: checks are deterministic code.

Checkpoints are persisted to SQLite so --resume works across CLI processes.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from agents.db_agent import db_commit_node
from agents.human_review_agent import (
    commit_approval_node,
    human_review_node,
    post_commit_verify_node,
)
from agents.intake_agent import intake_node
from agents.sanity_agent import sanity_node
from agents.state import IntakeState
from agents.validation_agent import validation_node
from config import CHECKPOINT_PATH, TEMPLATE_PATH


def _route_after_intake(state: IntakeState) -> Literal["validate", "human_review", "end"]:
    status = state.get("status")
    if status in {"failed"}:
        return "end"
    if state.get("needs_human_review") and status != "parsed":
        return "human_review"
    return "validate"


def _route_after_validation(
    state: IntakeState,
) -> Literal["sanity", "human_review", "end"]:
    if state.get("status") == "validated" and (state.get("validation_report") or {}).get(
        "ok"
    ):
        return "sanity"
    if state.get("status") == "rejected":
        return "end"
    return "human_review"


def _route_after_sanity(
    state: IntakeState,
) -> Literal["commit_approval", "human_review", "end"]:
    if state.get("status") == "sanity_ok" and (state.get("sanity_report") or {}).get("ok"):
        return "commit_approval"
    if state.get("status") == "rejected":
        return "end"
    return "human_review"


def _route_after_human(
    state: IntakeState,
) -> Literal["intake", "validate", "sanity", "commit_approval", "advance_sheet", "end"]:
    decision = (state.get("human_decision") or "").lower()
    status = state.get("status")
    if decision == "reject" or status == "rejected":
        return "advance_sheet" if state.get("all_sheets") else "end"
    # After correcting the source file, re-parse then validate/sanity.
    if decision == "correct" or status == "revalidate":
        return "intake"
    if decision == "approve" or status == "human_approved":
        # Accept the reported issues and move forward. Do not re-run the
        # check that just failed. Sanity still runs if it has not run yet.
        validation_ok = (state.get("validation_report") or {}).get("ok")
        sanity_report = state.get("sanity_report") or {}
        sanity_ran = "ok" in sanity_report
        if validation_ok and sanity_report.get("ok"):
            return "commit_approval"
        if not validation_ok and not sanity_ran:
            return "sanity"
        return "commit_approval"
    return "end"


def _route_after_commit_approval(
    state: IntakeState,
) -> Literal["db_commit", "advance_sheet", "end"]:
    if state.get("commit_approved"):
        return "db_commit"
    return "advance_sheet" if state.get("all_sheets") else "end"


def _route_after_post_commit(
    state: IntakeState,
) -> Literal["advance_sheet", "end"]:
    return "advance_sheet" if state.get("all_sheets") else "end"


def _route_after_advance(
    state: IntakeState,
) -> Literal["intake", "end"]:
    if state.get("status") == "next_sheet":
        return "intake"
    return "end"


def _route_after_select(
    state: IntakeState,
) -> Literal["intake", "end"]:
    if state.get("status") == "failed":
        return "end"
    return "intake"


def select_sheet_node(state: IntakeState) -> dict[str, Any]:
    """In workbook mode, queue every template sheet. Otherwise leave the sheet as given."""
    if not state.get("all_sheets"):
        return {"status": state.get("status") or "started"}

    from agents.intake_agent import list_workbook_sheets
    from schema.template_schema import classify_sheet

    names = [
        name
        for name in list_workbook_sheets(state["source_path"])
        if classify_sheet(name) is not None
    ]
    if not names:
        return {
            "status": "failed",
            "sheet_queue": [],
            "issues": [
                {
                    "stage": "intake",
                    "code": "no_template_sheets",
                    "message": "No template sheets were found in the workbook.",
                }
            ],
            "messages": ["Workbook mode found no template sheets."],
        }
    return {
        "sheet_queue": names,
        "sheet_index": 0,
        "sheet_name": names[0],
        "status": "started",
        "messages": [
            f"Workbook mode: {len(names)} template sheets. "
            f"Starting with {names[0]}."
        ],
    }


def advance_sheet_node(state: IntakeState) -> dict[str, Any]:
    """Move to the next template sheet, or finish the workbook."""
    queue = list(state.get("sheet_queue") or [])
    index = int(state.get("sheet_index") or 0)
    finished = state.get("sheet_name") or ""
    outcome = state.get("status") or "done"
    next_index = index + 1
    if not state.get("all_sheets") or next_index >= len(queue):
        return {
            "status": "workbook_complete" if state.get("all_sheets") else outcome,
            "messages": [
                f"Finished '{finished}' ({outcome}). "
                + (
                    "No further template sheets."
                    if state.get("all_sheets")
                    else "Single-sheet run complete."
                )
            ],
        }
    next_sheet = queue[next_index]
    return {
        "sheet_index": next_index,
        "sheet_name": next_sheet,
        "parsed": {},
        "validation_report": {},
        "sanity_report": {},
        "issues": [],
        "needs_human_review": False,
        "human_decision": "",
        "human_notes": "",
        "commit_approved": False,
        "post_commit_verified": False,
        "batch_id": None,
        "review_id": None,
        "status": "next_sheet",
        "messages": [
            f"Finished '{finished}' ({outcome}). "
            f"Next sheet {next_index + 1} of {len(queue)}: {next_sheet}."
        ],
    }


def build_graph(checkpointer: SqliteSaver):
    graph = StateGraph(IntakeState)

    graph.add_node("select_sheet", select_sheet_node)
    graph.add_node("intake", intake_node)
    graph.add_node("validate", validation_node)
    graph.add_node("sanity", sanity_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("commit_approval", commit_approval_node)
    graph.add_node("db_commit", db_commit_node)
    graph.add_node("post_commit_verify", post_commit_verify_node)
    graph.add_node("advance_sheet", advance_sheet_node)

    graph.set_entry_point("select_sheet")
    graph.add_conditional_edges(
        "select_sheet",
        _route_after_select,
        {"intake": "intake", "end": END},
    )
    graph.add_conditional_edges(
        "intake",
        _route_after_intake,
        {
            "validate": "validate",
            "human_review": "human_review",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "validate",
        _route_after_validation,
        {
            "sanity": "sanity",
            "human_review": "human_review",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "sanity",
        _route_after_sanity,
        {
            "commit_approval": "commit_approval",
            "human_review": "human_review",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "human_review",
        _route_after_human,
        {
            "intake": "intake",
            "validate": "validate",
            "sanity": "sanity",
            "commit_approval": "commit_approval",
            "advance_sheet": "advance_sheet",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "commit_approval",
        _route_after_commit_approval,
        {
            "db_commit": "db_commit",
            "advance_sheet": "advance_sheet",
            "end": END,
        },
    )
    graph.add_edge("db_commit", "post_commit_verify")
    graph.add_conditional_edges(
        "post_commit_verify",
        _route_after_post_commit,
        {"advance_sheet": "advance_sheet", "end": END},
    )
    graph.add_conditional_edges(
        "advance_sheet",
        _route_after_advance,
        {"intake": "intake", "end": END},
    )

    return graph.compile(checkpointer=checkpointer)


@contextmanager
def graph_session() -> Iterator[Any]:
    """Open the persistent SQLite checkpointer and compiled graph."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(CHECKPOINT_PATH)) as checkpointer:
        checkpointer.setup()
        yield build_graph(checkpointer)


def thread_has_checkpoint(checkpointer: SqliteSaver, thread_id: str) -> bool:
    config = {"configurable": {"thread_id": thread_id}}
    return checkpointer.get_tuple(config) is not None


def run_intake(
    source_path: str | Path,
    sheet_name: str | None = None,
    *,
    thread_id: str = "up-intake-1",
    auto_approve: bool = False,
) -> dict[str, Any]:
    """
    Run the intake graph.

    When auto_approve is False (default), the graph pauses at interrupt()
    points for human decisions. Use langgraph Command(resume=...) to continue.

    When auto_approve is True, interrupts are not used — for smoke tests only;
    production should keep human gates enabled.
    """
    source_path = Path(source_path)

    initial: IntakeState = {
        "source_path": str(source_path),
        "sheet_name": sheet_name,
        "template_path": str(TEMPLATE_PATH),
        "issues": [],
        "messages": [],
        "needs_human_review": False,
        "commit_approved": False,
        "post_commit_verified": False,
        "status": "started",
    }

    if auto_approve:
        from agents.db_agent import db_commit_node as _commit
        from agents.intake_agent import intake_node as _intake
        from agents.sanity_agent import sanity_node as _sanity
        from agents.validation_agent import validation_node as _validate

        state: dict[str, Any] = dict(initial)
        for node in (_intake, _validate, _sanity):
            updates = node(state)  # type: ignore[arg-type]
            state.update(updates)
            if updates.get("issues") is not None:
                state["issues"] = updates["issues"]
            if not (state.get("validation_report") or {}).get("ok", True):
                break
            if node is _sanity and not (state.get("sanity_report") or {}).get("ok", True):
                break
        else:
            state["commit_approved"] = True
            state.update(_commit(state))  # type: ignore[arg-type]
            state["post_commit_verified"] = True
            state["status"] = "complete"
            state.setdefault("messages", []).append(
                "auto_approve smoke path completed (human gates bypassed)."
            )
        return state

    config = {"configurable": {"thread_id": thread_id}}
    with graph_session() as app:
        return app.invoke(initial, config=config)
