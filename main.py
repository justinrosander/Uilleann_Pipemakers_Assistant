#!/usr/bin/env python
"""CLI entrypoint for the UP measurement LangGraph intake pipeline.

Examples:
  py main.py --source somefile.ods --sheet "Tenor Drone Bore" --thread-id run-1
  py main.py --resume --thread-id run-1 --decision correct
  py main.py --resume --thread-id run-1 --commit-approved
  py main.py --sheet "Chanter Bore" --auto-approve
  py main.py --list-sheets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from langgraph.types import Command

from agents.orchestrator import graph_session, run_intake, thread_has_checkpoint
from agents.eda_agent import run_eda_framework
from agents.bore_optimizer import optimize_bore_set, optimize_bore_source
from agents.tonehole_agent import tonehole_agent_set, tonehole_agent_source
from agents.eda_execute import execute_eda, execute_eda_set
from agents.preprocess_agent import run_preprocess
from config import CHECKPOINT_PATH, ROOT, TEMPLATE_PATH


def _list_sheets(path: Path) -> list[str]:
    from agents.intake_agent import list_workbook_sheets

    return list_workbook_sheets(path)


def _print_state(state: dict[str, Any]) -> None:
    print("\n=== Pipeline status ===")
    print("status:", state.get("status"))
    print(
        "sheet:",
        state.get("sheet_name")
        or (state.get("parsed") or {}).get("source_sheet"),
    )
    queue = state.get("sheet_queue") or []
    if queue:
        index = int(state.get("sheet_index") or 0)
        print(f"workbook: sheet {index + 1} of {len(queue)}")
    print("source_path:", state.get("source_path"))
    print("batch_id:", state.get("batch_id"))
    print("review_id:", state.get("review_id"))
    messages = state.get("messages") or []
    if messages:
        print("messages:")
        for m in messages:
            print(" -", m)
    issues = state.get("issues") or []
    if issues:
        print(f"issues ({len(issues)}):")
        for issue in issues[:25]:
            print(" -", json.dumps(issue, default=str))
        if len(issues) > 25:
            print(f" - ... {len(issues) - 25} more")


def _interrupt_value(result: dict[str, Any]) -> dict[str, Any]:
    interrupts = result.get("__interrupt__") or []
    if not interrupts:
        return {}
    value = getattr(interrupts[0], "value", None)
    return value if isinstance(value, dict) else {}


def _print_resume_help(thread_id: str, result: dict[str, Any] | None = None) -> None:
    info = _interrupt_value(result or {})
    prompt = str(info.get("prompt") or "")
    resume_format = info.get("resume_format") or {}
    print("\nGraph paused for human input.")
    progress = info.get("sheet_progress") or ""
    if progress:
        print(progress.strip())
    if "post-commit" in prompt.lower() or "post_commit_verified" in resume_format:
        print("Confirm the rows in up_db, then:")
        print(
            f"  py main.py --resume --thread-id {thread_id} --post-commit-verified"
        )
    elif "committing" in prompt.lower() or "commit_approved" in resume_format:
        print("Checks passed. Approve writing this sheet to the database with either:")
        print(f"  py main.py --resume --thread-id {thread_id} --commit-approved")
        print(f"  py main.py --resume --thread-id {thread_id} --decision approve")
    else:
        print("Validation or sanity needs a decision:")
        print(f"  py main.py --resume --thread-id {thread_id} --decision approve")
        print("    accept the data despite the issues and continue")
        print(f"  py main.py --resume --thread-id {thread_id} --decision correct")
        print("    you edited the source file; re-read it and run the checks again")
        print(f"  py main.py --resume --thread-id {thread_id} --decision reject")
        if info.get("remaining_sheets"):
            print("    skip this sheet and continue with the next one")
        else:
            print("    stop this intake")
    remaining = info.get("remaining_sheets") or []
    if remaining:
        print("After this sheet is committed or skipped, the run continues with:")
        print("  " + ", ".join(remaining[:8]) + (" ..." if len(remaining) > 8 else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Uilleann pipe measurement intake orchestrator"
    )
    parser.add_argument(
        "--source",
        default=str(TEMPLATE_PATH),
        help=(
            "Path to workbook (.xlsx/.xlsm/.ods). "
            "Default: UP_measurement_template.xlsx"
        ),
    )
    parser.add_argument("--sheet", default=None, help="Worksheet name to ingest")
    parser.add_argument(
        "--eda",
        action="store_true",
        help=(
            "Run bore EDA on --source: statistics, regressions, and plots. "
            "Without --source, print the plan only."
        ),
    )
    parser.add_argument(
        "--optimize-bore",
        action="store_true",
        help=(
            "Calculate the area-equivalent bore design from round and oval probes. "
            "Writes a spreadsheet and an HTML figure. Does not update the database."
        ),
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help=(
            "Talk with the orchestrator. It can run intake, EDA, the bore optimizer, "
            "and the tonehole model. The other commands stay available."
        ),
    )
    parser.add_argument(
        "--tonehole-model",
        action="store_true",
        help=(
            "Predict tonehole placement and dimensions from sparse measurements. "
            "Trains small ANN, MLP, and CNN models and writes separate Benade, "
            "Nederveen, and Lefebvre analyses. HTML, PDF, or both."
        ),
    )
    parser.add_argument(
        "--set",
        default=None,
        help=(
            "Instrument set name. With --eda, load that set from PostgreSQL. "
            "With intake, assign the committed sheet to that set. "
            "With --create-set, the name to create."
        ),
    )
    parser.add_argument(
        "--create-set",
        action="store_true",
        help=(
            "Create or update an instrument set and attach the latest committed "
            "batch for each sheet of --pipemaker and --key."
        ),
    )
    parser.add_argument("--pipemaker", default=None, help="Pipemaker name for --create-set")
    parser.add_argument("--key", default=None, help="Musical key for --create-set, such as D")
    parser.add_argument("--list-sets", action="store_true", help="List instrument sets in up_db")
    parser.add_argument(
        "--format",
        choices=["html", "pdf", "both"],
        default="html",
        help="EDA report format. Use html, pdf, or both. Default: html.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help=(
            "Interactively map a non-standard .csv, .txt, .xlsx, .ods, or .pdf "
            "into a copy of the measurement template. The raw file is copied to "
            "data/raw and the result is saved in data/processed."
        ),
    )
    parser.add_argument(
        "--all-sheets",
        action="store_true",
        help=(
            "Process every template sheet in the workbook, in sheet order. "
            "Each sheet pauses for its own check and commit approval."
        ),
    )
    parser.add_argument(
        "--thread-id",
        default="up-intake-1",
        help="LangGraph thread id for interrupt/resume",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Smoke-test mode: bypass human interrupt gates and commit if checks pass",
    )
    parser.add_argument(
        "--list-sheets",
        action="store_true",
        help="List worksheets in the source workbook and exit",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a paused graph thread with a human decision",
    )
    parser.add_argument(
        "--decision",
        choices=["approve", "reject", "correct"],
        help="Human decision when resuming from human_review",
    )
    parser.add_argument(
        "--commit-approved",
        action="store_true",
        help="When resuming commit_approval, approve the DB commit",
    )
    parser.add_argument(
        "--post-commit-verified",
        action="store_true",
        help="When resuming post_commit_verify, confirm verification",
    )
    parser.add_argument("--notes", default="", help="Human notes for resume payload")
    args = parser.parse_args(argv)

    if args.chat:
        from agents.chat_orchestrator import chat_loop

        return chat_loop()

    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source

    if args.list_sets:
        from db.sets import list_sets

        rows = list_sets()
        if not rows:
            print("No instrument sets yet.")
            return 0
        for row in rows:
            print(
                f"{row['id']}: {row['name']} | {row['pipemaker']} | "
                f"key {row['musical_key']} | {row['batches']} batches"
            )
        return 0

    if args.create_set:
        from db.sets import attach_latest_batches, create_set

        if not args.pipemaker or not args.key:
            print("Create a set with --pipemaker and --key. Optional --set names it.")
            print('  py main.py --create-set --pipemaker "R.L. O\'Mealy" --key D')
            return 1
        set_name = args.set or f"{args.pipemaker} {args.key}"
        set_id = create_set(set_name, args.pipemaker, args.key)
        attached = attach_latest_batches(set_id, args.pipemaker, args.key)
        print(f"Set {set_name!r} (id={set_id}) has {len(attached)} sheet batch(es):")
        for row in attached:
            print(f"  {row['id']}: {row['source_sheet']}")
        return 0

    if args.tonehole_model:
        formats = {"html", "pdf"} if args.format == "both" else {args.format}
        try:
            if args.set:
                paths = tonehole_agent_set(args.set, formats)
            elif "--source" in sys.argv:
                paths = tonehole_agent_source(source, formats)
            else:
                print("The tonehole agent needs a workbook or a database set.")
                print('  py main.py --tonehole-model --set "Example Set" --format html')
                print('  py main.py --tonehole-model --source "set.xlsx" --format pdf')
                return 1
        except Exception as exc:
            print(f"Tonehole model failed: {exc}")
            return 1
        print("Tonehole model written:")
        for path in paths:
            print(f"  {path}")
        return 0

    if args.optimize_bore:
        try:
            if args.set:
                paths = optimize_bore_set(args.set)
            elif "--source" in sys.argv:
                paths = optimize_bore_source(source)
            else:
                print("The bore optimizer needs a workbook or a database set.")
                print('  py main.py --optimize-bore --set "Example Set"')
                print('  py main.py --optimize-bore --source "set.xlsx"')
                return 1
        except Exception as exc:
            print(f"Bore optimizer failed: {exc}")
            return 1
        print("Area-equivalent bore design written:")
        for path in paths:
            print(f"  {path}")
        return 0

    if args.eda:
        formats = {"html", "pdf"} if args.format == "both" else {args.format}
        if args.set:
            try:
                paths = execute_eda_set(args.set, formats)
            except Exception as exc:
                print(f"EDA failed: {exc}")
                return 1
            print("EDA report written:")
            for path in paths:
                print(f"  {path}")
            return 0
        if "--source" not in sys.argv:
            plan = run_eda_framework(None)
            print("\n".join(plan.as_lines()))
            print("\nTo run the analysis, pass a workbook or a database set:")
            print('  py main.py --eda --source "set.xlsx" --format html')
            print('  py main.py --eda --set "R.L. O\'Mealy D" --format pdf')
            return 0
        try:
            paths = execute_eda(source, formats)
        except Exception as exc:
            print(f"EDA failed: {exc}")
            return 1
        print("EDA report written:")
        for path in paths:
            print(f"  {path}")
        return 0

    if args.preprocess:
        if "--source" not in sys.argv:
            print(
                "Preprocess needs the raw file: "
                'py main.py --preprocess --source "path\\to\\measurements.csv"'
            )
            return 1
        try:
            run_preprocess(source)
        except Exception as exc:
            print(f"Preprocess failed: {exc}")
            return 1
        return 0

    if args.list_sheets:
        for name in _list_sheets(source):
            print(name)
        return 0

    if args.resume:
        config = {"configurable": {"thread_id": args.thread_id}}
        payload: dict[str, Any] = {"human_notes": args.notes}
        if args.decision:
            payload["human_decision"] = args.decision
        if args.commit_approved:
            payload["commit_approved"] = True
        if args.post_commit_verified:
            payload["post_commit_verified"] = True

        with graph_session() as app:
            # Access checkpointer via app to verify thread exists
            checkpointer = app.checkpointer
            if checkpointer is None or not thread_has_checkpoint(
                checkpointer, args.thread_id
            ):
                print(
                    f"No checkpoint found for thread-id '{args.thread_id}'.\n"
                    f"Checkpoints are stored in: {CHECKPOINT_PATH}\n"
                    "Start a new intake first, then resume with the same "
                    "--thread-id. Example:\n"
                    f'  py main.py --source yourfile.ods --sheet "Sheet Name" '
                    f"--thread-id {args.thread_id}"
                )
                return 1

            snapshot = app.get_state(config)
            pending = bool(getattr(snapshot, "next", None)) or bool(
                getattr(snapshot, "interrupts", None)
            )
            if not pending:
                status = (snapshot.values or {}).get("status")
                print(
                    f"Thread '{args.thread_id}' is not waiting for input "
                    f"(status: {status}).\n"
                    "That run already finished. Start a new thread id, for example:\n"
                    f'  py main.py --source "{source}" --sheet "Chanter Toneholes" '
                    f"--thread-id {args.thread_id}-2"
                )
                return 1

            result = app.invoke(Command(resume=payload), config=config)

        _print_state(result)
        if isinstance(result, dict) and result.get("__interrupt__"):
            _print_resume_help(args.thread_id, result)
            return 0
        return (
            0
            if result.get("status")
            in {
                "complete",
                "committed",
                "sanity_ok",
                "validated",
                "human_approved",
                "commit_approved",
                "parsed",
            }
            else 1
        )

    if args.auto_approve:
        state = run_intake(
            source,
            sheet_name=args.sheet,
            thread_id=args.thread_id,
            auto_approve=True,
        )
        _print_state(state)
        return 0 if state.get("status") == "complete" else 1

    # Interactive interrupt-aware run (persistent checkpoint)
    config = {"configurable": {"thread_id": args.thread_id}}
    initial = {
        "source_path": str(source),
        "sheet_name": None if args.all_sheets else args.sheet,
        "all_sheets": bool(args.all_sheets),
        "sheet_queue": [],
        "sheet_index": 0,
        "template_path": str(TEMPLATE_PATH),
        "issues": [],
        "messages": [],
        "needs_human_review": False,
        "commit_approved": False,
        "post_commit_verified": False,
        "status": "started",
        "set_name": args.set,
    }
    with graph_session() as app:
        if thread_has_checkpoint(app.checkpointer, args.thread_id):
            snapshot = app.get_state(config)
            pending = bool(getattr(snapshot, "next", None)) or bool(
                getattr(snapshot, "interrupts", None)
            )
            status = (snapshot.values or {}).get("status")
            if pending:
                print(
                    f"Thread '{args.thread_id}' is already paused "
                    f"(status: {status}).\n"
                    "Repeating the start command does not begin a clean run; "
                    "it continues this thread and mixes old messages in.\n"
                    "Resume the decision, or start over with a new --thread-id."
                )
                _print_state(dict(snapshot.values or {}))
                # Rebuild a result-like view so resume help can describe the gate.
                # get_state interrupts may be empty depending on version; still
                # print the standard decision help for a human-review pause.
                fake = {"__interrupt__": getattr(snapshot, "interrupts", None) or []}
                if status == "needs_human_review" or not fake["__interrupt__"]:
                    _print_resume_help(args.thread_id, fake if fake["__interrupt__"] else {
                        "__interrupt__": []
                    })
                    if not fake["__interrupt__"] and status == "needs_human_review":
                        print(
                            "\nThis pause is a failed check. Use --decision "
                            "approve, correct, or reject."
                        )
                else:
                    _print_resume_help(args.thread_id, fake)
                return 1
            print(
                f"Thread '{args.thread_id}' already finished "
                f"(status: {status}).\n"
                "Use a new --thread-id so this intake does not append to "
                "the old history."
            )
            return 1

        if args.all_sheets and args.sheet:
            print(
                "Both --all-sheets and --sheet were given. "
                "Workbook mode will process every template sheet; "
                f'"{args.sheet}" is not used as a single-sheet filter.'
            )

        if args.all_sheets:
            from schema.template_schema import classify_sheet

            known = [name for name in _list_sheets(source) if classify_sheet(name)]
            print(f"Workbook mode: {len(known)} template sheets.")
            for name in known:
                print(f"  - {name}")
        elif args.sheet is None:
            from schema.template_schema import classify_sheet

            names = _list_sheets(source)
            known = [name for name in names if classify_sheet(name)]
            print(
                "No --sheet given. The pipeline will use the first template "
                f"sheet in the workbook: {known[0] if known else '(none found)'}."
            )
            if len(known) > 1:
                print("Other template sheets in this file:")
                for name in known[1:]:
                    print(f"  - {name}")
                print("Pass --sheet to choose one.")

        result = app.invoke(initial, config=config)

    _print_state(result)

    if isinstance(result, dict) and result.get("__interrupt__"):
        _print_resume_help(args.thread_id, result)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
