"""Chat orchestrator. An LLM calls the existing agents; the CLI stays available.

Manual file edits pause the chat until the user says the workbook is saved.
EDA asks for one-off observations and writes them into that report only.
Validation and database commits stay in the existing Python graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph.types import Command

from agents.bore_optimizer import optimize_bore_set, optimize_bore_source
from agents.eda_execute import execute_eda, execute_eda_set
from agents.orchestrator import graph_session, thread_has_checkpoint
from agents.tonehole_agent import tonehole_agent_set, tonehole_agent_source
from config import ROOT, TEMPLATE_PATH, XAI_API_KEY, XAI_BASE_URL, XAI_MODEL

_SYSTEM = """You are the workshop orchestrator for uilleann pipe measurements.
Talk with the user and call tools to run the existing agents. Keep replies short.
Do not invent measurements, and do not say a check passed unless a tool result says so.

Intake:
- Start intake only when the user names a workbook, and a sheet or all sheets.
- When a tool result is paused for review, explain the issues and the file path.
- If the user will edit the workbook, wait. Do not call resume_intake with correct until a later message says the edits are saved.
- Approve or reject as soon as the user says so.
- When the pause is a database commit or a post-commit check, ask for an explicit yes, then resume with commit or verify.

EDA:
- Before the first run_eda call for a request, the tool will tell you to ask about special observations.
- Ask whether this report should include extra notes, including one-off comparisons that should stay out of the permanent method.
- After the user answers, call run_eda. Pass their notes, or an empty string if they have none.

The bore optimizer and the tonehole model run when asked. The command-line interface still works; mention a command only if the user asks for it.
"""

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_sets",
            "description": "List instrument sets stored in PostgreSQL.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sheets",
            "description": "List worksheet names in a workbook.",
            "parameters": {
                "type": "object",
                "properties": {"source_path": {"type": "string"}},
                "required": ["source_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_intake",
            "description": "Start workbook intake. Pauses for the user when a check or a commit needs a decision.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_path": {"type": "string"},
                    "sheet_name": {"type": "string"},
                    "all_sheets": {"type": "boolean"},
                    "set_name": {"type": "string"},
                },
                "required": ["source_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resume_intake",
            "description": (
                "Continue a paused intake. correct re-reads the workbook and is allowed "
                "only after the user has confirmed the manual edits are saved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["approve", "reject", "correct", "commit", "verify"],
                    },
                    "notes": {"type": "string"},
                },
                "required": ["decision"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_eda",
            "description": "Run the EDA report after the user has answered the observations question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "set_name": {"type": "string"},
                    "source_path": {"type": "string"},
                    "format": {"type": "string", "enum": ["html", "pdf", "both"]},
                    "observations": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bore_optimizer",
            "description": "Calculate the area-equivalent bore design workbook and HTML figure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "set_name": {"type": "string"},
                    "source_path": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tonehole_model",
            "description": "Train the sparse tonehole models and write the acoustic report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "set_name": {"type": "string"},
                    "source_path": {"type": "string"},
                    "format": {"type": "string", "enum": ["html", "pdf", "both"]},
                },
                "additionalProperties": False,
            },
        },
    },
]


def _path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _interrupt_value(result: dict[str, Any] | None) -> dict[str, Any]:
    interrupts = (result or {}).get("__interrupt__") or []
    if not interrupts:
        return {}
    value = getattr(interrupts[0], "value", None)
    if isinstance(value, dict):
        return value
    if isinstance(interrupts[0], dict):
        inner = interrupts[0].get("value", interrupts[0])
        return inner if isinstance(inner, dict) else {}
    return {}


def _pause_kind(info: dict[str, Any]) -> str:
    prompt = str(info.get("prompt") or "").lower()
    resume_format = info.get("resume_format") or {}
    if "post_commit_verified" in resume_format or "post-commit" in prompt:
        return "post_commit"
    if "commit_approved" in resume_format or "committing" in prompt:
        return "commit"
    return "review"


def _edits_sound_done(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "saved",
        "done",
        "finished",
        "i edited",
        "i have edited",
        "edits are",
        "file is updated",
        "corrected the",
        "ready to continue",
        "you can continue",
        "re-read",
        "reread",
    )
    return any(marker in lowered for marker in markers)


class ChatSession:
    """Remembers the intake thread and the two waits the model must honor."""

    def __init__(self) -> None:
        self.thread_id = "chat-intake-1"
        self.thread_serial = 1
        self.awaiting_decision = False
        self.decision_confirmed = False
        self.awaiting_edit = False
        self.edit_confirmed = False
        self.eda_asked = False
        self.eda_reply_ready = False
        self.last_user_message = ""

    def begin_user_turn(self, text: str) -> None:
        self.last_user_message = text
        if self.awaiting_decision:
            self.decision_confirmed = True
        if self.awaiting_edit:
            self.edit_confirmed = True
        if self.eda_asked:
            self.eda_reply_ready = True

    def correct_is_allowed(self) -> bool:
        if not self.decision_confirmed:
            return False
        if self.awaiting_edit and not self.edit_confirmed:
            return False
        return _edits_sound_done(self.last_user_message)

    def _describe(self, result: dict[str, Any]) -> dict[str, Any]:
        info = _interrupt_value(result)
        if not info and not result.get("__interrupt__"):
            self.awaiting_decision = False
            self.decision_confirmed = False
            self.awaiting_edit = False
            self.edit_confirmed = False
            return {
                "ok": True,
                "paused": False,
                "status": result.get("status"),
                "messages": result.get("messages") or [],
                "batch_id": result.get("batch_id"),
                "sheet": result.get("sheet_name"),
            }
        kind = _pause_kind(info)
        self.awaiting_decision = True
        self.decision_confirmed = False
        self.awaiting_edit = kind == "review"
        self.edit_confirmed = False
        return {
            "ok": True,
            "paused": True,
            "kind": kind,
            "thread_id": self.thread_id,
            "prompt": info.get("prompt"),
            "sheet": info.get("sheet") or result.get("sheet_name"),
            "source": info.get("source") or info.get("source_path") or result.get("source_path"),
            "issues": (info.get("issues") or [])[:12],
            "remaining_sheets": info.get("remaining_sheets") or [],
            "batch_id": info.get("batch_id") or result.get("batch_id"),
        }

    def start_intake(
        self,
        source_path: str,
        sheet_name: str | None = None,
        all_sheets: bool = False,
        set_name: str | None = None,
    ) -> dict[str, Any]:
        source = _path(source_path)
        if not source.exists():
            return {"ok": False, "error": f"Workbook not found: {source}"}
        config = {"configurable": {"thread_id": self.thread_id}}
        with graph_session() as app:
            if thread_has_checkpoint(app.checkpointer, self.thread_id):
                snapshot = app.get_state(config)
                pending = bool(getattr(snapshot, "next", None)) or bool(
                    getattr(snapshot, "interrupts", None)
                )
                if pending:
                    return {
                        "ok": False,
                        "error": (
                            f"Thread {self.thread_id} is already paused. "
                            "Resume it, or ask to start a fresh intake."
                        ),
                        "thread_id": self.thread_id,
                    }
                self.thread_serial += 1
                self.thread_id = f"chat-intake-{self.thread_serial}"
                config = {"configurable": {"thread_id": self.thread_id}}
            initial = {
                "source_path": str(source),
                "sheet_name": None if all_sheets else sheet_name,
                "all_sheets": bool(all_sheets),
                "sheet_queue": [],
                "sheet_index": 0,
                "template_path": str(TEMPLATE_PATH),
                "issues": [],
                "messages": [],
                "needs_human_review": False,
                "commit_approved": False,
                "post_commit_verified": False,
                "status": "started",
                "set_name": set_name,
            }
            result = app.invoke(initial, config=config)
        described = self._describe(result if isinstance(result, dict) else {})
        described["thread_id"] = self.thread_id
        described["source"] = described.get("source") or str(source)
        return described

    def resume_intake(self, decision: str, notes: str = "") -> dict[str, Any]:
        choice = decision.lower().strip()
        if self.awaiting_decision and not self.decision_confirmed:
            return {
                "ok": False,
                "waiting_for_user": True,
                "thread_id": self.thread_id,
                "message": "Ask the user before continuing this intake.",
            }
        if choice == "correct" and not self.correct_is_allowed():
            self.awaiting_edit = True
            self.edit_confirmed = False
            return {
                "ok": False,
                "waiting_for_edit": True,
                "thread_id": self.thread_id,
                "message": (
                    "Do not continue yet. Tell the user which file to edit and wait "
                    "until a later message says the edits are saved."
                ),
            }
        payload: dict[str, Any] = {"human_notes": notes}
        if choice in {"approve", "reject", "correct"}:
            payload["human_decision"] = choice
        elif choice == "commit":
            payload["commit_approved"] = True
            payload["human_decision"] = "approve"
        elif choice == "verify":
            payload["post_commit_verified"] = True
        else:
            return {"ok": False, "error": f"Unknown decision: {decision}"}
        config = {"configurable": {"thread_id": self.thread_id}}
        with graph_session() as app:
            if not thread_has_checkpoint(app.checkpointer, self.thread_id):
                return {"ok": False, "error": "No intake is waiting. Start one first."}
            snapshot = app.get_state(config)
            pending = bool(getattr(snapshot, "next", None)) or bool(
                getattr(snapshot, "interrupts", None)
            )
            if not pending:
                return {"ok": False, "error": "That intake is not waiting for a decision."}
            result = app.invoke(Command(resume=payload), config=config)
        self.awaiting_edit = False
        self.edit_confirmed = False
        described = self._describe(result if isinstance(result, dict) else {})
        described["thread_id"] = self.thread_id
        return described

    def run_eda(
        self,
        set_name: str | None = None,
        source_path: str | None = None,
        report_format: str = "html",
        observations: str = "",
    ) -> dict[str, Any]:
        if not self.eda_reply_ready:
            self.eda_asked = True
            self.eda_reply_ready = False
            return {
                "ok": False,
                "need_observations": True,
                "message": (
                    "Ask whether this EDA report should include special observations. "
                    "After the user answers, call run_eda again with their notes, "
                    "or an empty observations string if they have none."
                ),
            }
        formats = {"html", "pdf"} if report_format == "both" else {report_format or "html"}
        notes = [line.strip() for line in observations.splitlines() if line.strip()]
        if set_name:
            paths = execute_eda_set(set_name, formats, observations=notes)
        elif source_path:
            paths = execute_eda(_path(source_path), formats, observations=notes)
        else:
            return {"ok": False, "error": "Provide a set name or a workbook path."}
        self.eda_asked = False
        self.eda_reply_ready = False
        return {"ok": True, "paths": [str(path) for path in paths], "observations": notes}

    def run_bore_optimizer(
        self, set_name: str | None = None, source_path: str | None = None
    ) -> dict[str, Any]:
        if set_name:
            paths = optimize_bore_set(set_name)
        elif source_path:
            paths = optimize_bore_source(_path(source_path))
        else:
            return {"ok": False, "error": "Provide a set name or a workbook path."}
        return {"ok": True, "paths": [str(path) for path in paths]}

    def run_tonehole_model(
        self,
        set_name: str | None = None,
        source_path: str | None = None,
        report_format: str = "html",
    ) -> dict[str, Any]:
        formats = {"html", "pdf"} if report_format == "both" else {report_format or "html"}
        if set_name:
            paths = tonehole_agent_set(set_name, formats)
        elif source_path:
            paths = tonehole_agent_source(_path(source_path), formats)
        else:
            return {"ok": False, "error": "Provide a set name or a workbook path."}
        return {"ok": True, "paths": [str(path) for path in paths]}


def _tool_arguments(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def dispatch_tool(session: ChatSession, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        if name == "list_sets":
            from db.sets import list_sets

            return {"ok": True, "sets": list_sets()}
        if name == "list_sheets":
            from agents.intake_agent import list_workbook_sheets

            return {"ok": True, "sheets": list_workbook_sheets(_path(arguments["source_path"]))}
        if name == "start_intake":
            return session.start_intake(
                arguments["source_path"],
                arguments.get("sheet_name"),
                bool(arguments.get("all_sheets")),
                arguments.get("set_name"),
            )
        if name == "resume_intake":
            return session.resume_intake(arguments["decision"], arguments.get("notes") or "")
        if name == "run_eda":
            return session.run_eda(
                arguments.get("set_name"),
                arguments.get("source_path"),
                arguments.get("format") or "html",
                arguments.get("observations") or "",
            )
        if name == "run_bore_optimizer":
            return session.run_bore_optimizer(arguments.get("set_name"), arguments.get("source_path"))
        if name == "run_tonehole_model":
            return session.run_tonehole_model(
                arguments.get("set_name"),
                arguments.get("source_path"),
                arguments.get("format") or "html",
            )
        return {"ok": False, "error": f"Unknown tool: {name}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _run_turn(client: Any, session: ChatSession, messages: list[dict[str, Any]]) -> str:
    for _round in range(8):
        completion = client.chat.completions.create(
            model=XAI_MODEL,
            messages=messages,
            tools=_TOOLS,
            temperature=0.2,
        )
        message = completion.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            text = message.content or ""
            messages.append({"role": "assistant", "content": text})
            return text
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ],
            }
        )
        for call in tool_calls:
            name = call.function.name
            print(f"\n[running {name}]")
            result = dispatch_tool(session, name, _tool_arguments(call.function.arguments))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                }
            )
    return "I stopped after several tool calls. Tell me how you want to continue."


def chat_loop() -> int:
    """Read a line at a time and let the model call the workshop tools."""
    if not XAI_API_KEY:
        print("Set XAI_API_KEY in .env before starting the chat orchestrator.")
        return 1
    from openai import OpenAI

    client = OpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL, timeout=120.0)
    session = ChatSession()
    messages: list[dict[str, Any]] = [{"role": "system", "content": _SYSTEM}]
    print("Workshop chat. Ask for intake, EDA, the bore design, or the tonehole model.")
    print("Type exit to leave. The command-line tools are unchanged.")
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            return 0
        session.begin_user_turn(user_text)
        messages.append({"role": "user", "content": user_text})
        try:
            reply = _run_turn(client, session, messages)
        except Exception as exc:
            print(f"\nAgent: The request failed: {exc}")
            continue
        print(f"\nAgent: {reply}")
