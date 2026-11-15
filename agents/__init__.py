"""Agents for UP measurement intake and preprocess."""

from __future__ import annotations

from typing import Any

__all__ = ["build_graph", "graph_session", "run_intake", "run_preprocess", "run_eda_framework"]


def __getattr__(name: str) -> Any:
    if name in {"build_graph", "graph_session", "run_intake"}:
        from . import orchestrator

        return getattr(orchestrator, name)
    if name == "run_preprocess":
        from .preprocess_agent import run_preprocess

        return run_preprocess
    if name == "run_eda_framework":
        from .eda_agent import run_eda_framework

        return run_eda_framework
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
