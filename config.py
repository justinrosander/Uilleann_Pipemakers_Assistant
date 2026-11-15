"""Runtime configuration for the UP measurement intake pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user@localhost:5432/up_db",
)
TEMPLATE_PATH = Path(
    os.getenv("TEMPLATE_PATH", str(ROOT / "UP_measurement_template.xlsx"))
)
if not TEMPLATE_PATH.is_absolute():
    TEMPLATE_PATH = ROOT / TEMPLATE_PATH

CHECKPOINT_PATH = Path(
    os.getenv("CHECKPOINT_PATH", str(ROOT / "data" / "langgraph_checkpoints.sqlite"))
)
if not CHECKPOINT_PATH.is_absolute():
    CHECKPOINT_PATH = ROOT / CHECKPOINT_PATH
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MAPPING_MEMORY_PATH = ROOT / "data" / "preprocess_memory" / "mappings.jsonl"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MAPPING_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)

XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4.7")

# Nominal is stored to 2 decimal places. Extra source digits are rounded
# at commit time; the sanity check compares that rounded value.
