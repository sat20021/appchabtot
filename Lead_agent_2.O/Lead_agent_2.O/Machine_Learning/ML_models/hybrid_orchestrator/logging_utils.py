"""
Centralized logging helpers for the hybrid orchestrator.

Writes structured task-level logs to `logs/hybrid_orchestrator.log` so that every
step (classifier call, routing decision, LangGraph handoff, etc.) is easy to audit.
"""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_FILE = LOG_DIR / "hybrid_orchestrator.log"

_LOGGER: Optional[logging.Logger] = None


def _configure_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER:
        return _LOGGER

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hybrid_orchestrator")
    logger.setLevel(logging.INFO)

    if not any(isinstance(handler, RotatingFileHandler) for handler in logger.handlers):
        handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _LOGGER = logger
    return logger


def log_task(task_name: str, status: str, **context: Any):
    """
    Emit a JSON-structured log entry describing an orchestrator task.

    Args:
        task_name: Logical name (e.g., "classifier.predict", "llm.route").
        status: Arbitrary string such as "start", "success", "error".
        context: Extra metadata (session_id, intent, latency, error, etc.).
    """
    logger = _configure_logger()
    payload: Dict[str, Any] = {
        "task": task_name,
        "status": status,
        **context,
    }
    logger.info(json.dumps(payload, default=str))

