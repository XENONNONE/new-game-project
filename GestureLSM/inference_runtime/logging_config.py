"""Structured logging configuration for the avatar inference runtime.

Provides JSON or human-readable log output with levels, file rotation,
and context injection. Replace bare ``print`` calls throughout the
runtime with ``logger = logging.getLogger("avatar")`` and structured
``logger.info("event", extra={...})`` calls.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_LOG_LEVEL = os.environ.get("AVATAR_LOG_LEVEL", "INFO").upper()
DEFAULT_LOG_FILE = os.environ.get("AVATAR_LOG_FILE", "")
LOG_FORMAT_JSON = os.environ.get("AVATAR_LOG_JSON", "0") == "1"

_LOG_FORMAT_HUMAN = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1] is not None:
            payload["exception"] = repr(record.exc_info[1])
        for key, value in record.__dict__.items():
            if key in (
                "args",
                "msg",
                "created",
                "msecs",
                "relativeCreated",
                "exc_info",
                "exc_text",
                "stack_info",
                "levelname",
                "levelno",
                "name",
                "filename",
                "module",
                "funcName",
                "lineno",
                "pathname",
                "process",
                "processName",
                "thread",
                "threadName",
                "taskName",
            ):
                continue
            payload[key] = value
        return json.dumps(payload, default=str, sort_keys=True)


def _build_formatter() -> logging.Formatter:
    if LOG_FORMAT_JSON:
        return JsonFormatter()
    return logging.Formatter(_LOG_FORMAT_HUMAN, datefmt=_DATE_FORMAT)


def _build_handlers() -> list[logging.Handler]:
    handlers: list[logging.Handler] = []
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(_build_formatter())
    handlers.append(stream_handler)
    log_file = os.environ.get("AVATAR_LOG_FILE", "")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(_build_formatter())
        handlers.append(file_handler)
    return handlers


def configure_logging(level: str | None = None) -> logging.Logger:
    """Configure the root ``avatar`` logger and return it.

    Safe to call multiple times; subsequent calls are no-ops.
    """
    root = logging.getLogger("avatar")
    if root.handlers:
        return root
    log_level = (level or os.environ.get("AVATAR_LOG_LEVEL", "INFO").upper()).upper()
    root.setLevel(log_level)
    for handler in _build_handlers():
        root.addHandler(handler)
    root.propagate = False
    return root


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``avatar`` namespace."""
    if name:
        return logging.getLogger(f"avatar.{name}")
    return logging.getLogger("avatar")


# Configure at import time so any module that calls get_logger()
# immediately gets a working logger.
configure_logging()
