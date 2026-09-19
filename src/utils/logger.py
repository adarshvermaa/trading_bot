"""Structured JSON logging with secret redaction.

- File handler   : logs/bot_{date}.jsonl  (JSON lines via orjson)
- Console handler: human-readable (suppressed during dashboard mode)
- Sensitive keys are NEVER logged.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import pathlib
import re
import uuid
from typing import Any

import orjson

# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

_SENSITIVE_PATTERNS = re.compile(
    r"(api[_-]?key|api[_-]?secret|password|token|authorization|signature)",
    re.IGNORECASE,
)

_SENSITIVE_KEYS = {
    "delta_api_key", "delta_api_secret", "api_key", "api_secret",
    "signature", "password", "token", "authorization",
    "litellm_api_key",
}


def _redact(data: Any) -> Any:
    """Recursively redact sensitive values from dicts."""
    if isinstance(data, dict):
        return {
            k: "***REDACTED***" if k.lower() in _SENSITIVE_KEYS else _redact(v)
            for k, v in data.items()
        }
    if isinstance(data, (list, tuple)):
        return [_redact(item) for item in data]
    if isinstance(data, str) and _SENSITIVE_PATTERNS.search(data):
        return "***REDACTED***"
    return data


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Formats log records as JSON lines using orjson."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach extra fields (correlation_id, etc.)
        for key in ("correlation_id", "symbol", "order_id", "event", "component"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return orjson.dumps(_redact(log_entry)).decode()


class HumanFormatter(logging.Formatter):
    """Concise coloured formatter for the console."""

    COLOURS = {
        "DEBUG": "\033[90m",
        "INFO": "\033[36m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.COLOURS.get(record.levelname, "")
        ts = dt.datetime.fromtimestamp(
            record.created, tz=dt.timezone.utc
        ).strftime("%H:%M:%S")
        msg = record.getMessage()
        # Redact any inline secrets
        if _SENSITIVE_PATTERNS.search(msg):
            msg = _SENSITIVE_PATTERNS.sub("***REDACTED***", msg)
        base = f"{colour}{ts} [{record.levelname[0]}] {record.name}: {msg}{self.RESET}"
        if record.exc_info and record.exc_info[1]:
            base += "\n" + self.formatException(record.exc_info)
        return base


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_SETUP_DONE = False


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    console: bool = True,
) -> None:
    """Initialise root logger with JSON file handler and optional console handler."""
    global _SETUP_DONE
    if _SETUP_DONE:
        return
    _SETUP_DONE = True

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # File handler
    log_path = pathlib.Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    file_handler = logging.FileHandler(log_path / f"bot_{today}.jsonl", encoding="utf-8")
    file_handler.setFormatter(JSONFormatter())
    root.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(HumanFormatter())
        root.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)


def new_correlation_id() -> str:
    """Generate a short correlation ID for tracing order lifecycles."""
    return uuid.uuid4().hex[:12]
