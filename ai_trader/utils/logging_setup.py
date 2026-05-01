"""Centralized logging — plain or JSON, file + stderr."""
from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

_INITIALIZED = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str | Path] = None,
) -> None:
    """Configure root logger. Idempotent — safe to call from every entry point."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    root = logging.getLogger()
    root.setLevel(level.upper())

    fmt: logging.Formatter
    if json_format:
        fmt = _JsonFormatter()
    else:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
            datefmt="%H:%M:%S",
        )

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rot = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10_000_000, backupCount=5
        )
        rot.setFormatter(fmt)
        root.addHandler(rot)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    if not _INITIALIZED:
        setup_logging()
    return logging.getLogger(name)
