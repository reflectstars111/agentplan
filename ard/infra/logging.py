"""Structured JSON-line logging for ARD."""

import json
import sys
from datetime import datetime, timezone
from typing import Any


class StructuredLogger:
    """Simple JSON-line logger for observability."""

    def __init__(self, name: str = "ard", stream=sys.stderr):
        self.name = name
        self.stream = stream

    def _emit(self, level: str, msg: str, **extra) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "logger": self.name,
            "level": level,
            "msg": msg,
            **extra,
        }
        print(json.dumps(record, default=str), file=self.stream)

    def info(self, msg: str, **extra) -> None:
        self._emit("INFO", msg, **extra)

    def warn(self, msg: str, **extra) -> None:
        self._emit("WARN", msg, **extra)

    def error(self, msg: str, **extra) -> None:
        self._emit("ERROR", msg, **extra)

    def debug(self, msg: str, **extra) -> None:
        self._emit("DEBUG", msg, **extra)


log = StructuredLogger()
