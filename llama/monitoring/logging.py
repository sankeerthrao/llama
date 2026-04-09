"""Structured logging for LLaMA inference serving.

Provides a thin wrapper around Python's :mod:`logging` module that supports
both human-readable text output and machine-parseable JSON lines.
"""

import json
import logging
import sys
import time
from typing import Any, Optional


class _JsonFormatter(logging.Formatter):
    """Logging formatter that emits one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        # Merge extra structured fields attached by StructuredLogger
        extra: dict[str, Any] = getattr(record, "_structured_extra", {})
        if extra:
            log_entry.update(extra)

        # Include exception info when present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class _TextFormatter(logging.Formatter):
    """Human-readable formatter with optional key=value structured fields."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        ts = self.formatTime(record, self.datefmt)
        base = f"{ts} [{record.levelname}] {record.getMessage()}"

        extra: dict[str, Any] = getattr(record, "_structured_extra", {})
        if extra:
            kv_pairs = " ".join(f"{k}={v}" for k, v in extra.items())
            base = f"{base} | {kv_pairs}"

        if record.exc_info and record.exc_info[0] is not None:
            base = f"{base}\n{self.formatException(record.exc_info)}"

        return base


class StructuredLogger:
    """Logger that attaches arbitrary structured fields to every message.

    Example::

        logger = StructuredLogger("llama.inference")
        logger.info("Request completed", latency=0.34, tokens=128)
    """

    def __init__(self, name: str = "llama") -> None:
        """Create a structured logger wrapping a stdlib :class:`logging.Logger`.

        Args:
            name: Logger name passed to :func:`logging.getLogger`.
        """
        self._logger = logging.getLogger(name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def info(self, message: str, **kwargs: Any) -> None:
        """Log at INFO level with optional structured fields."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log at WARNING level with optional structured fields."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log at ERROR level with optional structured fields."""
        self._log(logging.ERROR, message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log at DEBUG level with optional structured fields."""
        self._log(logging.DEBUG, message, **kwargs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Emit a log record with structured extras attached."""
        if not self._logger.isEnabledFor(level):
            return
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        record._structured_extra = kwargs  # type: ignore[attr-defined]
        self._logger.handle(record)


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def setup_logging(
    level: str = "INFO",
    format: str = "text",  # noqa: A002
    log_file: Optional[str] = None,
) -> StructuredLogger:
    """Configure and return a :class:`StructuredLogger`.

    Args:
        level: Logging level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``).
        format: Output format — ``"text"`` for human-readable output or
            ``"json"`` for one-JSON-object-per-line output.
        log_file: If provided, log output is written to this file path
            *in addition to* stderr.

    Returns:
        A configured :class:`StructuredLogger` instance.
    """
    logger_name = "llama"
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers when called multiple times
    logger.handlers.clear()

    formatter: logging.Formatter
    if format.lower() == "json":
        formatter = _JsonFormatter()
    else:
        formatter = _TextFormatter()

    # Always add a stderr handler
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Optionally add a file handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return StructuredLogger(logger_name)
