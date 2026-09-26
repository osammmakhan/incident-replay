"""
Log inspection utilities for incident-replay agents.

Parses structured log files produced by common Python logging formats and
extracts error lines, HTTP status events, request metadata, and inline
stack traces.

Supported log line format (loosely matched via regex):
    YYYY-MM-DD HH:MM:SS,mmm LEVEL  logger_name  message key=value ...

All public functions accept a path string or a list of raw log lines so
that tests can pass strings directly without hitting the filesystem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Standard Python logging timestamp: 2026-09-26 22:01:05,047
_TIMESTAMP_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?)"
)

# Full log-line parser (timestamp + level + logger + message)
_LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?)"
    r"\s+(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)"
    r"\s+(?P<logger>\S+)"
    r"\s+(?P<message>.+)$"
)

# HTTP status line: POST /checkout 500 44ms
_HTTP_RE = re.compile(
    r"(?P<method>GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+"
    r"(?P<path>/\S*)\s+"
    r"(?P<status>\d{3})\s+"
    r"(?P<duration>\S+)"
)

# Key=value pairs in a log message (handles quoted values too)
_KV_RE = re.compile(r"(\w+)=([^\s]+)")

# Python traceback markers
_TRACEBACK_START_RE = re.compile(r"^Traceback \(most recent call last\):")
_TRACEBACK_FRAME_RE = re.compile(r'^\s+File "(?P<file>[^"]+)", line (?P<line>\d+)')
_EXCEPTION_LINE_RE = re.compile(r"^(?P<exc_type>[A-Za-z][A-Za-z0-9_.]*Error|TypeError|ValueError|KeyError|AttributeError|RuntimeError|Exception):\s*(?P<msg>.+)")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    """A single parsed log line."""

    raw: str
    timestamp: Optional[datetime]   # None if unparseable
    level: str                      # "INFO", "ERROR", etc.; "" if unknown
    logger: str                     # Logger name; "" if unknown
    message: str                    # Message body
    kv: dict[str, str] = field(default_factory=dict)  # Parsed key=value pairs


@dataclass
class HttpEvent:
    """An HTTP request/response event extracted from the log."""

    timestamp: Optional[datetime]
    method: str
    path: str
    status: int
    duration_raw: str               # e.g. "44ms" — kept as string
    request_id: Optional[str]
    raw: str


@dataclass
class ErrorEvent:
    """An ERROR-level log entry with optional parsed error type."""

    entry: LogEntry
    error_type: Optional[str]       # e.g. "TypeError"
    error_message: Optional[str]    # The message after the colon
    request_id: Optional[str]


@dataclass
class ParsedLog:
    """Structured view of an entire log file."""

    entries: list[LogEntry]
    errors: list[ErrorEvent]
    http_events: list[HttpEvent]
    first_error_time: Optional[datetime]
    error_signature: Optional[str]  # Most common error type/message
    failing_request_ids: list[str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_log(path: str) -> list[str]:
    """Read a log file and return its lines.

    Raises ``OSError`` if the file cannot be opened.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read().splitlines()


def parse_log_lines(lines: list[str]) -> ParsedLog:
    """Parse a list of raw log lines into a :class:`ParsedLog`.

    Handles both well-formed structured lines and plain-text lines that
    lack a recognised prefix (they are stored with empty level/logger).
    """
    entries: list[LogEntry] = []
    errors: list[ErrorEvent] = []
    http_events: list[HttpEvent] = []

    for raw in lines:
        entry = _parse_line(raw)
        entries.append(entry)

        # Classify HTTP events
        http = _extract_http(entry)
        if http is not None:
            http_events.append(http)

        # Classify error events
        if entry.level in ("ERROR", "CRITICAL", "FATAL"):
            errors.append(_build_error_event(entry))

    first_error_time = errors[0].entry.timestamp if errors else None
    error_signature = _dominant_error_signature(errors)
    failing_ids = _collect_failing_request_ids(errors)

    return ParsedLog(
        entries=entries,
        errors=errors,
        http_events=http_events,
        first_error_time=first_error_time,
        error_signature=error_signature,
        failing_request_ids=failing_ids,
    )


def parse_log_file(path: str) -> ParsedLog:
    """Convenience wrapper: read *path* then parse its lines."""
    return parse_log_lines(read_log(path))


def extract_error_lines(lines: list[str]) -> list[str]:
    """Return only the lines whose level is ERROR, CRITICAL, or FATAL."""
    result = []
    for raw in lines:
        m = _LOG_LINE_RE.match(raw)
        if m and m.group("level") in ("ERROR", "CRITICAL", "FATAL"):
            result.append(raw)
    return result


def extract_stack_trace(lines: list[str]) -> Optional[str]:
    """Find and return the first Python traceback block in *lines*.

    Returns ``None`` if no traceback is present.
    A traceback block starts with "Traceback (most recent call last):"
    and ends before the next non-indented, non-exception line.
    """
    in_tb = False
    block: list[str] = []

    for line in lines:
        if _TRACEBACK_START_RE.match(line):
            in_tb = True
            block = [line]
            continue
        if in_tb:
            # Indented lines or exception-type lines belong to the block
            if line.startswith(" ") or _EXCEPTION_LINE_RE.match(line):
                block.append(line)
            else:
                # Non-indented line ends the traceback
                break

    return "\n".join(block) if block else None


def extract_request_ids(lines: list[str]) -> list[str]:
    """Return all unique request_id values found in *lines*, in order of first
    appearance."""
    seen: set[str] = set()
    result: list[str] = []
    for raw in lines:
        for _, v in _KV_RE.findall(raw):
            pass  # we only want request_id key
        kv = dict(_KV_RE.findall(raw))
        rid = kv.get("request_id")
        if rid and rid not in seen:
            seen.add(rid)
            result.append(rid)
    return result


def get_timestamps(lines: list[str]) -> list[datetime]:
    """Return a list of parsed timestamps from *lines*, skipping unparseable ones."""
    result = []
    for raw in lines:
        m = _TIMESTAMP_RE.match(raw)
        if m:
            dt = _parse_log_ts(m.group("ts"))
            if dt:
                result.append(dt)
    return result


def first_error_timestamp(lines: list[str]) -> Optional[datetime]:
    """Return the timestamp of the first ERROR/CRITICAL line, or ``None``."""
    for raw in lines:
        m = _LOG_LINE_RE.match(raw)
        if m and m.group("level") in ("ERROR", "CRITICAL", "FATAL"):
            return _parse_log_ts(m.group("ts"))
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_line(raw: str) -> LogEntry:
    m = _LOG_LINE_RE.match(raw)
    if m:
        ts = _parse_log_ts(m.group("ts"))
        kv = dict(_KV_RE.findall(m.group("message")))
        return LogEntry(
            raw=raw,
            timestamp=ts,
            level=m.group("level"),
            logger=m.group("logger"),
            message=m.group("message"),
            kv=kv,
        )
    # Unrecognised format — store raw with empty metadata
    ts_m = _TIMESTAMP_RE.match(raw)
    ts = _parse_log_ts(ts_m.group("ts")) if ts_m else None
    return LogEntry(raw=raw, timestamp=ts, level="", logger="", message=raw, kv={})


def _parse_log_ts(ts_str: str) -> Optional[datetime]:
    """Parse ``YYYY-MM-DD HH:MM:SS[,mmm]`` to a UTC-aware datetime."""
    # Strip milliseconds suffix if present
    ts_clean = ts_str.replace(",", ".")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(ts_clean, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _extract_http(entry: LogEntry) -> Optional[HttpEvent]:
    """Build an :class:`HttpEvent` from *entry* if it contains an HTTP line."""
    m = _HTTP_RE.search(entry.message)
    if not m:
        return None
    return HttpEvent(
        timestamp=entry.timestamp,
        method=m.group("method"),
        path=m.group("path"),
        status=int(m.group("status")),
        duration_raw=m.group("duration"),
        request_id=entry.kv.get("request_id"),
        raw=entry.raw,
    )


def _build_error_event(entry: LogEntry) -> ErrorEvent:
    """Extract error type and message from an ERROR-level :class:`LogEntry`."""
    # Look for "error=TypeName: message" pattern in the message body
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Pattern: error=TypeError: some message
    err_kv = re.search(r"error=([A-Za-z][A-Za-z0-9_.]*(?:Error|Exception)):\s*(.+?)(?:\s+\w+=|$)", entry.message)
    if err_kv:
        error_type = err_kv.group(1)
        error_message = err_kv.group(2).strip()
    else:
        # Pattern: ErrorType: message at start or after whitespace
        exc_m = _EXCEPTION_LINE_RE.search(entry.message)
        if exc_m:
            error_type = exc_m.group("exc_type")
            error_message = exc_m.group("msg")

    return ErrorEvent(
        entry=entry,
        error_type=error_type,
        error_message=error_message,
        request_id=entry.kv.get("request_id"),
    )


def _dominant_error_signature(errors: list[ErrorEvent]) -> Optional[str]:
    """Return the most frequently occurring ``error_type: error_message`` string."""
    if not errors:
        return None
    from collections import Counter
    sigs = [
        f"{e.error_type}: {e.error_message}"
        for e in errors
        if e.error_type and e.error_message
    ]
    if not sigs:
        return None
    return Counter(sigs).most_common(1)[0][0]


def _collect_failing_request_ids(errors: list[ErrorEvent]) -> list[str]:
    """Return unique request IDs from error events, in order of first appearance."""
    seen: set[str] = set()
    result: list[str] = []
    for e in errors:
        rid = e.request_id
        if rid and rid not in seen:
            seen.add(rid)
            result.append(rid)
    return result
