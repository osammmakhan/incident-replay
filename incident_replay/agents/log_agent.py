"""
Log Investigator agent.

Responsibility: analyse a supplied incident log and return a structured
:class:`LogFindings` report of *factual observations only*.

Design rules
------------
OBSERVATION vs INTERPRETATION
  Every field in LogFindings describes what the log *actually contains*.
  The agent does not infer root causes, blame commits, or suggest fixes.
  Interpretation is left to the synthesis stage.

No hardcoded answers
  Patterns are generic enough to work against any structured log following
  the ``YYYY-MM-DD HH:MM:SS,mmm LEVEL logger message [key=value ...]``
  convention.  The demo project's production.log is one valid input, not
  the only one.

No LLM required
  This agent is entirely deterministic regex + structural analysis.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from incident_replay.utils.log_utils import (
    ErrorEvent,
    HttpEvent,
    ParsedLog,
    parse_log_file,
    parse_log_lines,
)


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class RequestSummary:
    """Observed facts about a single request that produced an error."""

    request_id: str
    endpoint: Optional[str]          # e.g. "POST /checkout"
    http_status: Optional[int]       # e.g. 500
    error_type: Optional[str]        # e.g. "TypeError"
    error_message: Optional[str]
    suspicious_kv: dict[str, str]    # key=value pairs that look anomalous
    raw_error_lines: list[str]       # the actual log lines, verbatim


@dataclass
class LogFindings:
    """
    Structured observations extracted from an incident log.

    All fields describe what the log *contains* — not what caused the
    incident.  The ``interpretation_hints`` list may suggest avenues for
    further investigation but explicitly labels them as hints, not
    conclusions.
    """

    # --- Coverage ---
    log_path: str
    total_lines: int
    parse_warnings: list[str]        # lines that could not be fully parsed

    # --- Timeline ---
    log_start_time: Optional[datetime]
    log_end_time: Optional[datetime]
    first_error_time: Optional[datetime]

    # --- Error pattern ---
    error_count: int                 # total ERROR/CRITICAL/FATAL lines
    dominant_error_signature: Optional[str]   # most frequent "ErrorType: msg"
    unique_error_signatures: list[str]        # all distinct signatures seen
    affected_endpoints: list[str]             # distinct endpoints with errors
    http_500_count: int

    # --- Per-request breakdown ---
    failing_requests: list[RequestSummary]

    # --- Suspicious input values ---
    # key=value pairs that appear exclusively (or predominantly) in failing
    # requests and never in successful ones
    suspicious_input_patterns: list[str]

    # --- Stack trace ---
    stack_trace_present: bool
    stack_trace: Optional[str]       # verbatim, if found in the log

    # --- Frequency / recurrence ---
    recurrence_count: int            # how many distinct failing request_ids
    recurrence_window_seconds: Optional[float]   # span from first to last error

    # --- Interpretation hints (explicitly labelled, NOT conclusions) ---
    interpretation_hints: list[str]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class LogAgentError(ValueError):
    """Raised when the log cannot be read or is unusable."""


def run(
    log_path: str,
    stack_trace: Optional[str] = None,
) -> LogFindings:
    """Analyse *log_path* and return a :class:`LogFindings` report.

    Parameters
    ----------
    log_path:
        Path to the incident log file.  May be ``None`` or empty — the
        agent will return an empty findings object rather than raising.
    stack_trace:
        Optional raw stack trace text supplied alongside the log (e.g.
        from a separate ``stacktrace.txt`` file).  When provided it is
        stored verbatim and takes priority over any inline traceback
        found inside the log.

    Returns
    -------
    LogFindings
        Always returns a findings object.  Use ``error_count == 0`` and
        ``parse_warnings`` to detect unusable / empty logs.

    Raises
    ------
    LogAgentError
        Only when *log_path* points to something that cannot be read at
        all (permission denied, is a directory, etc.) and the path is
        non-empty.
    """
    # --- Handle missing / empty path ---
    if not log_path or not log_path.strip():
        return _empty_findings("<no path supplied>")

    if not os.path.exists(log_path):
        return _empty_findings(log_path, warning=f"Log file not found: {log_path!r}")

    if not os.path.isfile(log_path):
        raise LogAgentError(f"Log path is not a regular file: {log_path!r}")

    try:
        parsed = parse_log_file(log_path)
    except OSError as exc:
        raise LogAgentError(f"Cannot read log file {log_path!r}: {exc}") from exc

    return _build_findings(log_path, parsed, stack_trace)


def run_from_lines(
    lines: list[str],
    stack_trace: Optional[str] = None,
    log_path: str = "<inline>",
) -> LogFindings:
    """Analyse log lines passed directly (useful for testing).

    Parameters
    ----------
    lines:
        Raw log lines (as strings, no trailing newline required).
    stack_trace:
        Optional external stack trace text.
    log_path:
        Label used in the findings for traceability.
    """
    if not lines:
        return _empty_findings(log_path, warning="Log contained no lines.")
    parsed = parse_log_lines(lines)
    return _build_findings(log_path, parsed, stack_trace)


# ---------------------------------------------------------------------------
# Internal builder
# ---------------------------------------------------------------------------

def _build_findings(
    log_path: str,
    parsed: ParsedLog,
    external_stack_trace: Optional[str],
) -> LogFindings:
    warnings: list[str] = []

    # Unparsed lines (level == "" means the line didn't match the log format)
    unparsed = [e.raw for e in parsed.entries if e.level == ""]
    if unparsed:
        warnings.append(
            f"{len(unparsed)} line(s) did not match the expected log format."
        )

    # --- Timeline ---
    all_ts = [e.timestamp for e in parsed.entries if e.timestamp]
    log_start = min(all_ts) if all_ts else None
    log_end = max(all_ts) if all_ts else None

    # --- HTTP 500s ---
    http_500s = [h for h in parsed.http_events if h.status >= 500]
    affected_endpoints = _distinct_ordered([
        f"{h.method} {h.path}" for h in http_500s
    ])

    # --- Per-request summaries ---
    failing_requests = _build_request_summaries(parsed)

    # --- Unique error signatures ---
    all_sigs = _distinct_ordered([
        f"{e.error_type}: {e.error_message}"
        for e in parsed.errors
        if e.error_type and e.error_message
    ])

    # --- Suspicious input patterns ---
    suspicious = _find_suspicious_inputs(parsed)

    # --- Stack trace ---
    # External stack trace (from a dedicated file) takes priority
    stack_trace_text: Optional[str] = external_stack_trace or None
    if not stack_trace_text:
        # Try to find an inline traceback in the raw log lines
        from incident_replay.utils.log_utils import extract_stack_trace
        inline_tb = extract_stack_trace([e.raw for e in parsed.entries])
        stack_trace_text = inline_tb

    # --- Recurrence window ---
    # Use the timestamp of the first error per failing request_id, so that
    # two ERROR lines from the same request (FAILED + HTTP-500) don't
    # falsely imply recurrence.
    first_ts_per_request: list[datetime] = []
    seen_for_window: set[str] = set()
    for e in parsed.errors:
        rid = e.request_id or "<unknown>"
        if rid not in seen_for_window and e.entry.timestamp:
            seen_for_window.add(rid)
            first_ts_per_request.append(e.entry.timestamp)
    recurrence_window: Optional[float] = None
    if len(first_ts_per_request) >= 2:
        span = max(first_ts_per_request) - min(first_ts_per_request)
        recurrence_window = span.total_seconds()

    # --- Interpretation hints (explicitly qualified) ---
    hints = _derive_hints(parsed, suspicious, failing_requests)

    return LogFindings(
        log_path=log_path,
        total_lines=len(parsed.entries),
        parse_warnings=warnings,
        log_start_time=log_start,
        log_end_time=log_end,
        first_error_time=parsed.first_error_time,
        error_count=len(parsed.errors),
        dominant_error_signature=parsed.error_signature,
        unique_error_signatures=all_sigs,
        affected_endpoints=affected_endpoints,
        http_500_count=len(http_500s),
        failing_requests=failing_requests,
        suspicious_input_patterns=suspicious,
        stack_trace_present=stack_trace_text is not None,
        stack_trace=stack_trace_text,
        recurrence_count=len(parsed.failing_request_ids),
        recurrence_window_seconds=recurrence_window,
        interpretation_hints=hints,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_request_summaries(parsed: ParsedLog) -> list[RequestSummary]:
    """Build one RequestSummary per failing request_id."""
    # Group errors by request_id
    by_id: dict[str, list[ErrorEvent]] = {}
    for err in parsed.errors:
        rid = err.request_id or "<unknown>"
        by_id.setdefault(rid, []).append(err)

    # Also collect the HTTP event for each request_id
    http_by_id: dict[str, HttpEvent] = {}
    for h in parsed.http_events:
        if h.request_id and h.status >= 500:
            http_by_id[h.request_id] = h

    # Collect all KV pairs from INFO lines for each request_id so we can
    # surface suspicious input values
    info_kv_by_id: dict[str, dict[str, str]] = {}
    for entry in parsed.entries:
        rid = entry.kv.get("request_id")
        if rid:
            info_kv_by_id.setdefault(rid, {}).update(entry.kv)

    summaries: list[RequestSummary] = []
    for rid in parsed.failing_request_ids:
        errors_for_rid = by_id.get(rid, [])
        http_ev = http_by_id.get(rid)
        all_kv = info_kv_by_id.get(rid, {})

        # Pick the first proper error (not an HTTP-status line) for type/msg
        typed_err = next(
            (e for e in errors_for_rid if e.error_type), None
        )

        endpoint: Optional[str] = None
        status: Optional[int] = None
        if http_ev:
            endpoint = f"{http_ev.method} {http_ev.path}"
            status = http_ev.status

        summaries.append(RequestSummary(
            request_id=rid,
            endpoint=endpoint,
            http_status=status,
            error_type=typed_err.error_type if typed_err else None,
            error_message=typed_err.error_message if typed_err else None,
            suspicious_kv=_suspicious_kv_for_request(all_kv),
            raw_error_lines=[e.entry.raw for e in errors_for_rid],
        ))

    return summaries


# Values that are structurally suspicious: "null", "none", empty, or
# obviously sentinel-like strings.
_SUSPICIOUS_VALUES_RE = re.compile(
    r"^(?:null|none|nil|undefined|nan|-1|0\.0|0)$", re.IGNORECASE
)
# Keys that carry input data (not metadata like request_id, order_id)
_INPUT_KEYS_RE = re.compile(
    r"^(?:discount|coupon|promo|code|amount|quantity|price|rate|value|input)$",
    re.IGNORECASE,
)


def _suspicious_kv_for_request(kv: dict[str, str]) -> dict[str, str]:
    """Return key=value pairs whose value looks anomalous."""
    return {
        k: v for k, v in kv.items()
        if _SUSPICIOUS_VALUES_RE.match(v) or _INPUT_KEYS_RE.match(k)
    }


def _find_suspicious_inputs(parsed: ParsedLog) -> list[str]:
    """
    Find input key=value patterns that appear in failing requests but
    not in successful ones.

    Returns strings of the form ``"key=value"`` — purely observational.
    """
    failing_ids = set(parsed.failing_request_ids)

    failing_kvs: Counter[str] = Counter()
    ok_kvs: Counter[str] = Counter()

    for entry in parsed.entries:
        rid = entry.kv.get("request_id")
        if not rid:
            continue
        for k, v in entry.kv.items():
            if k in ("request_id", "order_id", "customer_id", "total",
                     "status", "host", "version", "commit", "branch",
                     "error"):
                continue
            token = f"{k}={v}"
            if rid in failing_ids:
                failing_kvs[token] += 1
            else:
                ok_kvs[token] += 1

    # Keep tokens that appear in failing requests but never in successful ones
    exclusive = [t for t in failing_kvs if t not in ok_kvs]
    return sorted(exclusive)


def _derive_hints(
    parsed: ParsedLog,
    suspicious: list[str],
    failing: list[RequestSummary],
) -> list[str]:
    """
    Produce interpretation *hints* — not conclusions.

    Each hint is a complete sentence that names its evidence and explicitly
    states it is a hint for further investigation.
    """
    hints: list[str] = []

    if suspicious:
        hints.append(
            f"HINT: The following input pattern(s) appear only in failing "
            f"requests and may be worth investigating: {', '.join(suspicious)}."
        )

    if parsed.error_signature:
        hints.append(
            f"HINT: The dominant error signature is "
            f"'{parsed.error_signature}'. Investigate code paths that "
            f"operate on the input types involved."
        )

    if len(parsed.failing_request_ids) > 1:
        hints.append(
            f"HINT: The error recurred {len(parsed.failing_request_ids)} "
            f"time(s), suggesting the triggering condition is reproducible "
            f"rather than a transient fault."
        )

    endpoints = {r.endpoint for r in failing if r.endpoint}
    if len(endpoints) == 1:
        ep = next(iter(endpoints))
        hints.append(
            f"HINT: All observed failures are on the same endpoint ({ep}), "
            f"narrowing the investigation to that handler."
        )

    return hints


def _distinct_ordered(items: list[str]) -> list[str]:
    """Return unique items preserving first-occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _empty_findings(log_path: str, warning: str = "") -> LogFindings:
    warnings = [warning] if warning else []
    return LogFindings(
        log_path=log_path,
        total_lines=0,
        parse_warnings=warnings,
        log_start_time=None,
        log_end_time=None,
        first_error_time=None,
        error_count=0,
        dominant_error_signature=None,
        unique_error_signatures=[],
        affected_endpoints=[],
        http_500_count=0,
        failing_requests=[],
        suspicious_input_patterns=[],
        stack_trace_present=False,
        stack_trace=None,
        recurrence_count=0,
        recurrence_window_seconds=None,
        interpretation_hints=[],
    )
