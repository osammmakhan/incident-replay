"""
Incident Timeline Builder.

Responsibility: correlate available timestamps from log findings, git findings,
and any additional context to produce a deterministic, chronologically-ordered
list of :class:`TimelineEvent` objects.

Design rules
------------
No invented timestamps
  Every event in the output timeline is sourced from real data supplied by the
  caller (log entries, git commit dates, explicit incident context).  When the
  exact time of an event is unknown but its relative position is clear (e.g. a
  deployment whose precise log timestamp is missing), the event is included with
  an explicit uncertainty marker rather than a fabricated time.

Uncertainty representation
  Timestamps with sub-second precision come from the log parser (milliseconds).
  Git commit timestamps are second-precision ISO-8601.  When neither source
  provides a timestamp, the string "(time unknown)" is used so the consumer can
  distinguish absence from zero.

Separation of concerns
  This module assembles and sorts events.  It does not interpret causation or
  assign blame.  The ``event`` field is a category label (deployment, commit,
  first_error, error_recurrence, alert, detected) — not a verdict.

No LLM required
  All logic is deterministic; the output depends only on the inputs supplied.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from incident_replay.models.schemas import TimelineEvent


# ---------------------------------------------------------------------------
# Supported event category labels
# ---------------------------------------------------------------------------

EVENT_DEPLOYMENT  = "deployment"   # service/version was deployed
EVENT_COMMIT      = "commit"       # a code commit that is relevant
EVENT_FIRST_ERROR = "first_error"  # earliest observed error instance
EVENT_ERROR       = "error"        # subsequent / recurring error
EVENT_ALERT       = "alert"        # monitoring / pager alert fired
EVENT_DETECTED    = "detected"     # incident was reported / detected


# ---------------------------------------------------------------------------
# Input data classes
# ---------------------------------------------------------------------------

class TimelineInput:
    """
    Collects all timestamp-bearing data available for timeline construction.

    All fields are optional; pass only what you have.  The builder will emit
    events only for the data it actually receives.
    """

    __slots__ = (
        "log_start_time",
        "log_end_time",
        "first_error_time",
        "deployment_time",
        "deployment_version",
        "deployment_commit",
        "deployment_host",
        "suspicious_commits",
        "error_signature",
        "affected_endpoints",
        "recurrence_count",
        "recurrence_window_seconds",
        "alert_time",
        "alert_description",
        "detected_time",
        "detected_description",
    )

    def __init__(
        self,
        *,
        # --- From LogFindings ---
        log_start_time: Optional[datetime] = None,
        log_end_time: Optional[datetime] = None,
        first_error_time: Optional[datetime] = None,
        error_signature: Optional[str] = None,
        affected_endpoints: Optional[list[str]] = None,
        recurrence_count: int = 0,
        recurrence_window_seconds: Optional[float] = None,
        # --- Deployment metadata (parsed from log or supplied separately) ---
        deployment_time: Optional[datetime] = None,
        deployment_version: Optional[str] = None,
        deployment_commit: Optional[str] = None,
        deployment_host: Optional[str] = None,
        # --- From GitFindings ---
        # Each entry: (commit_date, short_sha, message, changed_files)
        suspicious_commits: Optional[list[tuple[datetime, str, str, list[str]]]] = None,
        # --- Alert / detection events ---
        alert_time: Optional[datetime] = None,
        alert_description: Optional[str] = None,
        detected_time: Optional[datetime] = None,
        detected_description: Optional[str] = None,
    ) -> None:
        self.log_start_time = log_start_time
        self.log_end_time = log_end_time
        self.first_error_time = first_error_time
        self.error_signature = error_signature
        self.affected_endpoints = affected_endpoints or []
        self.recurrence_count = recurrence_count
        self.recurrence_window_seconds = recurrence_window_seconds
        self.deployment_time = deployment_time
        self.deployment_version = deployment_version
        self.deployment_commit = deployment_commit
        self.deployment_host = deployment_host
        self.suspicious_commits = suspicious_commits or []
        self.alert_time = alert_time
        self.alert_description = alert_description
        self.detected_time = detected_time
        self.detected_description = detected_description


# ---------------------------------------------------------------------------
# Public helpers: extract TimelineInput from agent findings
# ---------------------------------------------------------------------------

def extract_from_log_findings(findings) -> dict:
    """
    Pull timeline-relevant fields from a ``LogFindings`` dataclass.

    Returns keyword arguments suitable for passing to :class:`TimelineInput`
    (or merging with git-derived kwargs before constructing one).

    This function accepts the ``LogFindings`` dataclass by structural duck-
    typing so the caller does not need to import the log agent here.
    """
    # Scan the log entries for deployment metadata embedded as kv pairs.
    # The demo log produces lines like:
    #   INFO shopco.deploy  Starting deployment version=2.4.1 commit=cd5456a
    #   INFO shopco.deploy  Deployment complete version=2.4.1 commit=cd5456a
    # These arrive pre-parsed in LogFindings only as raw request summaries;
    # we re-derive deployment fields from log_start_time and parse_warnings.
    # The cleanest approach: surface what LogFindings directly exposes.

    alert_time: Optional[datetime] = None
    alert_description: Optional[str] = None
    detected_time: Optional[datetime] = None
    detected_description: Optional[str] = None

    # last_error_time is approximated as first_error + recurrence_window
    last_error_time: Optional[datetime] = None
    if findings.first_error_time and findings.recurrence_window_seconds:
        from datetime import timedelta
        last_error_time = findings.first_error_time + timedelta(
            seconds=findings.recurrence_window_seconds
        )

    return dict(
        log_start_time=findings.log_start_time,
        log_end_time=findings.log_end_time,
        first_error_time=findings.first_error_time,
        error_signature=findings.dominant_error_signature,
        affected_endpoints=findings.affected_endpoints,
        recurrence_count=findings.recurrence_count,
        recurrence_window_seconds=findings.recurrence_window_seconds,
        alert_time=alert_time,
        alert_description=alert_description,
        detected_time=detected_time,
        detected_description=detected_description,
    )


def extract_from_git_findings(findings) -> dict:
    """
    Pull timeline-relevant fields from a ``GitFindings`` dataclass.

    Returns kwargs suitable for merging into :class:`TimelineInput`.
    """
    commits: list[tuple[datetime, str, str, list[str]]] = []
    for cf in findings.suspicious_commits:
        commits.append((cf.date, cf.short_sha, cf.message, cf.changed_files))
    return dict(suspicious_commits=commits)


# ---------------------------------------------------------------------------
# Deployment metadata extraction from log lines
# ---------------------------------------------------------------------------

import re as _re

# Matches lines like:
#   INFO shopco.deploy  Starting deployment version=2.4.1 commit=cd5456a ...
#   INFO shopco.deploy  Deployment complete version=2.4.1 ...
_DEPLOY_LINE_RE = _re.compile(
    r"\b(?:deployment|deploy)\b", _re.IGNORECASE
)
_KV_RE = _re.compile(r"(\w+)=([^\s]+)")
_ALERT_LINE_RE = _re.compile(
    r"\b(?:alert|spike|pagerduty|page[rd]uty)\b", _re.IGNORECASE
)


def extract_deployment_from_log_lines(
    raw_lines: list[str],
) -> tuple[Optional[datetime], Optional[str], Optional[str], Optional[str]]:
    """
    Scan raw log lines for deployment metadata.

    Returns ``(deploy_time, version, commit, host)`` from the first
    ``deployment complete`` (or ``starting deployment``) line found.
    All values may be ``None`` if not present.
    """
    from incident_replay.utils.log_utils import parse_log_lines

    parsed = parse_log_lines(raw_lines)
    deploy_time: Optional[datetime] = None
    version: Optional[str] = None
    commit_sha: Optional[str] = None
    host: Optional[str] = None

    for entry in parsed.entries:
        if not _DEPLOY_LINE_RE.search(entry.message):
            continue
        kv = {k: v for k, v in _KV_RE.findall(entry.message)}
        if "version" in kv or "commit" in kv:
            if deploy_time is None:
                deploy_time = entry.timestamp
            version = version or kv.get("version")
            commit_sha = commit_sha or kv.get("commit")
            host = host or kv.get("host")

    return deploy_time, version, commit_sha, host


def extract_alert_from_log_lines(
    raw_lines: list[str],
) -> tuple[Optional[datetime], Optional[str]]:
    """
    Scan raw log lines for monitoring alert events.

    Returns ``(alert_time, description)`` for the earliest alert line found,
    or ``(None, None)`` if none is present.
    """
    from incident_replay.utils.log_utils import parse_log_lines

    parsed = parse_log_lines(raw_lines)
    for entry in parsed.entries:
        if _ALERT_LINE_RE.search(entry.message):
            desc = entry.message.strip()
            return entry.timestamp, desc
    return None, None


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build(inp: TimelineInput) -> list[TimelineEvent]:
    """
    Build and return a chronologically-sorted list of :class:`TimelineEvent`.

    Each event is sourced directly from the data in *inp*.  No timestamps are
    invented.  Events with unknown times appear at the front of the list with
    timestamp ``"(time unknown)"`` and are ordered before dated events.

    Parameters
    ----------
    inp:
        A :class:`TimelineInput` containing all available timestamped data.

    Returns
    -------
    list[TimelineEvent]
        Ordered oldest-first.  Each entry has a non-empty ``timestamp``,
        a category ``event``, and a human-readable ``description``.
    """
    raw: list[tuple[Optional[datetime], str, str]] = []
    # Each entry: (dt_or_None, event_category, description)

    # ------------------------------------------------------------------
    # 1. Deployment event
    # ------------------------------------------------------------------
    if inp.deployment_time is not None:
        parts = ["Deployment started"]
        if inp.deployment_version:
            parts.append(f"version {inp.deployment_version}")
        if inp.deployment_commit:
            parts.append(f"commit {inp.deployment_commit}")
        if inp.deployment_host:
            parts.append(f"on {inp.deployment_host}")
        raw.append((inp.deployment_time, EVENT_DEPLOYMENT, "; ".join(parts) + "."))
    elif inp.deployment_version or inp.deployment_commit:
        # Version/commit known but exact time is not
        parts = ["Deployment of"]
        if inp.deployment_version:
            parts.append(f"version {inp.deployment_version}")
        if inp.deployment_commit:
            parts.append(f"(commit {inp.deployment_commit})")
        parts.append("— exact deploy time not recorded in log.")
        raw.append((None, EVENT_DEPLOYMENT, " ".join(parts)))

    # ------------------------------------------------------------------
    # 2. Suspicious commit events
    # ------------------------------------------------------------------
    for (commit_date, short_sha, message, changed_files) in inp.suspicious_commits:
        files_label = ""
        if changed_files:
            shown = changed_files[:3]
            files_label = "; changed: " + ", ".join(shown)
            if len(changed_files) > 3:
                files_label += f" (+{len(changed_files) - 3} more)"
        desc = f"Commit {short_sha}: \"{message}\"{files_label}."
        raw.append((commit_date, EVENT_COMMIT, desc))

    # ------------------------------------------------------------------
    # 3. First error event
    # ------------------------------------------------------------------
    if inp.first_error_time is not None:
        sig = inp.error_signature or "unknown error"
        endpoints = inp.affected_endpoints
        ep_label = f" on {', '.join(endpoints)}" if endpoints else ""
        desc = f"First observed error{ep_label}: {sig}."
        raw.append((inp.first_error_time, EVENT_FIRST_ERROR, desc))

    # ------------------------------------------------------------------
    # 4. Error recurrence event (only if >1 distinct failure)
    # ------------------------------------------------------------------
    if inp.recurrence_count > 1 and inp.first_error_time is not None:
        if inp.recurrence_window_seconds is not None:
            from datetime import timedelta
            last_ts = inp.first_error_time + timedelta(
                seconds=inp.recurrence_window_seconds
            )
            window_label = _format_duration(inp.recurrence_window_seconds)
            desc = (
                f"Error recurred {inp.recurrence_count} times "
                f"over {window_label}; last occurrence at "
                f"{_fmt(last_ts)}."
            )
            raw.append((last_ts, EVENT_ERROR, desc))
        else:
            desc = (
                f"Error recurred {inp.recurrence_count} times "
                f"(exact recurrence window not available)."
            )
            raw.append((None, EVENT_ERROR, desc))

    # ------------------------------------------------------------------
    # 5. Alert event
    # ------------------------------------------------------------------
    if inp.alert_time is not None or inp.alert_description:
        if inp.alert_description:
            desc = inp.alert_description
        else:
            desc = "Monitoring alert fired."
        raw.append((inp.alert_time, EVENT_ALERT, desc))

    # ------------------------------------------------------------------
    # 6. Incident detected / reported event
    # ------------------------------------------------------------------
    if inp.detected_time is not None or inp.detected_description:
        if inp.detected_description:
            desc = inp.detected_description
        else:
            desc = "Incident detected."
        raw.append((inp.detected_time, EVENT_DETECTED, desc))

    return _sort_and_format(raw)


# ---------------------------------------------------------------------------
# Convenience: build directly from agent findings objects
# ---------------------------------------------------------------------------

def build_from_findings(
    log_findings=None,
    git_findings=None,
    log_lines: Optional[list[str]] = None,
    detected_time: Optional[datetime] = None,
    detected_description: Optional[str] = None,
) -> list[TimelineEvent]:
    """
    Convenience wrapper: construct a :class:`TimelineInput` from agent
    findings objects and call :func:`build`.

    Parameters
    ----------
    log_findings:
        A ``LogFindings`` dataclass (from ``log_agent.run()``), or ``None``.
    git_findings:
        A ``GitFindings`` dataclass (from ``git_agent.run()``), or ``None``.
    log_lines:
        Raw log lines as a list of strings.  When provided, deployment and
        alert metadata are extracted from them.  If *log_findings* was
        already produced from the same source, pass the same lines here to
        enable deployment/alert event extraction.
    detected_time:
        UTC datetime when the incident was formally reported / detected.
    detected_description:
        Human-readable incident detection description.

    Returns
    -------
    list[TimelineEvent]
    """
    kwargs: dict = {}

    if log_findings is not None:
        kwargs.update(extract_from_log_findings(log_findings))

    if git_findings is not None:
        kwargs.update(extract_from_git_findings(git_findings))

    # Extract deployment and alert metadata from raw lines if available
    if log_lines:
        dep_time, dep_version, dep_commit, dep_host = (
            extract_deployment_from_log_lines(log_lines)
        )
        if dep_time is not None or dep_version or dep_commit:
            kwargs["deployment_time"] = dep_time
            kwargs["deployment_version"] = dep_version
            kwargs["deployment_commit"] = dep_commit
            kwargs["deployment_host"] = dep_host

        alert_time, alert_desc = extract_alert_from_log_lines(log_lines)
        if alert_time is not None or alert_desc:
            kwargs["alert_time"] = alert_time
            kwargs["alert_description"] = alert_desc

    if detected_time is not None:
        kwargs["detected_time"] = detected_time
    if detected_description is not None:
        kwargs["detected_description"] = detected_description

    return build(TimelineInput(**kwargs))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_UNKNOWN_TS = "(time unknown)"


def _fmt(dt: datetime) -> str:
    """Format a UTC datetime as a compact ISO-8601 string without microseconds."""
    utc = dt.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_duration(seconds: float) -> str:
    """Return a human-readable duration string, e.g. '2m 25s'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s" if s else f"{m}m"
    h, m = divmod(m, 60)
    return f"{h}h {m}m" if m else f"{h}h"


def _sort_and_format(
    raw: list[tuple[Optional[datetime], str, str]],
) -> list[TimelineEvent]:
    """
    Sort events chronologically and convert to :class:`TimelineEvent`.

    Events with ``None`` datetime sort to the front (they represent things
    that happened before the log window or at an unrecorded time).
    """
    def _key(item: tuple[Optional[datetime], str, str]):
        dt = item[0]
        if dt is None:
            return (0, 0.0)
        return (1, dt.timestamp())

    raw.sort(key=_key)

    events: list[TimelineEvent] = []
    for dt, category, description in raw:
        ts = _fmt(dt) if dt is not None else _UNKNOWN_TS
        events.append(TimelineEvent(
            timestamp=ts,
            event=category,
            description=description,
        ))
    return events
