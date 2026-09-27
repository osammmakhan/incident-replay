"""
Evidence normalization and aggregation layer.

Purpose
-------
Each investigator agent returns its own findings dataclass with
source-specific field names (``LogFindings``, ``GitFindings``,
``CodeFindings``, ``TestInvestigatorFindings``).  This module converts those
heterogeneous findings into a single uniform type —
:class:`incident_replay.models.schemas.Evidence` — so downstream stages
consume one list instead of four.  Conversion is purely structural: every
findings object is read through duck-typed attribute access and nothing in
this module imports from ``incident_replay.agents`` (same approach as
:mod:`incident_replay.analysis.timeline`).

Traceability convention for ``location``
----------------------------------------
Every emitted :class:`Evidence` carries a ``location`` string that lets a
reader re-read the underlying fact without re-running the pipeline:

* log facts    — the log path plus a locator suffix that points at the
  specific evidence inside the file, e.g. ``f"{log_path}"``,
  ``f"{log_path} (first error {iso_ts})"``,
  ``f"{log_path} (request_id={rid})"``,
  ``f"{log_path} (endpoint={ep})"``.
* git facts    — ``f"commit {short_sha}"`` (the commit is its own address).
* code facts   — ``f"{file}:{start_line}"``.
* test facts   — ``f"{file}:{start_line}"``; coverage gaps, which have no
  single source line, use ``f"{test_root} (coverage gap)"``.

When the underlying path is absent, a ``"<... unknown>"`` marker (in the
spirit of the timeline module's ``"(time unknown)"``) is substituted so the
field is never empty.

Observation vs hypothesis
-------------------------
``observation`` restates a fact that is *already present* in the input
findings; ``relevance`` restates why the producing agent recorded that fact
(its ``relevance_reasons`` / ``relevance`` / scenario metadata).  This layer
never adds numbers, timestamps, files, commits, or conclusions that are not
in the inputs, and it never uses causal language ("caused", "introduced the
bug", "root cause") of its own — causal judgement belongs to the synthesis
stage.  Input text (error signatures, suspicious input patterns, commit
messages, agent-authored observations) is passed through verbatim, even when
the input itself sounds causal; we normalize facts, we do not rewrite them.

Dedupe rule
-----------
Two entries are duplicates when their normalized
``(source, location, observation)`` tuple is identical, where each text
component is casefolded and has its whitespace runs collapsed to single
spaces.  ``relevance`` is *not* part of the key.  :func:`dedupe` keeps the
first-seen entry and drops later duplicates, preserving input order;
:func:`collect` applies it to the concatenated converter output (log → git →
code → test) so its result never repeats a fact.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from incident_replay.models.schemas import Evidence


# ---------------------------------------------------------------------------
# Source labels (must match Evidence's Literal["log", "git", "code", "test"])
# ---------------------------------------------------------------------------

SOURCE_LOG = "log"
SOURCE_GIT = "git"
SOURCE_CODE = "code"
SOURCE_TEST = "test"


# ---------------------------------------------------------------------------
# Fallback markers used when the input lacks a path/identifier.
# They guarantee every ``location`` is non-empty without inventing data.
# ---------------------------------------------------------------------------

_UNKNOWN_LOG_PATH = "<log path unknown>"
_UNKNOWN_TEST_ROOT = "<test root unknown>"
_UNKNOWN_SHA = "<commit sha unknown>"


# Stack-trace parsing (used only on text supplied by the caller)
_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')
_EXC_LINE_RE = re.compile(r"^\s*([\w.]+(?:Error|Exception))\s*:", re.MULTILINE)
_EXC_NAME_RE = re.compile(r"\b([A-Za-z_][\w.]*(?:Error|Exception))\b")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _text(value) -> str:
    """Coerce *value* to a stripped string (``None`` → ``""``)."""
    if value is None:
        return ""
    return str(value).strip()


def _iso(ts) -> str:
    """
    Render a timestamp as a deterministic ISO-8601 string.

    Aware datetimes are converted to UTC and rendered as
    ``YYYY-MM-DDTHH:MM:SSZ``; naive datetimes are rendered as-is; anything
    else is stringified.  Returns ``""`` for ``None``.
    """
    if ts is None:
        return ""
    if isinstance(ts, datetime):
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc)
            return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        return ts.strftime("%Y-%m-%dT%H:%M:%S")
    return str(ts)


def _entries(findings, name: str) -> list:
    """Return ``findings.<name>`` when it is a non-empty list, else ``[]``."""
    if findings is None:
        return []
    value = getattr(findings, name, None)
    if not isinstance(value, list):
        return []
    return [item for item in value if item is not None]


def _norm(value) -> str:
    """Casefold text and collapse whitespace runs to single spaces."""
    if value is None:
        return ""
    return " ".join(str(value).split()).casefold()


# ---------------------------------------------------------------------------
# Public API: converters (duck-typed; no imports from incident_replay.agents)
# ---------------------------------------------------------------------------

def from_log_findings(findings) -> list[Evidence]:
    """
    Convert a ``LogFindings``-shaped object into log-sourced evidence.

    Emits one :class:`Evidence` per distinct fact found in the findings:
    the dominant error signature (with error count and first-error time),
    each affected endpoint (with failure counts), each suspicious input
    pattern (verbatim), the recurrence fact when more than one distinct
    failing request id was recorded, and the stack-trace presence fact.
    Returns ``[]`` when *findings* is ``None`` or contains no such facts.
    """
    if findings is None:
        return []

    out: list[Evidence] = []
    log_path = _text(getattr(findings, "log_path", None)) or _UNKNOWN_LOG_PATH

    # --- 1. Dominant error signature + error count + first error time ---
    signature = _text(getattr(findings, "dominant_error_signature", None))
    if signature:
        count = getattr(findings, "error_count", None)
        first_iso = _iso(getattr(findings, "first_error_time", None))
        parts = [f"dominant error signature: {signature}"]
        if count is not None:
            parts.append(f"log contains {count} error line(s)")
        if first_iso:
            parts.append(f"first error at {first_iso}")
        observation = "; ".join(parts) + "."
        location = (
            f"{log_path} (first error {first_iso})" if first_iso else log_path
        )
        out.append(Evidence(
            source=SOURCE_LOG,
            location=location,
            observation=observation,
            relevance="Most frequent error signature recorded in the log.",
        ))

    # --- 2. Affected endpoint(s) with failure counts ---
    failing_requests = _entries(findings, "failing_requests")
    endpoints: list[str] = []
    for ep in _entries(findings, "affected_endpoints"):
        ep_text = _text(ep)
        if ep_text and ep_text not in endpoints:
            endpoints.append(ep_text)
    for req in failing_requests:
        ep_text = _text(getattr(req, "endpoint", None))
        if ep_text and ep_text not in endpoints:
            endpoints.append(ep_text)

    http_500 = getattr(findings, "http_500_count", None)
    for ep in endpoints:
        rids = [
            _text(getattr(req, "request_id", None))
            for req in failing_requests
            if _text(getattr(req, "endpoint", None)) == ep
        ]
        rids = [r for r in rids if r]
        bits: list[str] = []
        if http_500 is not None:
            bits.append(f"log records {http_500} HTTP 500 response(s)")
        bits.append(f"{len(rids)} failing request(s) reference this endpoint")
        observation = f"Endpoint {ep}: " + "; ".join(bits) + "."
        if rids:
            observation += f" request_ids: {', '.join(rids)}."
        out.append(Evidence(
            source=SOURCE_LOG,
            location=f"{log_path} (endpoint={ep})",
            observation=observation,
            relevance="Endpoint appears in the log's affected-endpoint list.",
        ))

    # --- 3. Each suspicious input pattern (verbatim value) ---
    for pattern in _entries(findings, "suspicious_input_patterns"):
        pat = str(pattern)
        if not _text(pat):
            continue
        out.append(Evidence(
            source=SOURCE_LOG,
            location=f"{log_path} (suspicious input: {pat})",
            observation=(
                "Suspicious input pattern recorded by the log investigator: "
                f"{pat}"
            ),
            relevance=(
                "Input value appears predominantly in failing requests."
            ),
        ))

    # --- 4. Recurrence fact (only when more than one distinct failure) ---
    recurrence_count = getattr(findings, "recurrence_count", None)
    if isinstance(recurrence_count, int) and recurrence_count > 1:
        window = getattr(findings, "recurrence_window_seconds", None)
        observation = (
            f"Error recurred across {recurrence_count} distinct "
            "failing request_id(s)"
        )
        if window is not None:
            observation += f" over {window} second(s)"
        else:
            observation += " (recurrence window not recorded)"
        observation += "."
        out.append(Evidence(
            source=SOURCE_LOG,
            location=f"{log_path} (recurrence)",
            observation=observation,
            relevance=(
                "More than one distinct failing request id in the log window."
            ),
        ))

    # --- 5. Stack-trace presence fact ---
    if getattr(findings, "stack_trace_present", False):
        stack = getattr(findings, "stack_trace", None)
        stack_text = stack if isinstance(stack, str) else ""
        observation = "Stack trace present in the log."
        exc_names = _EXC_LINE_RE.findall(stack_text)
        if exc_names:
            observation += f" Exception type: {exc_names[-1]}."
        else:
            m = _EXC_NAME_RE.search(stack_text)
            if m:
                observation += f" Exception type: {m.group(1)}."
        frame = _FRAME_RE.search(stack_text)
        if frame:
            observation += (
                f' First frame: File "{frame.group(1)}", '
                f"line {frame.group(2)}."
            )
        out.append(Evidence(
            source=SOURCE_LOG,
            location=f"{log_path} (stack trace)",
            observation=observation,
            relevance=(
                "Findings record a stack trace alongside the log entries."
            ),
        ))

    return out


def from_git_findings(findings) -> list[Evidence]:
    """
    Convert a ``GitFindings``-shaped object into git-sourced evidence.

    Emits one :class:`Evidence` per entry in ``suspicious_commits``.  The
    observation states the commit factually (short sha, message, changed
    files); the relevance is the commit's ``relevance_reasons`` joined into
    one string.  ``top_suspect`` is never emitted separately — it duplicates
    an entry of ``suspicious_commits``.  Returns ``[]`` for ``None`` or an
    empty ``suspicious_commits`` list.
    """
    if findings is None:
        return []

    out: list[Evidence] = []
    for commit in _entries(findings, "suspicious_commits"):
        short_sha = (
            _text(getattr(commit, "short_sha", None))
            or _text(getattr(commit, "sha", None))
            or _UNKNOWN_SHA
        )
        author = _text(getattr(commit, "author", None))
        date_iso = _iso(getattr(commit, "date", None))
        message = _text(getattr(commit, "message", None))
        files = [
            _text(f) for f in _entries(commit, "changed_files") if _text(f)
        ]

        who = []
        if author:
            who.append(f"author {author}")
        if date_iso:
            who.append(f"date {date_iso}")
        head = f"Commit {short_sha}" + (f" ({', '.join(who)})" if who else "")
        message_part = f'message "{message}"' if message else "no message recorded"
        files_part = (
            "changed files: " + ", ".join(files) if files
            else "changed files: none recorded"
        )
        observation = f"{head}: {message_part}; {files_part}."

        reasons = [
            _text(r) for r in _entries(commit, "relevance_reasons") if _text(r)
        ]
        if reasons:
            relevance = "; ".join(reasons)
        else:
            score = getattr(commit, "relevance_score", None)
            if score is not None:
                relevance = f"Flagged with relevance score {score}."
            else:
                relevance = "No relevance reasons recorded."

        out.append(Evidence(
            source=SOURCE_GIT,
            location=f"commit {short_sha}",
            observation=observation,
            relevance=relevance,
        ))

    return out


def from_code_findings(findings) -> list[Evidence]:
    """
    Convert a ``CodeFindings``-shaped object into code-sourced evidence.

    Emits one :class:`Evidence` per entry in ``observations`` with location
    ``f"{file}:{start_line}"``; the agent-authored observation and relevance
    strings are passed through verbatim.  Returns ``[]`` for ``None`` or an
    empty ``observations`` list.
    """
    if findings is None:
        return []

    out: list[Evidence] = []
    for obs in _entries(findings, "observations"):
        file = _text(getattr(obs, "file", None))
        observation = _text(getattr(obs, "observation", None))
        if not file or not observation:
            continue
        start_line = getattr(obs, "start_line", None)
        location = f"{file}:{start_line}" if start_line is not None else file
        relevance = _text(getattr(obs, "relevance", None)) or (
            "No relevance note recorded in the code findings."
        )
        out.append(Evidence(
            source=SOURCE_CODE,
            location=location,
            observation=observation,
            relevance=relevance,
        ))

    return out


def from_test_findings(findings) -> list[Evidence]:
    """
    Convert a ``TestInvestigatorFindings``-shaped object into test-sourced
    evidence.

    Emits one :class:`Evidence` per entry in ``all_relevant_tests``
    (location ``f"{file}:{start_line}"``, observation naming the test, the
    scenario it covers, and the ``covers_affected_function`` fact) and one
    per entry in ``missing_scenarios`` (location
    ``f"{test_root} (coverage gap)"``).  ``covered_scenarios`` is not
    emitted — it is redundant with the per-test evidence.  Returns ``[]``
    for ``None`` or empty inputs.
    """
    if findings is None:
        return []

    out: list[Evidence] = []
    test_root = _text(getattr(findings, "test_root", None)) or _UNKNOWN_TEST_ROOT
    incident_relevance = _text(
        getattr(findings, "relevance_to_incident", None)
    )

    # --- Relevant test cases ---
    for tc in _entries(findings, "all_relevant_tests"):
        name = (
            _text(getattr(tc, "qualified_name", None))
            or _text(getattr(tc, "name", None))
            or "<unnamed test>"
        )
        file = _text(getattr(tc, "file", None)) or "<test file unknown>"
        start_line = getattr(tc, "start_line", None)
        location = f"{file}:{start_line}" if start_line is not None else file

        scenario = _text(getattr(tc, "scenario_description", None))
        covers = bool(getattr(tc, "covers_affected_function", False))
        if scenario:
            observation = (
                f"Test {name} exercises the scenario: {scenario} "
                f"(covers_affected_function={covers})."
            )
        else:
            observation = (
                f"Test {name} recorded with covers_affected_function={covers}."
            )

        if covers:
            relevance = (
                "Test exercises a function listed among the affected "
                "functions."
            )
        else:
            relevance = (
                "Test flagged relevant by the test investigator without "
                "exercising an affected function."
            )
        if incident_relevance:
            relevance += (
                f" Findings rate relevance to the incident as "
                f"{incident_relevance}."
            )

        out.append(Evidence(
            source=SOURCE_TEST,
            location=location,
            observation=observation,
            relevance=relevance,
        ))

    # --- Coverage gaps (missing scenarios) ---
    for gap in _entries(findings, "missing_scenarios"):
        gap_text = str(gap)
        if not _text(gap_text):
            continue
        relevance = "Scenario not exercised by any test under the test root."
        if incident_relevance:
            relevance += (
                f" Findings rate relevance to the incident as "
                f"{incident_relevance}."
            )
        out.append(Evidence(
            source=SOURCE_TEST,
            location=f"{test_root} (coverage gap)",
            observation=f"Coverage gap: {gap_text}",
            relevance=relevance,
        ))

    return out


# ---------------------------------------------------------------------------
# Public API: aggregation and deduplication
# ---------------------------------------------------------------------------

def collect(
    log_findings=None,
    git_findings=None,
    code_findings=None,
    test_findings=None,
) -> list[Evidence]:
    """
    Run all four converters in the fixed order log → git → code → test and
    return the deduplicated concatenation.

    Any argument may be ``None``; only the supplied sources contribute
    evidence.  The result never repeats a ``(source, location, observation)``
    tuple (see :func:`dedupe`).
    """
    evidence: list[Evidence] = []
    evidence.extend(from_log_findings(log_findings))
    evidence.extend(from_git_findings(git_findings))
    evidence.extend(from_code_findings(code_findings))
    evidence.extend(from_test_findings(test_findings))
    return dedupe(evidence)


def dedupe(evidence: list[Evidence]) -> list[Evidence]:
    """
    Remove duplicate evidence entries, preserving first-seen order.

    Two entries are duplicates when their normalized
    ``(source, location, observation)`` tuple is identical — each text
    component is casefolded and its whitespace runs are collapsed to single
    spaces.  ``relevance`` is not part of the key.  The first entry with a
    given key is kept; later duplicates are dropped.
    """
    if not evidence:
        return []

    seen: set[tuple[str, str, str]] = set()
    result: list[Evidence] = []
    for ev in evidence:
        key = (
            _norm(getattr(ev, "source", None)),
            _norm(getattr(ev, "location", None)),
            _norm(getattr(ev, "observation", None)),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(ev)
    return result
