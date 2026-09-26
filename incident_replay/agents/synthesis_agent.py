"""
Evidence Synthesizer agent.

Purpose
-------
Merge the factual output of the four investigator agents — log, git, code
and test — together with the normalized evidence list produced by
:mod:`incident_replay.analysis.evidence` into one deterministic
:class:`SynthesisResult`: a labelled hypothesis, the files and functions
it implicates, the commit it flags, the evidence that supports it, the
evidence that contradicts it, a numeric confidence, and the open
questions that remain.

Observation vs hypothesis
-------------------------
Every investigator records observations only; this module is the single
stage allowed to state a causal claim.  Each causal statement is prefixed
with ``"Hypothesis: "`` and is assembled exclusively from values copied
out of the inputs (field name, incident value, error type, endpoint, file
path, function name, commit sha) — nothing is invented, defaulted, or
imported from outside the supplied findings.  Files, functions and
commits are only named when a *supporting* evidence entry backs them, so
a reader can always re-read the fact that justifies the sentence.  When
fewer than two distinct sources corroborate the correlation, the
hypothesis states plainly that the evidence is insufficient, and
confidence is capped below the moderate band.

Correlation rules (deterministic)
---------------------------------
log   — the dominant error signature, the suspicious ``key=value``
        input patterns, the affected endpoints, the first error time and
        the error count establish *what* failed and *when*; log evidence
        mentioning any of those facts supports the hypothesis.
code  — observations whose text mentions the same field token as the log
        supply ``affected_files`` / ``affected_functions``; their
        evidence entries support the hypothesis.
git   — the best ``suspicious_commits`` entry that either touches an
        affected file or carries a null-safety-removal relevance reason
        supplies ``suspicious_commit``; its evidence supports the
        hypothesis unless a conflict class below fires.
test  — ``missing_scenarios`` (coverage gaps) support the hypothesis;
        tests that already exercise the incident field/value contradict
        it.

Confidence rubric
-----------------
Base, by the number of distinct supporting sources (log/git/code/test)::

    0 sources -> 0.00     2 sources -> 0.55
    1 source  -> 0.35     3 sources -> 0.70
    4 sources -> 0.80

Adjustments (each one recorded in ``confidence_reasons``)::

    +0.10  suspicious commit corroborated: it touches an affected file,
           a code observation matches the incident field, and no
           conflict class fires against it
    +0.05  the log records a dominant error signature
    +0.05  the test findings record a coverage gap (missing scenario)
    +0.05  a code observation reports the field used without a
           null-safety guard
    -0.15  per detected conflict class (a), (b) or (c)

Caps, clamps and thresholds::

    clamp                    -> [0.00, 0.90]  (correlation is never certainty;
                               only the verification stage could justify 1.00)
    no supporting evidence   -> 0.00
    fewer than 2 sources     -> capped at 0.45 (always below 0.50)
    >= 0.75 high | >= 0.50 moderate | >= 0.25 low | else insufficient

Conflict classes
----------------
(a) tests already exercise the incident field/value — a relevant test
    passes the very field and value of the incident, so the supposedly
    missing scenario is covered yet the incident happened anyway; that
    test's evidence moves to ``conflicting_evidence``.
(b) the first error precedes the change — the log's first error time is
    earlier than the selected commit's date (or earlier than a
    deployment time when the log findings expose one), so the change
    cannot have introduced a failure that was already occurring; the
    commit's evidence moves to ``conflicting_evidence``.
(c) the commit is uncorroborated — it touches none of the affected
    files, or no code observation mentions the incident field, so git
    and code do not corroborate each other; the commit's evidence moves
    to ``conflicting_evidence``.

Each detected class lowers confidence by 0.15 and appends at least one
entry to ``uncertainty``; classes (b) and (c) additionally move the
commit's evidence to ``conflicting_evidence`` and keep that commit out of
``root_cause``, while class (a) moves the contradicting test evidence and
leaves an otherwise-corroborated commit untouched.  ``supporting_evidence``
and ``conflicting_evidence`` are always disjoint and always subsets (by
equality) of the input list.

LLM / fallback design
---------------------
``llm_backend`` defaults to ``None``: the module performs no network
access, no file writes, no randomness, and never requires a model.  When
a callable is supplied it is invoked exactly once as
``llm_backend(evidence, context)`` and may return a dict with the
optional keys ``root_cause``, ``affected_files``,
``affected_functions`` and ``suspicious_commit`` — or ``None``.  Only
claims traceable to the supplied evidence are adopted: every file,
function or commit named by the model must be backed by an entry in
``supporting_evidence``.  If the call raises, returns ``None`` or a
non-dict, yields no usable ``root_cause``, or fewer than two distinct
sources support the correlation, the deterministic result is returned
with ``used_llm=False``.  ``confidence`` is always computed from the
evidence alone, so a model can never inflate it.

Missing input
-------------
``run`` never raises for absent, empty or partially populated findings;
it returns a low-confidence result with explanatory ``uncertainty``
entries instead.  :class:`SynthesisAgentError` is reserved for a
structural mistake in an explicit argument (an ``evidence`` argument that
is neither ``None`` nor an iterable of evidence entries).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from incident_replay.analysis.evidence import (
    SOURCE_CODE,
    SOURCE_GIT,
    SOURCE_LOG,
    SOURCE_TEST,
    collect as collect_evidence,
)
from incident_replay.models.schemas import Evidence

__all__ = [
    "SynthesisAgentError",
    "SynthesisResult",
    "run",
]


# ---------------------------------------------------------------------------
# Public error type and result data class
# ---------------------------------------------------------------------------

class SynthesisAgentError(ValueError):
    """
    Raised when an explicitly supplied argument has an unusable structure.

    Missing, empty or partially populated *findings* never raise — they
    produce a low-confidence :class:`SynthesisResult` with explanatory
    ``uncertainty`` entries instead.  This error is reserved for a
    structural mistake in an explicit argument (an ``evidence`` argument
    that is neither ``None`` nor an iterable of evidence entries).
    """


@dataclass
class SynthesisResult:
    """
    Outcome of evidence synthesis: one labelled hypothesis plus the
    evidence, confidence and open questions that back it.

    ``root_cause`` always starts with ``"Hypothesis: "``.  Files,
    functions and commits named inside it are guaranteed to appear in
    ``supporting_evidence``.  ``confidence`` is derived from the evidence
    alone using the rubric documented in the module docstring.
    """

    root_cause: str
    affected_files: list[str]
    affected_functions: list[str]
    suspicious_commit: str | None
    supporting_evidence: list[Evidence]
    conflicting_evidence: list[Evidence]
    confidence: float
    confidence_reasons: list[str]
    uncertainty: list[str]
    used_llm: bool


# ---------------------------------------------------------------------------
# Confidence rubric constants (documented in the module docstring)
# ---------------------------------------------------------------------------

BASE_BY_SOURCE_COUNT: dict[int, float] = {0: 0.00, 1: 0.35, 2: 0.55, 3: 0.70, 4: 0.80}
BONUS_COMMIT_CORROBORATED = 0.10
BONUS_ERROR_SIGNATURE = 0.05
BONUS_COVERAGE_GAP = 0.05
BONUS_UNGUARDED_FIELD = 0.05
PENALTY_PER_CONFLICT = 0.15
CAP_SINGLE_SOURCE = 0.45
# Correlation alone never reaches certainty: no hypothesis has been
# confirmed by the verification stage (regression test run) yet.
MAX_CONFIDENCE = 0.90

BAND_HIGH = 0.75
BAND_MODERATE = 0.50
BAND_LOW = 0.25

CONFLICT_A = "a-tests-cover-incident-value"
CONFLICT_B = "b-error-precedes-change"
CONFLICT_C = "c-commit-uncorroborated"

_ALL_SOURCES = (SOURCE_LOG, SOURCE_GIT, SOURCE_CODE, SOURCE_TEST)


# ---------------------------------------------------------------------------
# Text / structure helpers
# ---------------------------------------------------------------------------

_NULL_WORDS = frozenset({"none", "null", "nil", "undefined", "<none>", "n/a", "na"})

_FIELD_STOPWORDS = frozenset({
    "none", "null", "nil", "field", "value", "values", "param", "parameter",
    "argument", "scenario", "test", "this", "that", "the", "incident",
    "function", "commit", "version",
})

_LOG_KV_STOPWORDS = frozenset({
    "endpoint", "request", "request_id", "response", "status", "version",
    "commit", "branch", "host", "error", "level", "logger", "order_id",
    "customer_id", "total", "timestamp", "time", "first", "last", "count",
    "method", "path", "http",
})

_KV_RES = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_]\w*)=([^\s;)\]]+)")
_FIELD_QUOTED_RES = (
    re.compile(r"\bfield\s+'([A-Za-z_]\w*)'", re.I),
    re.compile(r"\bparameter\s+'([A-Za-z_]\w*)'", re.I),
    re.compile(
        r"'([A-Za-z_]\w*)'\s+(?:is|was|are|were)\s+(?:never\s+)?"
        r"(?:tested|passed|supplied|referenced|used|exercised|given)",
        re.I,
    ),
    re.compile(r"'([A-Za-z_]\w*)'\s+appears\s+in\s+the\s+error\s+signature", re.I),
    re.compile(r"error signature\(s\)?[:\s]+([A-Za-z_]\w*)", re.I),
)
_ERROR_TYPE_RES = re.compile(r"^\s*([\w.]+(?:Error|Exception))\b")
_ENDPOINT_RES = re.compile(r"\bEndpoint\s+([A-Za-z]+\s+/\S+?):")
_FIRST_ERROR_RES = re.compile(
    r"first error (?:at\s+)?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)"
)
_ERROR_COUNT_RES = re.compile(r"log contains (\d+) error line")
_GAP_PREFIX = "coverage gap:"
_COMMIT_DATE_RES = re.compile(
    r"\bdate\s+(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)"
)
_COMMIT_SHA_RES = re.compile(r"\bcommit\s+([0-9a-fA-F]{4,40})", re.I)
_COMMIT_MESSAGE_RES = re.compile(r'message "(.*?)"')
_CHANGED_FILES_RES = re.compile(r"changed files:\s*(.+?)(?:\.|$)")
_QUALIFIED_NAME_RES = re.compile(r"'([A-Za-z_]\w*\.[A-Za-z_]\w*)'")
_LINE_SUFFIX_RES = re.compile(r":(\d+)$")
_INCIDENT_VALUE_RES = re.compile(r"incident value was\s+([^)]+?)\s*\)")

# Tokens that describe this module's own API rather than incident data;
# they are ignored when checking that model output is evidence-traceable.
_META_TOKENS = frozenset({
    "root_cause", "affected_files", "affected_functions", "suspicious_commit",
    "supporting_evidence", "conflicting_evidence", "confidence_reasons",
    "used_llm", "missing_scenarios", "error_count", "first_error_time",
    "dominant_error_signature", "suspicious_input_patterns",
    "affected_endpoints", "interpretation_hints", "field_values_tested",
    "covers_affected_function", "relevance_reasons", "changed_files",
    "all_relevant_tests", "test_root", "repo_path", "log_path",
    "start_line", "end_line", "short_sha",
})

_FILE_TOKEN_RES = re.compile(
    r"[A-Za-z0-9_./\\-]+\.(?:py|pyi|js|jsx|ts|tsx|java|go|rb|c|cc|cpp|h|"
    r"hpp|rs|php|cs|kt|scala|yaml|yml|json|toml|ini|cfg|conf|sql|sh|txt)\b",
    re.I,
)
_HEX_SHA_RES = re.compile(r"\b[0-9a-f]{7,40}\b", re.I)
_DOTTED_NAME_RES = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")
_SNAKE_NAME_RES = re.compile(r"\b[A-Za-z_]\w*_[A-Za-z0-9_]+\b")


def _text(value) -> str:
    """Coerce *value* to a stripped string (``None`` -> ``""``)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _norm(value) -> str:
    """Casefold text and collapse whitespace runs to single spaces."""
    return " ".join(_text(value).split()).casefold()


def _attr(obj, name, default=None):
    """Duck-typed ``getattr`` that never raises on hostile input."""
    if obj is None:
        return default
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _list_attr(obj, name) -> list:
    """Return ``obj.<name>`` when it is a list/tuple of non-``None`` items."""
    value = _attr(obj, name, None)
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if item is not None]


def _str_attr(obj, name) -> str:
    """Return ``str(obj.<name>).strip()`` or ``""``."""
    return _text(_attr(obj, name, None))


def _distinct(items) -> list:
    """Drop empties and duplicates, preserving first-seen order."""
    out = []
    for item in items:
        if not item:
            continue
        if item not in out:
            out.append(item)
    return out


def _is_evidence_like(item) -> bool:
    """True when *item* exposes the four :class:`Evidence` attributes."""
    return all(
        hasattr(item, name)
        for name in ("source", "location", "observation", "relevance")
    )


def _ev_source(ev) -> str:
    return _text(_attr(ev, "source"))


def _ev_location(ev) -> str:
    return _text(_attr(ev, "location"))


def _ev_observation(ev) -> str:
    return _text(_attr(ev, "observation"))


def _ev_relevance(ev) -> str:
    return _text(_attr(ev, "relevance"))


def _ev_key(ev) -> tuple[str, str, str]:
    """Normalized identity used for set membership and disjointness."""
    return (_norm(_ev_source(ev)), _norm(_ev_location(ev)), _norm(_ev_observation(ev)))


def _ev_raw(ev) -> str:
    """Full unmodified text of an evidence entry (case preserved)."""
    return f"{_ev_location(ev)} {_ev_observation(ev)} {_ev_relevance(ev)}"


def _ev_text(ev) -> str:
    """Case-folded, slash-normalized text of an evidence entry."""
    return _ev_raw(ev).replace("\\", "/").casefold()


def _mentions(text: str, token: str) -> bool:
    """True when *token* occurs in *text* on identifier boundaries."""
    if not token or not text:
        return False
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, text, re.I) is not None


def _is_null_token(value: str) -> bool:
    """True when *value* denotes a missing/null input."""
    return _text(value).strip("'\"").casefold() in _NULL_WORDS


def _as_datetime(value) -> datetime | None:
    """Best-effort conversion of *value* to a ``datetime`` (else ``None``)."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        raw = value.strip()
        if raw.endswith(("Z", "z")):
            raw = raw[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
    return None


def _precedes(a: datetime | None, b: datetime | None) -> bool | None:
    """
    True when *a* is strictly earlier than *b*; ``None`` when unknown.

    Naive timestamps are treated as UTC when compared against aware ones
    so that a mixed comparison never raises.
    """
    if a is None or b is None:
        return None
    try:
        if a.tzinfo is None and b.tzinfo is not None:
            a = a.replace(tzinfo=timezone.utc)
        elif b.tzinfo is None and a.tzinfo is not None:
            b = b.replace(tzinfo=timezone.utc)
        return a < b
    except Exception:
        return None


def _iso(value) -> str:
    """Render a timestamp deterministically (``""`` for ``None``)."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
            return value.strftime("%Y-%m-%dT%H:%M:%SZ")
        return value.strftime("%Y-%m-%dT%H:%M:%S")
    return str(value)


def _band(score: float) -> str:
    """Map a confidence score to its reporting band."""
    if score >= BAND_HIGH:
        return "high"
    if score >= BAND_MODERATE:
        return "moderate"
    if score >= BAND_LOW:
        return "low"
    return "insufficient"


def _paths_overlap(a: str, b: str) -> bool:
    """True when two path strings share their trailing path components."""
    pa = [p for p in _text(a).replace("\\", "/").casefold().split("/") if p and p != "."]
    pb = [p for p in _text(b).replace("\\", "/").casefold().split("/") if p and p != "."]
    if not pa or not pb:
        return False
    n = min(len(pa), len(pb))
    return pa[len(pa) - n:] == pb[len(pb) - n:]


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

def _normalize_evidence(evidence, log_findings, git_findings, code_findings,
                        test_findings, notes: list[str]) -> list:
    """
    Return the evidence list to correlate against.

    ``None`` triggers :func:`incident_replay.analysis.evidence.collect`;
    an iterable is taken as-is (non-evidence entries are dropped with a
    note).  Collection failures degrade to an empty list instead of
    raising.
    """
    if evidence is None:
        try:
            collected = collect_evidence(
                log_findings, git_findings, code_findings, test_findings
            )
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(
                f"Evidence collection failed ({type(exc).__name__}: {exc}); "
                "continuing without evidence."
            )
            return []
        if isinstance(collected, (str, bytes, dict)) or not hasattr(collected, "__iter__"):
            notes.append(
                "Evidence collection returned an unexpected type; continuing "
                "without evidence."
            )
            return []
        items = list(collected)
    else:
        if isinstance(evidence, (str, bytes, bytearray, dict)) or not hasattr(evidence, "__iter__"):
            raise SynthesisAgentError(
                "evidence must be an iterable of Evidence entries or None."
            )
        items = list(evidence)

    kept = [item for item in items if _is_evidence_like(item)]
    dropped = len(items) - len(kept)
    if dropped:
        notes.append(
            f"{dropped} supplied evidence entr{'y' if dropped == 1 else 'ies'} "
            "were skipped because they are not Evidence objects."
        )
    return kept


# ---------------------------------------------------------------------------
# Fact extraction (findings first, evidence text as fallback)
# ---------------------------------------------------------------------------

def _error_type_from(signature: str) -> str:
    """Return the exception class named by *signature* (or ``""``)."""
    signature = _text(signature)
    if not signature:
        return ""
    match = _ERROR_TYPE_RES.match(signature)
    if match:
        return match.group(1)
    if ":" in signature:
        return signature.split(":", 1)[0].strip()
    return ""


def _log_facts(log_findings, log_evidence: list) -> dict:
    """
    Extract the log-side facts used for correlation.

    Falls back to parsing the supplied log evidence when the findings
    object is absent, so an evidence-only call still works.
    """
    signature = _str_attr(log_findings, "dominant_error_signature")
    patterns = _distinct([
        _text(p).rstrip(".,;")
        for p in _list_attr(log_findings, "suspicious_input_patterns")
        if "=" in _text(p)
    ])
    endpoints = _distinct([
        _text(e) for e in _list_attr(log_findings, "affected_endpoints") if _text(e)
    ])
    first_error = _as_datetime(_attr(log_findings, "first_error_time", None))

    count_raw = _attr(log_findings, "error_count", None)
    error_count = count_raw if isinstance(count_raw, int) else None

    for ev in log_evidence:
        observation = _ev_observation(ev)
        location = _ev_location(ev)
        blob = f"{location} {observation}"
        if not signature:
            match = re.search(r"dominant error signature:\s*([^;.]+)", observation)
            if match:
                signature = match.group(1).strip()
        if not endpoints:
            match = _ENDPOINT_RES.search(observation)
            if match:
                endpoints.append(match.group(1).rstrip("."))
        if error_count is None:
            match = _ERROR_COUNT_RES.search(observation)
            if match:
                error_count = int(match.group(1))
        if first_error is None:
            match = _FIRST_ERROR_RES.search(blob)
            if match:
                first_error = _as_datetime(match.group(1))
        for match in _KV_RES.finditer(blob):
            key = match.group(1)
            if key.casefold() in _LOG_KV_STOPWORDS:
                continue
            token = match.group(0).rstrip(".,;)")
            if token not in patterns:
                patterns.append(token)

    return {
        "signature": signature,
        "error_type": _error_type_from(signature),
        "patterns": patterns,
        "endpoints": endpoints,
        "first_error": first_error,
        "error_count": error_count,
    }


def _code_views(code_findings, code_evidence: list) -> list[dict]:
    """
    Return one dict per correlated code observation.

    Falls back to parsing code evidence (file from the ``file:line``
    location, qualified function from quoted text) when no observations
    were supplied.
    """
    views: list[dict] = []
    for obs in _list_attr(code_findings, "observations"):
        file = _str_attr(obs, "file")
        function = _str_attr(obs, "function")
        observation = _str_attr(obs, "observation")
        relevance = _str_attr(obs, "relevance")
        if not (file or function or observation):
            continue
        start_line = _attr(obs, "start_line", None)
        location = f"{file}:{start_line}" if (file and start_line is not None) else file
        views.append({
            "file": file,
            "function": function,
            "location": location,
            "observation": observation,
            "relevance": relevance,
            "blob": f"{observation} {relevance}",
        })
    if views:
        return views

    for ev in code_evidence:
        location = _ev_location(ev)
        observation = _ev_observation(ev)
        relevance = _ev_relevance(ev)
        match = _LINE_SUFFIX_RES.search(location)
        file = location[: match.start()] if match else location
        if not file:
            continue
        name_match = _QUALIFIED_NAME_RES.search(f"{observation} {relevance}")
        views.append({
            "file": file,
            "function": name_match.group(1) if name_match else "",
            "location": location,
            "observation": observation,
            "relevance": relevance,
            "blob": f"{observation} {relevance}",
        })
    return views


def _commit_views(git_findings, git_evidence: list) -> list[dict]:
    """
    Return one dict per suspicious commit (findings first, evidence text
    as fallback) with ``sha``, ``date``, ``message``, ``changed_files``
    and ``reasons``.
    """
    views: list[dict] = []
    for commit in _list_attr(git_findings, "suspicious_commits"):
        sha = _str_attr(commit, "short_sha") or _str_attr(commit, "sha")
        if not sha:
            continue
        views.append({
            "sha": sha,
            "date": _as_datetime(_attr(commit, "date", None)),
            "message": _str_attr(commit, "message"),
            "changed_files": _distinct([
                _text(f) for f in _list_attr(commit, "changed_files") if _text(f)
            ]),
            "reasons": _distinct([
                _text(r) for r in _list_attr(commit, "relevance_reasons") if _text(r)
            ]),
        })
    if views:
        return views

    for ev in git_evidence:
        observation = _ev_observation(ev)
        location = _ev_location(ev)
        match = _COMMIT_SHA_RES.search(location) or _COMMIT_SHA_RES.search(observation)
        if not match:
            continue
        date_match = _COMMIT_DATE_RES.search(observation)
        files_match = _CHANGED_FILES_RES.search(observation)
        changed: list[str] = []
        if files_match:
            raw = files_match.group(1).strip()
            if "none recorded" not in raw.casefold():
                changed = _distinct([f.strip() for f in raw.split(",") if f.strip()])
        message_match = _COMMIT_MESSAGE_RES.search(observation)
        views.append({
            "sha": match.group(1),
            "date": _as_datetime(date_match.group(1)) if date_match else None,
            "message": message_match.group(1) if message_match else "",
            "changed_files": changed,
            "reasons": [_ev_relevance(ev)] if _ev_relevance(ev) else [],
        })
    return views


def _missing_scenarios(test_findings, test_evidence: list) -> list[str]:
    """Coverage gaps from the test findings, else from coverage-gap evidence."""
    gaps = _distinct([
        _text(g) for g in _list_attr(test_findings, "missing_scenarios") if _text(g)
    ])
    if gaps:
        return gaps
    derived: list[str] = []
    for ev in test_evidence:
        observation = _ev_observation(ev)
        if observation.casefold().startswith(_GAP_PREFIX):
            gap = observation.split(":", 1)[1].strip()
            if gap:
                derived.append(gap)
    return _distinct(derived)


# ---------------------------------------------------------------------------
# Field-token correlation
# ---------------------------------------------------------------------------

def _add_candidate(candidates: list[str], token: str) -> None:
    """Append a field-name candidate when it looks like an identifier."""
    token = _text(token).strip("'\"").rstrip(".,;")
    if not token or not re.fullmatch(r"[A-Za-z_]\w*", token):
        return
    if token.casefold() in _FIELD_STOPWORDS or token.casefold() in _NULL_WORDS:
        return
    if token not in candidates:
        candidates.append(token)


def _field_candidates(blobs: dict[str, str], log_facts: dict) -> list[str]:
    """
    Derive candidate field names from the four sources.

    Log ``key=value`` patterns contribute their key; every source
    contributes quoted field/parameter mentions.  Order is source order
    so that ties resolve deterministically.
    """
    candidates: list[str] = []
    for pattern in log_facts["patterns"]:
        if "=" in pattern:
            _add_candidate(candidates, pattern.split("=", 1)[0])
    for source in _ALL_SOURCES:
        blob = blobs.get(source, "")
        if not blob:
            continue
        for regex in _FIELD_QUOTED_RES:
            for match in regex.findall(blob):
                _add_candidate(candidates, match)
    return candidates


def _choose_field_token(candidates: list[str], blobs: dict[str, str]) -> str | None:
    """Pick the candidate mentioned by the largest number of sources."""
    best: str | None = None
    best_score = 0
    for candidate in candidates:
        score = sum(1 for blob in blobs.values() if _mentions(blob, candidate))
        if score > best_score:
            best, best_score = candidate, score
    return best


def _incident_value(field_token: str | None, patterns: list[str],
                    missing_scenarios: list[str]) -> str | None:
    """Return the incident value recorded for *field_token*, if any."""
    if not field_token:
        return None
    for pattern in patterns:
        if "=" not in pattern:
            continue
        key, value = pattern.split("=", 1)
        if key.strip() == field_token:
            return value.strip().rstrip(".,;")
    for gap in missing_scenarios:
        match = _INCIDENT_VALUE_RES.search(gap)
        if match and _mentions(gap, field_token):
            return match.group(1).strip().strip("'\"")
    return None


def _touches_affected(view: dict, affected_files: list[str]) -> bool:
    """True when a commit's changed files overlap the affected files."""
    changed = view.get("changed_files") or []
    if not changed or not affected_files:
        return False
    return any(
        _paths_overlap(changed_file, target)
        for changed_file in changed
        for target in affected_files
    )


def _removes_null_safety(view: dict) -> bool:
    """True when one of the commit's relevance reasons removes a null guard."""
    for reason in view.get("reasons") or []:
        lowered = _text(reason).casefold()
        if "null-safety" in lowered or "null safety" in lowered:
            return True
    return False


def _git_evidence_for(sha: str, git_evidence: list):
    """Return the evidence entry that documents commit *sha*, else ``None``."""
    target = _norm(f"commit {sha}")
    for ev in git_evidence:
        if _norm(_ev_location(ev)) == target:
            return ev
    probe = _text(sha).casefold()
    if probe:
        for ev in git_evidence:
            if probe in _ev_text(ev):
                return ev
    return None


# ---------------------------------------------------------------------------
# Test coverage conflicts (conflict class a)
# ---------------------------------------------------------------------------

def _test_coverage_conflicts(field_token: str | None, incident_value: str | None,
                             test_findings, test_evidence: list) -> list[tuple[str, str]]:
    """
    Return ``(test_location, note)`` pairs for tests that already exercise
    the incident field/value (conflict class (a)).

    Inspects ``TestCase.field_values_tested`` when available and falls
    back to scanning non-gap test evidence otherwise.
    """
    if not field_token:
        return []

    pairs: list[tuple[str, str]] = []
    inspected = False
    for tc in _list_attr(test_findings, "all_relevant_tests"):
        field_values = _attr(tc, "field_values_tested", None)
        if not isinstance(field_values, dict):
            continue
        inspected = True
        if field_token not in field_values:
            continue
        tested = [_text(v) for v in (field_values.get(field_token) or [])]
        label = (
            _str_attr(tc, "qualified_name")
            or _str_attr(tc, "name")
            or "<unnamed test>"
        )
        file = _str_attr(tc, "file")
        start_line = _attr(tc, "start_line", None)
        location = f"{file}:{start_line}" if (file and start_line is not None) else file

        if incident_value is None:
            hit = True
            detail = (
                f"field '{field_token}' (the incident value was not recorded)"
            )
        elif _is_null_token(incident_value):
            hit = any(_is_null_token(v) for v in tested)
            detail = (
                f"field '{field_token}' with a null value, the value recorded "
                "in the incident"
            )
        else:
            wanted = incident_value.strip("'\"")
            hit = any(v.strip("'\"") == wanted for v in tested)
            detail = (
                f"field '{field_token}' with the incident value {incident_value}"
            )
        if hit:
            pairs.append((
                location,
                f"Conflict (a): test {label} already exercises {detail}, yet "
                "the incident still occurred.",
            ))

    if inspected:
        return pairs

    for ev in test_evidence:
        observation = _ev_observation(ev)
        if observation.casefold().startswith(_GAP_PREFIX):
            continue
        blob = _ev_text(ev)
        if not _mentions(blob, field_token):
            continue
        if incident_value is None:
            hit = True
            detail = (
                f"field '{field_token}' (the incident value was not recorded)"
            )
        elif _is_null_token(incident_value):
            hit = _mentions(blob, "null") or _mentions(blob, "none")
            detail = (
                f"field '{field_token}' with a null value, the value recorded "
                "in the incident"
            )
        else:
            hit = _mentions(blob, incident_value)
            detail = f"field '{field_token}' with the incident value {incident_value}"
        if hit:
            pairs.append((
                _ev_location(ev),
                f"Conflict (a): the test evidence at {_ev_location(ev)} already "
                f"exercises {detail}, yet the incident still occurred.",
            ))
    return pairs


# ---------------------------------------------------------------------------
# Evidence traceability
# ---------------------------------------------------------------------------

def _file_backed(file_name: str, supporting: list[Evidence]) -> bool:
    """True when *file_name* appears in a supporting evidence entry."""
    token = _text(file_name).replace("\\", "/").casefold()
    if not token:
        return False
    return any(token in _ev_text(ev) for ev in supporting)


def _function_backed(function: str, supporting: list[Evidence]) -> bool:
    """True when *function* (or its last dotted segment) is evidenced."""
    token = _text(function).casefold()
    if not token:
        return False
    last = token.split(".")[-1]
    return any(
        token in blob or (last and last in blob)
        for blob in (_ev_text(ev) for ev in supporting)
    )


def _commit_backed(sha: str, supporting: list[Evidence]) -> bool:
    """True when git-sourced supporting evidence names *sha*."""
    token = _text(sha).casefold()
    if not token:
        return False
    return any(
        _ev_source(ev) == SOURCE_GIT and token in _ev_text(ev)
        for ev in supporting
    )


def _untraceable(text: str, supporting: list[Evidence]) -> list[str]:
    """
    Return file/function/commit tokens in *text* that no supporting
    evidence entry backs.  An empty result means every name the model
    used can be re-read from the evidence.
    """
    blobs = [_ev_text(ev) for ev in supporting]
    found: list[str] = []
    found.extend(_FILE_TOKEN_RES.findall(text))
    found.extend(_HEX_SHA_RES.findall(text))
    for match in _DOTTED_NAME_RES.findall(text):
        if all(len(part) >= 2 for part in match.split(".")):
            found.append(match)
    for match in _SNAKE_NAME_RES.findall(text):
        if match.casefold() not in _META_TOKENS:
            found.append(match)

    bad: list[str] = []
    for token in found:
        probe = token.replace("\\", "/").casefold()
        if probe and not any(probe in blob for blob in blobs) and token not in bad:
            bad.append(token)
    return bad


def _str_list(value) -> list[str]:
    """Coerce an LLM list claim into a list of strings."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [item for item in (_text(v) for v in value) if item]
    return []


def _try_adopt_llm(llm_backend, evidence_list: list, context: dict,
                   supporting: list[Evidence]) -> tuple[dict | None, list[str]]:
    """
    Call *llm_backend* once and adopt only evidence-traceable claims.

    Returns ``(claims, notes)``; ``claims`` is ``None`` when the output
    is unusable and the deterministic hypothesis must be kept.
    """
    try:
        raw = llm_backend(evidence_list, context)
    except Exception as exc:
        return None, [
            f"LLM backend raised {type(exc).__name__}: {exc}; using the "
            "deterministic hypothesis."
        ]
    if raw is None:
        return None, ["LLM backend returned None; using the deterministic hypothesis."]
    if not isinstance(raw, dict):
        return None, [
            f"LLM backend returned {type(raw).__name__} instead of a dict; "
            "using the deterministic hypothesis."
        ]
    claim = raw.get("root_cause")
    if not isinstance(claim, str) or not claim.strip():
        return None, [
            "LLM backend returned no usable root_cause claim; using the "
            "deterministic hypothesis."
        ]

    text = claim.strip()
    bad = _untraceable(text, supporting)
    if bad:
        return None, [
            "LLM root_cause named item(s) absent from supporting evidence ("
            + ", ".join(bad)
            + "); using the deterministic hypothesis."
        ]
    if not text.casefold().startswith("hypothesis:"):
        text = f"Hypothesis: {text}"

    raw_files = _str_list(raw.get("affected_files"))
    raw_functions = _str_list(raw.get("affected_functions"))
    files = [f for f in raw_files if _file_backed(f, supporting)]
    functions = [f for f in raw_functions if _function_backed(f, supporting)]
    commit = _text(raw.get("suspicious_commit"))
    commit_claimed = bool(commit)
    if commit and not _commit_backed(commit, supporting):
        commit = ""

    claims = {
        "root_cause": text,
        "affected_files": _distinct(files),
        "affected_functions": _distinct(functions),
        "suspicious_commit": commit or None,
    }
    notes = [
        "Adopted the root_cause from the supplied LLM backend; every claim "
        "was checked against supporting evidence and confidence remains "
        "deterministic."
    ]
    if len(files) < len(raw_files):
        notes.append(
            "LLM affected_files claim(s) were dropped as untraceable to "
            "supporting evidence."
        )
    if len(functions) < len(raw_functions):
        notes.append(
            "LLM affected_functions claim(s) were dropped as untraceable to "
            "supporting evidence."
        )
    if commit_claimed and not claims["suspicious_commit"]:
        notes.append(
            "The LLM suspicious_commit claim was dropped as untraceable to "
            "supporting evidence."
        )
    return claims, notes


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def _confidence(supporting_sources: list[str], has_signature: bool, has_gap: bool,
                has_unguarded: bool, commit_corroborated: bool,
                conflict_classes: list[str]) -> tuple[float, list[str]]:
    """
    Apply the documented rubric and return ``(score, reasons)``.

    The score is a pure function of the supporting evidence and the
    detected conflict classes; it is clamped to ``[0.0, MAX_CONFIDENCE]``.
    """
    reasons: list[str] = []
    count = len(supporting_sources)
    base = BASE_BY_SOURCE_COUNT.get(min(count, 4), 0.80)
    labels = ", ".join(supporting_sources) if supporting_sources else "none"
    reasons.append(
        f"Base {base:.2f} from {count} distinct supporting source(s): {labels}."
    )

    score = base
    if commit_corroborated:
        score += BONUS_COMMIT_CORROBORATED
        reasons.append(
            f"+{BONUS_COMMIT_CORROBORATED:.2f}: the suspicious commit touches "
            "an affected file and is matched by a code observation."
        )
    if has_signature:
        score += BONUS_ERROR_SIGNATURE
        reasons.append(
            f"+{BONUS_ERROR_SIGNATURE:.2f}: the log records a dominant error signature."
        )
    if has_gap:
        score += BONUS_COVERAGE_GAP
        reasons.append(
            f"+{BONUS_COVERAGE_GAP:.2f}: the test findings record a coverage gap "
            "for the incident field/value."
        )
    if has_unguarded:
        score += BONUS_UNGUARDED_FIELD
        reasons.append(
            f"+{BONUS_UNGUARDED_FIELD:.2f}: a code observation reports the field "
            "used without a null-safety guard."
        )
    if score > MAX_CONFIDENCE:
        score = MAX_CONFIDENCE
        reasons.append(
            f"Capped at {MAX_CONFIDENCE:.2f}: correlation alone is not "
            "certainty — the hypothesis has not been confirmed by the "
            "verification stage."
        )
    for class_id in conflict_classes:
        score -= PENALTY_PER_CONFLICT
        reasons.append(
            f"-{PENALTY_PER_CONFLICT:.2f}: conflict class detected ({class_id})."
        )

    if count == 0:
        score = 0.0
        reasons.append("No supporting evidence; confidence fixed at 0.00.")
    elif count < 2:
        if score > CAP_SINGLE_SOURCE:
            score = CAP_SINGLE_SOURCE
            reasons.append(
                f"Capped at {CAP_SINGLE_SOURCE:.2f}: fewer than two distinct "
                "sources support the hypothesis, so confidence must stay below 0.50."
            )
        else:
            reasons.append(
                f"Only {count} distinct source(s) support the hypothesis; "
                f"confidence must stay below 0.50 (cap {CAP_SINGLE_SOURCE:.2f})."
            )

    score = round(max(0.0, min(MAX_CONFIDENCE, score)), 4)
    reasons.append(f"Final confidence {score:.2f} (band: {_band(score)}).")
    return score, reasons


# ---------------------------------------------------------------------------
# Hypothesis composition
# ---------------------------------------------------------------------------

def _compose_insufficient(supporting_sources: list[str], field_token: str | None,
                          incident_value: str | None, log_facts: dict,
                          endpoint: str) -> str:
    """Build the hypothesis text used when fewer than two sources agree."""
    count = len(supporting_sources)
    if count == 0:
        detail = (
            "no supporting evidence was collected from the log, git, code or "
            "test sources"
        )
    else:
        detail = (
            f"only {count} distinct supporting source(s) "
            f"({', '.join(supporting_sources)}) corroborate the correlation"
        )

    signals: list[str] = []
    if field_token:
        signal = f"field '{field_token}'"
        if incident_value is not None:
            signal += f" supplied as {incident_value}"
        signals.append(signal)
    if log_facts["error_type"]:
        signals.append(f"error type {log_facts['error_type']}")
    if endpoint:
        signals.append(f"endpoint {endpoint}")
    signal_text = "; ".join(signals) if signals else "no named signal"

    return (
        "Hypothesis: the evidence is insufficient to identify a root cause — "
        f"{detail}, and at least two distinct sources are required. "
        f"Recorded but uncorroborated signals: {signal_text}."
    )


def _compose_hypothesis(*, field_token: str | None, incident_value: str | None,
                        error_type: str, endpoint: str, function: str,
                        file_name: str, commit_sha: str | None,
                        supporting: list[Evidence],
                        supporting_sources: list[str]) -> str:
    """
    Build the hypothesis from input-derived values only.

    A file, function or commit is appended only when a supporting
    evidence entry backs it, so the output honours the traceability rule.
    """
    failure = f"{error_type} failure" if error_type else "recorded failure"
    on_endpoint = f" on {endpoint}" if endpoint else ""

    backed_function = function if _function_backed(function, supporting) else ""
    backed_file = file_name if _file_backed(file_name, supporting) else ""
    backed_commit = commit_sha if _commit_backed(commit_sha or "", supporting) else ""

    if field_token and incident_value is not None:
        cause = f"the field '{field_token}' supplied as {incident_value}"
    elif field_token:
        cause = f"the field '{field_token}'"
    else:
        cause = ""

    if cause:
        head = f"Hypothesis: {cause} is the cause of the {failure}{on_endpoint}."
    elif backed_function:
        head = f"Hypothesis: the {failure}{on_endpoint} originates in {backed_function}."
    elif backed_file:
        head = f"Hypothesis: the {failure}{on_endpoint} originates in {backed_file}."
    else:
        head = (
            f"Hypothesis: the {failure}{on_endpoint} correlates with the evidence "
            f"from {', '.join(supporting_sources)}."
        )

    sentences = [head]
    if cause and (backed_function or backed_file):
        if backed_function and backed_file:
            where = f"{backed_function} ({backed_file})"
        elif backed_function:
            where = backed_function
        else:
            where = backed_file
        sentences.append(f"The failure surfaces in {where}.")
    if backed_commit:
        sentences.append(
            f"Commit {backed_commit} is the change correlated with the failure."
        )
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(log_findings=None, git_findings=None, code_findings=None, test_findings=None,
        *, evidence=None, llm_backend=None) -> SynthesisResult:
    """
    Synthesize the four investigators' findings into a :class:`SynthesisResult`.

    Parameters
    ----------
    log_findings, git_findings, code_findings, test_findings:
        Findings dataclasses returned by the investigator agents, or
        ``None``.  Any subset may be supplied; nothing is required.
    evidence:
        Pre-collected evidence entries.  When ``None`` they are obtained
        from :func:`incident_replay.analysis.evidence.collect`.
    llm_backend:
        Optional callable invoked as ``llm_backend(evidence, context)``
        returning a dict of claims or ``None``.  Unused by default.

    Returns
    -------
    SynthesisResult
        Never raises for missing or empty findings; a low-confidence
        result with explanatory ``uncertainty`` entries is returned
        instead.
    """
    notes: list[str] = []

    evidence_list = _normalize_evidence(
        evidence, log_findings, git_findings, code_findings, test_findings, notes
    )

    by_source: dict[str, list] = {source: [] for source in _ALL_SOURCES}
    for ev in evidence_list:
        source = _ev_source(ev)
        if source in by_source:
            by_source[source].append(ev)
    if not evidence_list:
        notes.append("No evidence was collected from any source.")

    # --- Facts from each investigator ------------------------------------
    log_facts = _log_facts(log_findings, by_source[SOURCE_LOG])
    code_views = _code_views(code_findings, by_source[SOURCE_CODE])
    commit_views = _commit_views(git_findings, by_source[SOURCE_GIT])
    missing_scenarios = _missing_scenarios(test_findings, by_source[SOURCE_TEST])

    blobs: dict[str, str] = {
        SOURCE_LOG: " ".join(
            [log_facts["signature"], *log_facts["patterns"], *log_facts["endpoints"]]
            + [_ev_raw(ev) for ev in by_source[SOURCE_LOG]]
        ),
        SOURCE_GIT: " ".join(
            [f"{reason} {view['message']}" for view in commit_views for reason in view["reasons"]]
            + [_ev_raw(ev) for ev in by_source[SOURCE_GIT]]
        ),
        SOURCE_CODE: " ".join(
            [view["blob"] for view in code_views]
            + [_ev_raw(ev) for ev in by_source[SOURCE_CODE]]
        ),
        SOURCE_TEST: " ".join(
            missing_scenarios + [_ev_raw(ev) for ev in by_source[SOURCE_TEST]]
        ),
    }

    candidates = _field_candidates(blobs, log_facts)
    field_token = _choose_field_token(candidates, blobs)
    findings_supplied = any(
        item is not None
        for item in (log_findings, git_findings, code_findings, test_findings)
    )
    if field_token is None and (findings_supplied or evidence_list):
        notes.append(
            "No incident field token could be derived from the supplied inputs; "
            "correlation fell back to whole-source matching."
        )

    incident_value = _incident_value(
        field_token, log_facts["patterns"], missing_scenarios
    )
    endpoint = log_facts["endpoints"][0] if log_facts["endpoints"] else ""

    # --- Code correlation ------------------------------------------------
    if field_token:
        matched_views = [v for v in code_views if _mentions(v["blob"], field_token)]
    else:
        matched_views = list(code_views)

    stack_functions = {
        view["function"]
        for view in code_views
        if view["function"] and "stack trace" in view["blob"].casefold()
    }

    def _view_rank(view: dict) -> int:
        """Prefer the observation closest to the failure: a function that
        appears in the stack trace, then an unguarded use of the incident
        field, then any field mention."""
        blob = view["blob"].casefold()
        rank = 0
        if view["function"] in stack_functions or "stack trace" in blob:
            rank += 4
        if "no null-safety guard" in blob:
            rank += 2
        if field_token and _mentions(blob, field_token):
            rank += 1
        return rank

    # Relevance order: stack-trace functions first, so the report names the
    # frame nearest the failure rather than the first file alphabetically.
    matched_views = sorted(matched_views, key=_view_rank, reverse=True)

    affected_files = _distinct([v["file"] for v in matched_views if v["file"]])
    affected_functions = _distinct([v["function"] for v in matched_views if v["function"]])

    if not code_views:
        summary_files = _distinct([
            _text(f) for f in _list_attr(code_findings, "affected_files") if _text(f)
        ])
        summary_functions = _distinct([
            _text(f) for f in _list_attr(code_findings, "affected_functions") if _text(f)
        ])
        if summary_files or summary_functions:
            affected_files = summary_files
            affected_functions = summary_functions
            notes.append(
                "No code observations were supplied; affected files and functions "
                "were taken from the code findings summary fields."
            )
    elif field_token and not matched_views:
        notes.append(
            f"No code observation mentions field '{field_token}'; no affected files "
            "or functions were correlated from code."
        )
    elif not field_token and code_views:
        notes.append(
            "No field token was derived, so every code observation was treated "
            "as correlated."
        )

    primary_function = ""
    primary_file = ""
    for view in matched_views:
        if view["function"]:
            primary_function = view["function"]
            primary_file = view["file"]
            break
    if not primary_file:
        for view in matched_views:
            if view["file"]:
                primary_file = view["file"]
                break

    # --- Evidence classification ----------------------------------------
    support_keys: set[tuple[str, str, str]] = set()
    conflict_keys: set[tuple[str, str, str]] = set()
    conflict_classes: list[str] = []

    def _register_conflict(class_id: str, note: str) -> None:
        if class_id not in conflict_classes:
            conflict_classes.append(class_id)
        notes.append(note)

    log_tokens = [
        token for token in (
            field_token, *log_facts["patterns"], log_facts["signature"],
            *log_facts["endpoints"],
        ) if token
    ]
    if log_findings is None and by_source[SOURCE_LOG]:
        notes.append(
            "Log findings were not supplied; the supplied log evidence is taken "
            "at face value."
        )
        selected_log = list(by_source[SOURCE_LOG])
    else:
        selected_log = [
            ev for ev in by_source[SOURCE_LOG]
            if any(_mentions(_ev_raw(ev), token) for token in log_tokens)
        ]
        if not selected_log and (log_facts["error_count"] or 0) > 0:
            selected_log = list(by_source[SOURCE_LOG])
    for ev in selected_log:
        support_keys.add(_ev_key(ev))

    code_lookup: dict[tuple[str, str], tuple[str, str, str]] = {}
    for ev in by_source[SOURCE_CODE]:
        code_lookup.setdefault(
            (_norm(_ev_location(ev)), _norm(_ev_observation(ev))), _ev_key(ev)
        )
    for view in matched_views:
        key = code_lookup.get((_norm(view["location"]), _norm(view["observation"])))
        if key is not None:
            support_keys.add(key)

    unmatched_gaps: list[str] = []
    for gap in missing_scenarios:
        matched_gap = False
        for ev in by_source[SOURCE_TEST]:
            observation = _ev_observation(ev)
            if not observation.casefold().startswith(_GAP_PREFIX):
                continue
            if _norm(gap) in _norm(observation):
                support_keys.add(_ev_key(ev))
                matched_gap = True
        if not matched_gap:
            unmatched_gaps.append(gap)
    if unmatched_gaps and test_findings is not None:
        notes.append(
            f"{len(unmatched_gaps)} coverage-gap finding(s) could not be matched "
            "to the supplied evidence."
        )

    for location, note in _test_coverage_conflicts(
        field_token, incident_value, test_findings, by_source[SOURCE_TEST]
    ):
        _register_conflict(CONFLICT_A, note)
        if location:
            for ev in by_source[SOURCE_TEST]:
                if _norm(_ev_location(ev)) == _norm(location):
                    conflict_keys.add(_ev_key(ev))

    if CONFLICT_A not in conflict_classes:
        for gap in missing_scenarios:
            notes.append(f"Open question: coverage gap — {gap}")

    suspicious_commit: str | None = None
    commit_corroborated = False
    if commit_views:
        qualified = [
            view for view in commit_views
            if _touches_affected(view, affected_files) or _removes_null_safety(view)
        ]
        candidate = qualified[0] if qualified else commit_views[0]
        suspicious_commit = candidate["sha"]
        commit_ev = _git_evidence_for(suspicious_commit, by_source[SOURCE_GIT])

        conflict_b = False
        if log_facts["first_error"] is not None and candidate["date"] is not None:
            if _precedes(log_facts["first_error"], candidate["date"]):
                conflict_b = True
                _register_conflict(
                    CONFLICT_B,
                    f"Conflict (b): the first error at {_iso(log_facts['first_error'])} "
                    f"precedes commit {suspicious_commit} dated "
                    f"{_iso(candidate['date'])}; the change cannot have introduced "
                    "an already-occurring failure.",
                )

        touches = _touches_affected(candidate, affected_files)
        code_matched = bool(field_token) and bool(matched_views)
        conflict_c = not touches or not code_matched
        if conflict_c:
            reasons: list[str] = []
            if not touches:
                listed = ", ".join(affected_files) if affected_files else "none identified"
                reasons.append(f"it touches none of the affected files ({listed})")
            if not code_matched:
                if field_token:
                    reasons.append(f"no code observation mentions field '{field_token}'")
                else:
                    reasons.append(
                        "no incident field token could be derived to correlate "
                        "git with code"
                    )
            _register_conflict(
                CONFLICT_C,
                f"Conflict (c): suspicious commit {suspicious_commit} is "
                "uncorroborated — " + "; ".join(reasons) + ".",
            )

        if commit_ev is not None:
            if conflict_b or conflict_c:
                conflict_keys.add(_ev_key(commit_ev))
            else:
                support_keys.add(_ev_key(commit_ev))
        commit_corroborated = not conflict_b and not conflict_c
    elif git_findings is not None or by_source[SOURCE_GIT]:
        notes.append(
            "No suspicious commit was supplied; no commit could be correlated "
            "with the incident."
        )

    deployment_time = _as_datetime(_attr(log_findings, "deployment_time", None))
    if deployment_time is not None and log_facts["first_error"] is not None:
        if _precedes(log_facts["first_error"], deployment_time):
            _register_conflict(
                CONFLICT_B,
                f"Conflict (b): the first error at {_iso(log_facts['first_error'])} "
                f"precedes the deployment at {_iso(deployment_time)}; the "
                "deployment cannot have introduced an already-occurring failure.",
            )
            for ev in by_source[SOURCE_LOG]:
                if "first error" in _ev_text(ev):
                    conflict_keys.add(_ev_key(ev))

    support_keys -= conflict_keys
    supporting = [ev for ev in evidence_list if _ev_key(ev) in support_keys]
    conflicting = [ev for ev in evidence_list if _ev_key(ev) in conflict_keys]
    supporting_sources = [
        source for source in _ALL_SOURCES
        if any(_ev_source(ev) == source for ev in supporting)
    ]
    insufficient = len(supporting_sources) < 2
    if insufficient:
        notes.append(
            f"Insufficient corroboration: {len(supporting_sources)} distinct "
            "source(s) support the correlation; at least two distinct "
            "sources are required for a firm hypothesis."
        )

    supporting_texts = [_ev_text(ev) for ev in supporting]
    has_signature = bool(log_facts["signature"]) and any(
        _ev_source(ev) == SOURCE_LOG for ev in supporting
    )
    has_gap = any(
        _ev_observation(ev).casefold().startswith(_GAP_PREFIX) for ev in supporting
    )
    has_unguarded = bool(field_token) and any(
        "no null-safety guard" in blob and _mentions(blob, field_token)
        for blob in supporting_texts
    )

    confidence, confidence_reasons = _confidence(
        supporting_sources, has_signature, has_gap, has_unguarded,
        commit_corroborated, conflict_classes,
    )

    # --- Hypothesis ------------------------------------------------------
    used_llm = False
    if insufficient:
        root_cause = _compose_insufficient(
            supporting_sources, field_token, incident_value, log_facts, endpoint
        )
        if llm_backend is not None:
            notes.append(
                "LLM backend not consulted: fewer than two distinct sources "
                "support the hypothesis."
            )
    else:
        root_cause = _compose_hypothesis(
            field_token=field_token,
            incident_value=incident_value,
            error_type=log_facts["error_type"],
            endpoint=endpoint,
            function=primary_function,
            file_name=primary_file,
            commit_sha=suspicious_commit if commit_corroborated else None,
            supporting=supporting,
            supporting_sources=supporting_sources,
        )
        if llm_backend is not None:
            if not callable(llm_backend):
                notes.append(
                    "LLM backend is not callable; using the deterministic hypothesis."
                )
            else:
                context = {
                    "field": field_token,
                    "incident_value": incident_value,
                    "error_type": log_facts["error_type"],
                    "error_signature": log_facts["signature"],
                    "affected_endpoints": list(log_facts["endpoints"]),
                    "affected_files": list(affected_files),
                    "affected_functions": list(affected_functions),
                    "suspicious_commit": suspicious_commit,
                    "first_error_time": _iso(log_facts["first_error"]),
                    "error_count": log_facts["error_count"],
                    "supporting_sources": list(supporting_sources),
                    "conflicts": list(conflict_classes),
                    "missing_scenarios": list(missing_scenarios),
                    "uncertainty": list(notes),
                    "instruction": (
                        "Return a dict with the optional keys 'root_cause', "
                        "'affected_files', 'affected_functions' and "
                        "'suspicious_commit' (or None to decline). Every claim "
                        "must be traceable to the supplied evidence."
                    ),
                }
                claims, llm_notes = _try_adopt_llm(
                    llm_backend, evidence_list, context, supporting
                )
                notes.extend(llm_notes)
                if claims is not None:
                    used_llm = True
                    root_cause = claims["root_cause"]
                    if claims["affected_files"]:
                        affected_files = claims["affected_files"]
                    if claims["affected_functions"]:
                        affected_functions = claims["affected_functions"]
                    if claims["suspicious_commit"]:
                        suspicious_commit = claims["suspicious_commit"]

    return SynthesisResult(
        root_cause=root_cause,
        affected_files=affected_files,
        affected_functions=affected_functions,
        suspicious_commit=suspicious_commit,
        supporting_evidence=supporting,
        conflicting_evidence=conflicting,
        confidence=confidence,
        confidence_reasons=confidence_reasons,
        uncertainty=notes,
        used_llm=used_llm,
    )
