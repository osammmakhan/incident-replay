"""
Git Investigator agent.

Responsibility: inspect the repository's commit history relative to a
reported incident and return structured, evidence-quality findings about
what changed — without declaring a root cause.

Design rules
------------
OBSERVATION vs INTERPRETATION
  Every :class:`CommitFinding` field describes what git *actually recorded*.
  The agent does not blame a commit for the incident; it surfaces relevance
  signals and leaves causal judgement to the synthesis stage.

No hardcoded answers
  Suspicious commits are identified through configurable heuristics that
  work on any repository:
    1. Commits whose timestamp falls within a time-window before the incident.
    2. Commits that touch files mentioned in the error stack trace.
    3. Commits whose diff removes null-safety patterns around field names
       named in the error signature (e.g. ``or 0``, ``or 0.0``, ``is None``).
    4. Commits whose message contains keywords associated with risk
       (refactor, simplify, remove, clean, fix, update, change).

No LLM required
  All analysis is deterministic string/regex matching on git output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from incident_replay.utils.git_utils import (
    CommitInfo,
    GitError,
    commits_near_time,
    get_commit_files,
    get_commit_metadata,
    get_diff,
    get_recent_commits,
)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class CommitFinding:
    """Observed facts about a single commit that may be relevant to the incident."""

    sha: str
    short_sha: str
    author: str
    date: datetime
    message: str
    changed_files: list[str]

    # The relevant portion of the diff (empty string if not fetched)
    relevant_diff: str

    # Ordered list of human-readable reasons this commit was flagged.
    # Each reason is a factual statement about the commit content, not a
    # causal claim.  E.g.:
    #   "Commit is within 24h before the incident timestamp."
    #   "Diff removes null-safety guard ('or 0.0') on field 'discount'."
    relevance_reasons: list[str]

    # Numeric score: higher = more relevance signals fired.
    # NOT a probability or confidence; purely a ranking aid.
    relevance_score: int


@dataclass
class GitFindings:
    """
    Structured observations from repository history investigation.

    All fields describe what git *contains* — not what caused the incident.
    """

    repo_path: str
    commits_inspected: int
    parse_warnings: list[str]

    # All commits examined (newest-first)
    all_commits: list[CommitInfo]

    # Subset that triggered at least one relevance heuristic, ranked by score
    suspicious_commits: list[CommitFinding]

    # The single highest-scoring CommitFinding, or None if none were flagged
    top_suspect: Optional[CommitFinding]

    # Interpretation hints — labelled, never root-cause claims
    interpretation_hints: list[str]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class GitAgentError(ValueError):
    """Raised when the repository is inaccessible or not a git repo."""


def run(
    repo_path: str,
    incident_time: Optional[datetime] = None,
    error_keywords: Optional[list[str]] = None,
    affected_files: Optional[list[str]] = None,
    n_commits: int = 30,
    time_window_hours: int = 24,
) -> GitFindings:
    """Investigate the git history of *repo_path* and return :class:`GitFindings`.

    Parameters
    ----------
    repo_path:
        Absolute or relative path to the target git repository root.
    incident_time:
        UTC-aware datetime of the incident's first observed error.
        When provided, commits in the *time_window_hours* before this
        time are scored as time-adjacent candidates.
    error_keywords:
        Words extracted from the error signature (e.g. ``["discount",
        "NoneType"]``).  Used to search diffs for related field names.
    affected_files:
        File paths extracted from the stack trace (relative to repo root).
        Commits that touch these files are scored higher.
    n_commits:
        How many recent commits to inspect (default 30).
    time_window_hours:
        How many hours before *incident_time* to treat as the suspicious
        window (default 24).

    Returns
    -------
    GitFindings

    Raises
    ------
    GitAgentError
        When *repo_path* is not a valid git repository.
    """
    if not repo_path or not repo_path.strip():
        raise GitAgentError("repo_path must not be empty.")

    try:
        commits = get_recent_commits(repo_path, n=n_commits)
    except GitError as exc:
        raise GitAgentError(f"Cannot read git history from {repo_path!r}: {exc}") from exc

    if not commits:
        return _empty_findings(repo_path, warning="Repository has no commits.")

    error_keywords = [k.lower() for k in (error_keywords or [])]
    affected_files = [f.lower() for f in (affected_files or [])]

    findings: list[CommitFinding] = []
    warnings: list[str] = []

    for commit in commits:
        try:
            cf = _evaluate_commit(
                repo_path=repo_path,
                commit=commit,
                incident_time=incident_time,
                time_window_hours=time_window_hours,
                error_keywords=error_keywords,
                affected_files=affected_files,
            )
        except GitError as exc:
            warnings.append(f"Could not inspect commit {commit.short_sha}: {exc}")
            continue

        if cf.relevance_score > 0:
            findings.append(cf)

    # Sort by score descending, then by date descending (most recent first)
    findings.sort(key=lambda f: (f.relevance_score, f.date.timestamp()), reverse=True)

    top = findings[0] if findings else None
    hints = _derive_hints(findings, incident_time)

    return GitFindings(
        repo_path=repo_path,
        commits_inspected=len(commits),
        parse_warnings=warnings,
        all_commits=commits,
        suspicious_commits=findings,
        top_suspect=top,
        interpretation_hints=hints,
    )


# ---------------------------------------------------------------------------
# Per-commit evaluation
# ---------------------------------------------------------------------------

# Keywords in commit messages that correlate with risky changes.
# These are stems, so we only anchor the *start* of the word with \b.
_RISKY_MESSAGE_WORDS = re.compile(
    r"\b(refactor|simplif|remov|clean|rewrite|restructur|reorgani[sz]|"
    r"optimis|optimiz|consolidat|eliminat|replac|strip|inline)",
    re.IGNORECASE,
)

# Null-safety patterns whose removal is meaningful.
# Non-capturing group — re.findall returns plain strings, not tuples.
_NULL_SAFETY_RE = re.compile(
    r"^\-.*(?:"
    r"\bor\s+0(?:\.0+)?\b"        # or 0 / or 0.0
    r"|\bor\s+['\"][\s]*['\"]"    # or ""
    r"|\bor\s+\[\]"               # or []
    r"|\bor\s+\{\}"               # or {}
    r"|\bis\s+None\b"             # is None
    r"|\bis\s+not\s+None\b"       # is not None
    r"|!=\s*None\b"               # != None
    r"|==\s*None\b"               # == None
    r"|\bOptional\["              # Optional[T] type hint
    r"|\bif\s+\w+\s*:"            # if x: (truthiness guard)
    r")",
    re.MULTILINE,
)


def _evaluate_commit(
    repo_path: str,
    commit: CommitInfo,
    incident_time: Optional[datetime],
    time_window_hours: int,
    error_keywords: list[str],
    affected_files: list[str],
) -> CommitFinding:
    """Score a single commit and return a :class:`CommitFinding`."""
    reasons: list[str] = []
    score = 0

    # Fetch changed files (cheap)
    changed_files = get_commit_files(repo_path, commit.sha)
    changed_lower = [f.lower() for f in changed_files]

    # ------------------------------------------------------------------ #
    # Heuristic 1 — time proximity                                        #
    # ------------------------------------------------------------------ #
    if incident_time is not None:
        window_start = incident_time - timedelta(hours=time_window_hours)
        if window_start <= commit.date <= incident_time:
            delta_h = (incident_time - commit.date).total_seconds() / 3600
            reasons.append(
                f"Commit timestamp ({commit.date.strftime('%Y-%m-%dT%H:%M:%SZ')}) "
                f"is {delta_h:.1f}h before the incident."
            )
            score += 2

    # ------------------------------------------------------------------ #
    # Heuristic 2 — touches files named in the stack trace               #
    # ------------------------------------------------------------------ #
    matched_files: list[str] = []
    for af in affected_files:
        for cf in changed_lower:
            # Match on filename tail, not full path (repo layout may differ)
            if af.endswith(cf) or cf.endswith(af) or _basename(af) == _basename(cf):
                matched_files.append(cf)
                break
    if matched_files:
        reasons.append(
            f"Commit modifies file(s) referenced in the stack trace: "
            f"{', '.join(matched_files)}."
        )
        score += 3

    # ------------------------------------------------------------------ #
    # Heuristic 3 — risky message keywords                               #
    # ------------------------------------------------------------------ #
    if _RISKY_MESSAGE_WORDS.search(commit.message):
        reasons.append(
            f"Commit message contains a keyword associated with structural "
            f"change: '{commit.message}'."
        )
        score += 1

    # ------------------------------------------------------------------ #
    # Heuristics 4 & 5 require the diff — fetch only if already scoring  #
    # or if we have error keywords to match                               #
    # ------------------------------------------------------------------ #
    diff_text = ""
    if score > 0 or error_keywords:
        diff_text = get_diff(repo_path, commit.sha)

    # ------------------------------------------------------------------ #
    # Heuristic 4 — error keywords appear in the diff                    #
    # ------------------------------------------------------------------ #
    if diff_text and error_keywords:
        matched_kw: list[str] = []
        diff_lower = diff_text.lower()
        for kw in error_keywords:
            if kw in diff_lower:
                matched_kw.append(kw)
        if matched_kw:
            reasons.append(
                f"Diff contains keyword(s) from the error signature: "
                f"{', '.join(matched_kw)}."
            )
            score += 2

    # ------------------------------------------------------------------ #
    # Heuristic 5 — null-safety guards removed in diff                   #
    # ------------------------------------------------------------------ #
    if diff_text:
        removed_guards = _NULL_SAFETY_RE.findall(diff_text)
        if removed_guards:
            # Each match is the full removed line (no capturing groups → plain str)
            unique_guards = list(dict.fromkeys(
                m.lstrip("-").strip() for m in removed_guards
            ))
            reasons.append(
                f"Diff removes {len(removed_guards)} null-safety guard(s): "
                f"{', '.join(repr(g) for g in unique_guards[:3])}."
            )
            score += 3

    # Trim diff to changed-file sections only to keep findings concise
    relevant_diff = _trim_diff(diff_text, changed_files) if diff_text else ""

    return CommitFinding(
        sha=commit.sha,
        short_sha=commit.short_sha,
        author=commit.author,
        date=commit.date,
        message=commit.message,
        changed_files=changed_files,
        relevant_diff=relevant_diff,
        relevance_reasons=reasons,
        relevance_score=score,
    )


# ---------------------------------------------------------------------------
# Interpretation hints
# ---------------------------------------------------------------------------

def _derive_hints(
    findings: list[CommitFinding],
    incident_time: Optional[datetime],
) -> list[str]:
    hints: list[str] = []

    if not findings:
        hints.append(
            "HINT: No commits were flagged by the relevance heuristics. "
            "Consider widening time_window_hours or providing error_keywords."
        )
        return hints

    top = findings[0]
    hints.append(
        f"HINT: The highest-scoring commit is {top.short_sha} "
        f"('{top.message}'). Investigate its diff for changes related "
        f"to the error signature."
    )

    null_findings = [
        f for f in findings
        if any("null-safety" in r for r in f.relevance_reasons)
    ]
    if null_findings:
        hints.append(
            f"HINT: {len(null_findings)} commit(s) removed null-safety "
            f"guards. Verify whether the removed guard protected against "
            f"the input value observed in the incident."
        )

    return hints


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _trim_diff(diff_text: str, changed_files: list[str]) -> str:
    """Return the diff sections that relate to source files only.

    Strips boilerplate git header lines; keeps all ``diff --git`` hunks.
    If the diff is short (<= 120 lines) returns it in full.
    """
    lines = diff_text.splitlines()
    if len(lines) <= 120:
        return diff_text

    # Keep only hunks for files that look like source (not logs, lock files)
    keep_exts = {".py", ".js", ".ts", ".rb", ".go", ".java", ".cs", ".cpp",
                 ".c", ".h", ".rs", ".php"}
    result: list[str] = []
    in_hunk = False
    for line in lines:
        if line.startswith("diff --git"):
            in_hunk = any(line.lower().endswith(ext) for ext in keep_exts)
        if in_hunk:
            result.append(line)

    return "\n".join(result) if result else diff_text[:4000]


def _basename(path: str) -> str:
    """Return the filename part of a path (cross-platform)."""
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def _empty_findings(repo_path: str, warning: str = "") -> GitFindings:
    return GitFindings(
        repo_path=repo_path,
        commits_inspected=0,
        parse_warnings=[warning] if warning else [],
        all_commits=[],
        suspicious_commits=[],
        top_suspect=None,
        interpretation_hints=[
            "HINT: No commits available to inspect."
        ],
    )
