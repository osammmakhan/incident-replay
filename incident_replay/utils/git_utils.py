"""
Git inspection utilities for incident-replay agents.

All functions shell out to ``git`` via subprocess.  No external git library
is required — just a working ``git`` binary on PATH and a valid repo at
``repo_path``.

Return values are plain dicts / lists so callers are not coupled to any
specific git library type.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _git(repo_path: str, *args: str) -> str:
    """Run a git command inside *repo_path* and return stdout as a string.

    Raises ``GitError`` on non-zero exit, if *repo_path* does not exist, or
    if the OS rejects the path (e.g. Windows ``NotADirectoryError``).
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise GitError(f"Cannot run git in {repo_path!r}: {exc}") from exc
    if result.returncode != 0:
        raise GitError(
            f"git {' '.join(args)!r} failed in {repo_path!r}:\n{result.stderr.strip()}"
        )
    return result.stdout


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class GitError(RuntimeError):
    """Raised when a git command fails or the repo is inaccessible."""


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class CommitInfo:
    """Metadata for a single git commit."""

    sha: str
    short_sha: str
    author: str
    date: datetime          # timezone-aware UTC
    message: str
    changed_files: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_recent_commits(repo_path: str, n: int = 20) -> list[CommitInfo]:
    """Return the *n* most recent commits, newest first.

    Each entry includes SHA, author, ISO date, and first line of the message.
    Changed-file lists are **not** populated here for performance; use
    :func:`get_commit_files` when you need them.
    """
    # %x1f = ASCII unit-separator as field delimiter (safe in any commit message)
    fmt = "%H%x1f%h%x1f%an%x1f%aI%x1f%s"
    raw = _git(repo_path, "log", f"-{n}", f"--format={fmt}")
    commits: list[CommitInfo] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        sha, short_sha, author, date_str, message = line.split("\x1f", 4)
        commits.append(CommitInfo(
            sha=sha,
            short_sha=short_sha,
            author=author,
            date=_parse_iso(date_str),
            message=message,
        ))
    return commits


def get_commit_metadata(repo_path: str, sha: str) -> CommitInfo:
    """Return full metadata for a single commit *sha*.

    Includes the list of changed files.
    """
    fmt = "%H%x1f%h%x1f%an%x1f%aI%x1f%s"
    raw = _git(repo_path, "show", "--no-patch", f"--format={fmt}", sha)
    line = raw.splitlines()[0].strip()
    full_sha, short_sha, author, date_str, message = line.split("\x1f", 4)
    changed = get_commit_files(repo_path, sha)
    return CommitInfo(
        sha=full_sha,
        short_sha=short_sha,
        author=author,
        date=_parse_iso(date_str),
        message=message,
        changed_files=changed,
    )


def get_commit_files(repo_path: str, sha: str) -> list[str]:
    """Return the list of files changed in *sha* (relative to repo root)."""
    raw = _git(repo_path, "diff-tree", "--no-commit-id", "-r", "--name-only", sha)
    return [p for p in raw.splitlines() if p.strip()]


def get_diff(repo_path: str, sha: str) -> str:
    """Return the full unified diff introduced by *sha*."""
    return _git(repo_path, "show", "--unified=3", sha)


def get_file_at_commit(repo_path: str, sha: str, file_path: str) -> str:
    """Return the contents of *file_path* as it existed in commit *sha*.

    Raises ``GitError`` if the file did not exist in that commit.
    """
    return _git(repo_path, "show", f"{sha}:{file_path}")


def commits_near_time(
    repo_path: str,
    incident_time: datetime,
    window_seconds: int = 3600,
    n: int = 50,
) -> list[CommitInfo]:
    """Return commits whose date falls within *window_seconds* **before**
    *incident_time* (up to *n* candidates searched).

    The window is intentionally one-sided: a commit *after* the incident
    cannot have caused it.
    """
    candidates = get_recent_commits(repo_path, n)
    cutoff_early = incident_time.timestamp() - window_seconds
    cutoff_late = incident_time.timestamp()
    return [
        c for c in candidates
        if cutoff_early <= c.date.timestamp() <= cutoff_late
    ]


def get_log_oneline(repo_path: str, n: int = 20) -> list[str]:
    """Return ``git log --oneline`` lines for the *n* most recent commits."""
    raw = _git(repo_path, "log", f"-{n}", "--oneline")
    return [l for l in raw.splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _parse_iso(date_str: str) -> datetime:
    """Parse an ISO-8601 date string (as produced by ``%aI``) to a UTC datetime."""
    # Python 3.7+ fromisoformat doesn't handle the trailing timezone offset in
    # all variants; replace the trailing +00:00 / -HH:MM for robustness.
    try:
        dt = datetime.fromisoformat(date_str)
    except ValueError:
        # Fallback: strip timezone suffix and treat as UTC
        dt = datetime.fromisoformat(date_str[:19]).replace(tzinfo=timezone.utc)
    # Normalise to UTC
    return dt.astimezone(timezone.utc)
