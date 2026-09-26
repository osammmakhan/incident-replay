"""
Tests for incident_replay.agents.git_agent.

All tests that exercise real repository logic require the demo project at
f:/GenAI/incident-replay-demo and are skipped when it is absent.

Unit-level tests (heuristic logic, data shape, edge cases) use a temporary
git repository created in-process so they run in any environment.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from incident_replay.agents.git_agent import (
    CommitFinding,
    GitAgentError,
    GitFindings,
    _NULL_SAFETY_RE,
    _RISKY_MESSAGE_WORDS,
    _basename,
    run,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_REPO = Path("f:/GenAI/incident-replay-demo")
DEMO_AVAILABLE = DEMO_REPO.is_dir() and (DEMO_REPO / ".git").is_dir()

# The incident's first error timestamp from the production log
INCIDENT_TIME = datetime(2026, 9, 26, 22, 1, 5, tzinfo=timezone.utc)

# Files named in the demo stack trace
STACK_TRACE_FILES = ["app/checkout.py", "app/api.py"]

# Error keywords drawn from the demo error signature
ERROR_KEYWORDS = ["discount", "nonetype", "float"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_local(*args, cwd: str) -> None:
    """Run a git command inside *cwd*, raising on failure."""
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, text=True)


def _make_repo(tmp_path: Path) -> str:
    """Create a minimal two-commit git repo in *tmp_path* and return its path."""
    repo = str(tmp_path / "repo")
    os.makedirs(repo)
    _git_local("init", cwd=repo)
    _git_local("config", "user.email", "test@example.com", cwd=repo)
    _git_local("config", "user.name", "Test", cwd=repo)

    # Commit 1 — initial file with null guard
    code_safe = textwrap.dedent("""\
        class Service:
            def calculate(self, value):
                rate = value or 0.0
                return rate * 100
    """)
    Path(repo, "service.py").write_text(code_safe)
    _git_local("add", "service.py", cwd=repo)
    _git_local("commit", "-m", "feat: initial service", cwd=repo)

    # Commit 2 — refactor removes null guard
    code_broken = textwrap.dedent("""\
        class Service:
            def calculate(self, value):
                return value * 100
    """)
    Path(repo, "service.py").write_text(code_broken)
    _git_local("add", "service.py", cwd=repo)
    _git_local("commit", "-m", "refactor: simplify calculation", cwd=repo)

    return repo


# ===========================================================================
# Regex unit tests (no repo needed)
# ===========================================================================

class TestNullSafetyRegex:
    def _matches(self, line: str) -> bool:
        return bool(_NULL_SAFETY_RE.search(line))

    def test_or_zero_float(self):
        assert self._matches("-        rate = value or 0.0")

    def test_or_zero_int(self):
        assert self._matches("-        rate = value or 0")

    def test_is_none_check(self):
        assert self._matches("-        if value is None:")

    def test_is_not_none_check(self):
        assert self._matches("-        if value is not None:")

    def test_inequality_none(self):
        assert self._matches("-        assert value != None")

    def test_optional_type_hint(self):
        assert self._matches("-    def foo(self, x: Optional[float]):")

    def test_added_line_not_matched(self):
        # Lines starting with + are additions, not removals — must not match
        assert not self._matches("+        rate = value or 0.0")

    def test_unchanged_line_not_matched(self):
        assert not self._matches("         rate = value or 0.0")


class TestRiskyMessageRegex:
    def _matches(self, msg: str) -> bool:
        return bool(_RISKY_MESSAGE_WORDS.search(msg))

    def test_refactor(self):
        assert self._matches("refactor: simplify discount handling")

    def test_simplify(self):
        assert self._matches("simplify the calculation logic")

    def test_remove(self):
        assert self._matches("remove intermediate variable")

    def test_clean(self):
        assert self._matches("clean up discount code")

    def test_feature_commit_not_matched(self):
        assert not self._matches("feat: add user authentication")

    def test_test_commit_not_matched(self):
        assert not self._matches("test: add unit tests for checkout")

    def test_case_insensitive(self):
        assert self._matches("Refactor: Simplify Calculation")


class TestBasename:
    def test_unix_path(self):
        assert _basename("app/checkout.py") == "checkout.py"

    def test_windows_path(self):
        assert _basename("app\\checkout.py") == "checkout.py"

    def test_bare_filename(self):
        assert _basename("checkout.py") == "checkout.py"


# ===========================================================================
# Edge-case / error handling (uses temp repo)
# ===========================================================================

class TestGitAgentErrors:
    def test_empty_repo_path_raises(self):
        with pytest.raises(GitAgentError):
            run("")

    def test_whitespace_repo_path_raises(self):
        with pytest.raises(GitAgentError):
            run("   ")

    def test_nonexistent_repo_raises(self):
        with pytest.raises(GitAgentError):
            run("/nonexistent/path/to/repo")


class TestGitAgentWithTempRepo:
    def test_returns_git_findings(self, tmp_path):
        repo = _make_repo(tmp_path)
        f = run(repo)
        assert isinstance(f, GitFindings)

    def test_inspects_all_commits(self, tmp_path):
        repo = _make_repo(tmp_path)
        f = run(repo)
        assert f.commits_inspected == 2

    def test_all_commits_populated(self, tmp_path):
        repo = _make_repo(tmp_path)
        f = run(repo)
        assert len(f.all_commits) == 2

    def test_refactor_commit_flagged(self, tmp_path):
        repo = _make_repo(tmp_path)
        f = run(repo, error_keywords=["value"])
        assert len(f.suspicious_commits) >= 1
        top = f.suspicious_commits[0]
        assert "refactor" in top.message.lower()

    def test_null_guard_removal_detected(self, tmp_path):
        repo = _make_repo(tmp_path)
        f = run(repo, error_keywords=["value"])
        top = f.suspicious_commits[0]
        null_reasons = [r for r in top.relevance_reasons if "null-safety" in r]
        assert len(null_reasons) >= 1

    def test_top_suspect_is_highest_scoring(self, tmp_path):
        repo = _make_repo(tmp_path)
        f = run(repo, error_keywords=["value"])
        assert f.top_suspect is not None
        assert f.top_suspect.relevance_score == max(
            c.relevance_score for c in f.suspicious_commits
        )

    def test_time_window_scores_recent_commit(self, tmp_path):
        repo = _make_repo(tmp_path)
        # Use a very large window so all commits qualify
        now = datetime.now(tz=timezone.utc) + timedelta(days=1)
        f = run(repo, incident_time=now, time_window_hours=24 * 365 * 10)
        time_reasons = [
            r for c in f.suspicious_commits
            for r in c.relevance_reasons
            if "before the incident" in r
        ]
        assert len(time_reasons) >= 1

    def test_commit_finding_fields_populated(self, tmp_path):
        repo = _make_repo(tmp_path)
        f = run(repo, error_keywords=["value"])
        top = f.top_suspect
        assert top is not None
        assert len(top.sha) == 40
        assert top.short_sha != ""
        assert top.author != ""
        assert isinstance(top.date, datetime)
        assert top.message != ""
        assert len(top.changed_files) >= 1
        assert isinstance(top.relevance_reasons, list)
        assert top.relevance_score > 0

    def test_relevant_diff_is_string(self, tmp_path):
        repo = _make_repo(tmp_path)
        f = run(repo, error_keywords=["value"])
        top = f.top_suspect
        assert isinstance(top.relevant_diff, str)

    def test_no_false_causal_claims_in_reasons(self, tmp_path):
        repo = _make_repo(tmp_path)
        f = run(repo, error_keywords=["value"])
        causal = ["root cause", "caused by", "because", "therefore", "bug is"]
        for c in f.suspicious_commits:
            for reason in c.relevance_reasons:
                for term in causal:
                    assert term not in reason.lower(), (
                        f"Causal claim found in reason: {reason!r}"
                    )

    def test_hints_are_labelled(self, tmp_path):
        repo = _make_repo(tmp_path)
        f = run(repo, error_keywords=["value"])
        for hint in f.interpretation_hints:
            assert hint.startswith("HINT:")

    def test_affected_files_heuristic(self, tmp_path):
        repo = _make_repo(tmp_path)
        f = run(repo, affected_files=["service.py"])
        file_reasons = [
            r for c in f.suspicious_commits
            for r in c.relevance_reasons
            if "stack trace" in r
        ]
        assert len(file_reasons) >= 1

    def test_no_suspicious_commits_when_no_signals(self, tmp_path):
        # Create a repo with only a safe, non-risky commit
        repo = str(tmp_path / "safe_repo")
        os.makedirs(repo)
        _git_local("init", cwd=repo)
        _git_local("config", "user.email", "t@e.com", cwd=repo)
        _git_local("config", "user.name", "T", cwd=repo)
        Path(repo, "readme.md").write_text("docs only")
        _git_local("add", "readme.md", cwd=repo)
        _git_local("commit", "-m", "docs: add readme", cwd=repo)

        f = run(repo)  # no keywords, no incident time, no affected files
        assert len(f.suspicious_commits) == 0
        assert f.top_suspect is None

    def test_empty_repo_warning(self, tmp_path):
        # init but no commits
        repo = str(tmp_path / "empty_repo")
        os.makedirs(repo)
        _git_local("init", cwd=repo)
        # git log on an empty repo exits non-zero — GitAgentError is expected
        with pytest.raises(GitAgentError):
            run(repo)


# ===========================================================================
# Integration tests against the real demo repository
# ===========================================================================

@pytest.mark.skipif(not DEMO_AVAILABLE, reason="incident-replay-demo repo not present")
class TestGitAgentWithDemoRepo:
    REPO = str(DEMO_REPO)

    def _run(self, **kwargs) -> GitFindings:
        return run(
            self.REPO,
            incident_time=INCIDENT_TIME,
            error_keywords=ERROR_KEYWORDS,
            affected_files=STACK_TRACE_FILES,
            time_window_hours=24,
            **kwargs,
        )

    def test_runs_without_error(self):
        f = self._run()
        assert isinstance(f, GitFindings)

    def test_inspects_four_commits(self):
        f = self._run()
        assert f.commits_inspected == 4

    def test_regression_commit_is_top_suspect(self):
        """The refactor commit must be ranked #1 without hardcoding its SHA."""
        f = self._run()
        assert f.top_suspect is not None
        # Identify it by message content, not by SHA
        assert "simplif" in f.top_suspect.message.lower() or \
               "refactor" in f.top_suspect.message.lower()

    def test_regression_commit_sha_is_full_length(self):
        f = self._run()
        assert len(f.top_suspect.sha) == 40

    def test_regression_commit_changes_checkout(self):
        f = self._run()
        changed = [_basename(p) for p in f.top_suspect.changed_files]
        assert "checkout.py" in changed

    def test_null_safety_removal_detected(self):
        f = self._run()
        null_reasons = [
            r for r in f.top_suspect.relevance_reasons
            if "null-safety" in r
        ]
        assert len(null_reasons) >= 1

    def test_time_proximity_reason_present(self):
        f = self._run()
        time_reasons = [
            r for r in f.top_suspect.relevance_reasons
            if "before the incident" in r
        ]
        assert len(time_reasons) >= 1

    def test_error_keywords_reason_present(self):
        f = self._run()
        kw_reasons = [
            r for r in f.top_suspect.relevance_reasons
            if "keyword" in r.lower()
        ]
        assert len(kw_reasons) >= 1

    def test_file_match_reason_present(self):
        f = self._run()
        file_reasons = [
            r for r in f.top_suspect.relevance_reasons
            if "stack trace" in r
        ]
        assert len(file_reasons) >= 1

    def test_top_suspect_outscores_others(self):
        f = self._run()
        if len(f.suspicious_commits) > 1:
            assert f.suspicious_commits[0].relevance_score >= \
                   f.suspicious_commits[1].relevance_score

    def test_relevant_diff_contains_discount(self):
        f = self._run()
        assert "discount" in f.top_suspect.relevant_diff.lower()

    def test_no_causal_claims_in_reasons(self):
        f = self._run()
        causal = ["root cause", "caused by", "because", "therefore", "bug is"]
        for c in f.suspicious_commits:
            for reason in c.relevance_reasons:
                for term in causal:
                    assert term not in reason.lower()

    def test_hints_labelled(self):
        f = self._run()
        for hint in f.interpretation_hints:
            assert hint.startswith("HINT:")

    def test_without_keywords_still_finds_suspect(self):
        """Even with no error keywords, time + file + message heuristics suffice."""
        f = run(
            self.REPO,
            incident_time=INCIDENT_TIME,
            affected_files=STACK_TRACE_FILES,
            time_window_hours=24,
        )
        assert f.top_suspect is not None

    def test_parse_warnings_empty_on_clean_repo(self):
        f = self._run()
        assert f.parse_warnings == []
