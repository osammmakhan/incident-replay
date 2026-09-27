"""
Unit tests for incident_replay.utils — git_utils, file_utils, log_utils.

git_utils tests use the real incident-replay-demo repo at a well-known
sibling path.  They are skipped gracefully if the repo is not present so
that CI environments without the demo repo still pass the rest of the suite.

file_utils and log_utils tests are fully self-contained (tmp files / inline
strings — no network, no external repo required).
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_REPO = Path("f:/GenAI/incident-replay-demo")
DEMO_AVAILABLE = DEMO_REPO.is_dir() and (DEMO_REPO / ".git").is_dir()

# ---------------------------------------------------------------------------
# Shared sample log lines (mirrors the real production.log format)
# ---------------------------------------------------------------------------

SAMPLE_LOG_LINES = [
    "2026-09-26 22:00:01,042 INFO  shopco.deploy  Starting deployment version=2.4.1 commit=cd5456a",
    "2026-09-26 22:00:15,334 INFO  shopco.api     checkout request_id=req-f3a1d2b7 order_id=ORD-8801",
    "2026-09-26 22:00:15,391 INFO  shopco.api     checkout OK request_id=req-f3a1d2b7 total=89.97",
    "2026-09-26 22:01:05,003 INFO  shopco.api     checkout request_id=req-c6e8d402 order_id=ORD-8804 discount=null",
    "2026-09-26 22:01:05,047 ERROR shopco.api     checkout FAILED request_id=req-c6e8d402 error=TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'",
    "2026-09-26 22:01:05,049 ERROR shopco.api     POST /checkout 500 44ms request_id=req-c6e8d402",
    "2026-09-26 22:01:18,231 INFO  shopco.api     checkout request_id=req-d71ba903 order_id=ORD-8805",
    "2026-09-26 22:01:44,560 INFO  shopco.api     checkout request_id=req-e02cf814 order_id=ORD-8806 discount=null",
    "2026-09-26 22:01:44,601 ERROR shopco.api     checkout FAILED request_id=req-e02cf814 error=TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'",
    "2026-09-26 22:01:44,603 ERROR shopco.api     POST /checkout 500 41ms request_id=req-e02cf814",
    "2026-09-26 22:03:55,001 WARN  shopco.ops     5xx spike detected endpoint=POST /checkout error_rate=0.56",
]

TRACEBACK_LINES = [
    "Traceback (most recent call last):",
    '  File "/app/app/checkout.py", line 45, in calculate_discount',
    "    discount_rate = max(0.0, min(order.discount, self.MAX_DISCOUNT_RATE))",
    "TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'",
    "Some unrelated line after the traceback",
]

SAMPLE_PYTHON_SOURCE = textwrap.dedent("""\
    class CheckoutService:
        MAX_DISCOUNT_RATE = 1.0

        def calculate_discount(self, order):
            discount_rate = max(0.0, min(order.discount, self.MAX_DISCOUNT_RATE))
            return order.subtotal * discount_rate

        def process_checkout(self, order):
            subtotal = order.subtotal
            discount_amount = self.calculate_discount(order)
            total = subtotal - discount_amount
            return {"total": total, "status": "confirmed"}


    def standalone_helper():
        return 42
""")


# ===========================================================================
# log_utils tests
# ===========================================================================

class TestLogUtilsParseLines:
    def test_info_line_parsed(self):
        from incident_replay.utils.log_utils import parse_log_lines
        result = parse_log_lines(SAMPLE_LOG_LINES)
        info_entries = [e for e in result.entries if e.level == "INFO"]
        assert len(info_entries) >= 5

    def test_error_lines_classified(self):
        from incident_replay.utils.log_utils import parse_log_lines
        result = parse_log_lines(SAMPLE_LOG_LINES)
        assert len(result.errors) == 4  # 2 FAILED + 2 HTTP 500 lines

    def test_first_error_time_is_earliest(self):
        from incident_replay.utils.log_utils import parse_log_lines
        result = parse_log_lines(SAMPLE_LOG_LINES)
        assert result.first_error_time is not None
        assert result.first_error_time == datetime(2026, 9, 26, 22, 1, 5, 47000, tzinfo=timezone.utc)

    def test_http_events_extracted(self):
        from incident_replay.utils.log_utils import parse_log_lines
        result = parse_log_lines(SAMPLE_LOG_LINES)
        http_500s = [h for h in result.http_events if h.status == 500]
        assert len(http_500s) == 2
        assert all(h.path == "/checkout" for h in http_500s)
        assert all(h.method == "POST" for h in http_500s)

    def test_error_signature_extracted(self):
        from incident_replay.utils.log_utils import parse_log_lines
        result = parse_log_lines(SAMPLE_LOG_LINES)
        assert result.error_signature is not None
        assert "TypeError" in result.error_signature

    def test_failing_request_ids(self):
        from incident_replay.utils.log_utils import parse_log_lines
        result = parse_log_lines(SAMPLE_LOG_LINES)
        assert "req-c6e8d402" in result.failing_request_ids
        assert "req-e02cf814" in result.failing_request_ids

    def test_kv_parsed_from_message(self):
        from incident_replay.utils.log_utils import parse_log_lines
        result = parse_log_lines(SAMPLE_LOG_LINES)
        deploy_entry = result.entries[0]
        assert deploy_entry.kv.get("version") == "2.4.1"
        assert deploy_entry.kv.get("commit") == "cd5456a"

    def test_empty_input_returns_empty_parsed_log(self):
        from incident_replay.utils.log_utils import parse_log_lines
        result = parse_log_lines([])
        assert result.entries == []
        assert result.errors == []
        assert result.http_events == []
        assert result.first_error_time is None
        assert result.error_signature is None
        assert result.failing_request_ids == []

    def test_unrecognised_lines_stored_without_raising(self):
        from incident_replay.utils.log_utils import parse_log_lines
        lines = ["some plain text", "another unstructured line"]
        result = parse_log_lines(lines)
        assert len(result.entries) == 2
        assert result.entries[0].level == ""


class TestLogUtilsHelpers:
    def test_extract_error_lines_filters_correctly(self):
        from incident_replay.utils.log_utils import extract_error_lines
        errors = extract_error_lines(SAMPLE_LOG_LINES)
        assert all("ERROR" in e for e in errors)
        assert len(errors) == 4

    def test_extract_stack_trace_returns_block(self):
        from incident_replay.utils.log_utils import extract_stack_trace
        tb = extract_stack_trace(TRACEBACK_LINES)
        assert tb is not None
        assert "Traceback" in tb
        assert "TypeError" in tb
        assert "Some unrelated" not in tb

    def test_extract_stack_trace_none_when_absent(self):
        from incident_replay.utils.log_utils import extract_stack_trace
        assert extract_stack_trace(SAMPLE_LOG_LINES) is None

    def test_extract_request_ids_unique_ordered(self):
        from incident_replay.utils.log_utils import extract_request_ids
        ids = extract_request_ids(SAMPLE_LOG_LINES)
        assert ids[0] == "req-f3a1d2b7"
        assert len(ids) == len(set(ids))  # all unique

    def test_get_timestamps_returns_datetimes(self):
        from incident_replay.utils.log_utils import get_timestamps
        tss = get_timestamps(SAMPLE_LOG_LINES)
        assert all(isinstance(t, datetime) for t in tss)
        assert len(tss) == len(SAMPLE_LOG_LINES)

    def test_first_error_timestamp(self):
        from incident_replay.utils.log_utils import first_error_timestamp
        ts = first_error_timestamp(SAMPLE_LOG_LINES)
        assert ts is not None
        assert ts.hour == 22
        assert ts.minute == 1

    def test_first_error_timestamp_none_for_clean_log(self):
        from incident_replay.utils.log_utils import first_error_timestamp
        clean = [l for l in SAMPLE_LOG_LINES if "ERROR" not in l]
        assert first_error_timestamp(clean) is None


# ===========================================================================
# file_utils tests
# ===========================================================================

class TestFileUtilsReadFile:
    def test_read_existing_file(self, tmp_path):
        from incident_replay.utils.file_utils import read_file
        f = tmp_path / "hello.txt"
        f.write_text("hello world\n", encoding="utf-8")
        assert read_file(str(f)) == "hello world\n"

    def test_read_missing_file_raises(self, tmp_path):
        from incident_replay.utils.file_utils import read_file, FileReadError
        with pytest.raises(FileReadError):
            read_file(str(tmp_path / "missing.txt"))

    def test_read_directory_raises(self, tmp_path):
        from incident_replay.utils.file_utils import read_file, FileReadError
        with pytest.raises(FileReadError):
            read_file(str(tmp_path))

    def test_safe_read_file_returns_default(self, tmp_path):
        from incident_replay.utils.file_utils import safe_read_file
        result = safe_read_file(str(tmp_path / "ghost.py"), default="fallback")
        assert result == "fallback"

    def test_normalises_crlf(self, tmp_path):
        from incident_replay.utils.file_utils import read_file
        f = tmp_path / "crlf.txt"
        f.write_bytes(b"line1\r\nline2\r\n")
        assert read_file(str(f)) == "line1\nline2\n"

    def test_file_exists_true_and_false(self, tmp_path):
        from incident_replay.utils.file_utils import file_exists
        f = tmp_path / "exists.py"
        f.write_text("x = 1")
        assert file_exists(str(f)) is True
        assert file_exists(str(tmp_path / "nope.py")) is False


class TestFileUtilsReadSection:
    def test_returns_requested_lines(self, tmp_path):
        from incident_replay.utils.file_utils import read_section
        f = tmp_path / "code.py"
        f.write_text("line1\nline2\nline3\nline4\nline5\n")
        section = read_section(str(f), 2, 4)
        assert section.start_line == 2
        assert section.end_line == 4
        assert section.content == "line2\nline3\nline4"

    def test_clamps_end_line_beyond_file(self, tmp_path):
        from incident_replay.utils.file_utils import read_section
        f = tmp_path / "short.py"
        f.write_text("a\nb\nc\n")
        section = read_section(str(f), 1, 100)
        assert section.end_line == 3

    def test_clamps_start_line_below_one(self, tmp_path):
        from incident_replay.utils.file_utils import read_section
        f = tmp_path / "start.py"
        f.write_text("a\nb\nc\n")
        section = read_section(str(f), 0, 2)
        assert section.start_line == 1


class TestFileUtilsExtractFunction:
    def _write_source(self, tmp_path: Path) -> str:
        f = tmp_path / "checkout.py"
        f.write_text(SAMPLE_PYTHON_SOURCE, encoding="utf-8")
        return str(f)

    def test_finds_method_in_class(self, tmp_path):
        from incident_replay.utils.file_utils import extract_function
        path = self._write_source(tmp_path)
        info = extract_function(path, "calculate_discount")
        assert info is not None
        assert info.name == "calculate_discount"
        assert info.qualified_name == "CheckoutService.calculate_discount"
        assert "discount_rate" in info.source

    def test_finds_top_level_function(self, tmp_path):
        from incident_replay.utils.file_utils import extract_function
        path = self._write_source(tmp_path)
        info = extract_function(path, "standalone_helper")
        assert info is not None
        assert info.qualified_name == "standalone_helper"
        assert "return 42" in info.source

    def test_returns_none_for_missing_function(self, tmp_path):
        from incident_replay.utils.file_utils import extract_function
        path = self._write_source(tmp_path)
        assert extract_function(path, "nonexistent_function") is None

    def test_start_and_end_lines_are_set(self, tmp_path):
        from incident_replay.utils.file_utils import extract_function
        path = self._write_source(tmp_path)
        info = extract_function(path, "process_checkout")
        assert info is not None
        assert info.start_line < info.end_line

    def test_invalid_python_raises_value_error(self, tmp_path):
        from incident_replay.utils.file_utils import extract_function
        f = tmp_path / "bad.py"
        f.write_text("def broken(\n")
        with pytest.raises(ValueError):
            extract_function(str(f), "broken")


class TestFindFunctionsInFile:
    def _write_source(self, tmp_path: Path) -> str:
        f = tmp_path / "checkout.py"
        f.write_text(SAMPLE_PYTHON_SOURCE, encoding="utf-8")
        return str(f)

    def test_finds_all_functions(self, tmp_path):
        from incident_replay.utils.file_utils import find_functions_in_file
        path = self._write_source(tmp_path)
        funcs = find_functions_in_file(path)
        names = [f.name for f in funcs]
        assert "calculate_discount" in names
        assert "process_checkout" in names
        assert "standalone_helper" in names

    def test_qualified_names_include_class_prefix(self, tmp_path):
        from incident_replay.utils.file_utils import find_functions_in_file
        path = self._write_source(tmp_path)
        funcs = find_functions_in_file(path)
        qualified = {f.qualified_name for f in funcs}
        assert "CheckoutService.calculate_discount" in qualified
        assert "CheckoutService.process_checkout" in qualified


# ===========================================================================
# git_utils tests (require demo repo)
# ===========================================================================

@pytest.mark.skipif(not DEMO_AVAILABLE, reason="incident-replay-demo repo not present")
class TestGitUtilsWithDemoRepo:
    REPO = str(DEMO_REPO)

    def test_get_recent_commits_returns_four(self):
        from incident_replay.utils.git_utils import get_recent_commits
        commits = get_recent_commits(self.REPO, n=10)
        assert len(commits) == 4

    def test_commit_fields_populated(self):
        from incident_replay.utils.git_utils import get_recent_commits
        commits = get_recent_commits(self.REPO)
        c = commits[0]
        assert len(c.sha) == 40
        assert len(c.short_sha) <= 10
        assert c.author != ""
        assert isinstance(c.date, datetime)
        assert c.date.tzinfo is not None
        assert c.message != ""

    def test_newest_commit_is_regression(self):
        from incident_replay.utils.git_utils import get_recent_commits
        commits = get_recent_commits(self.REPO)
        assert "simplify" in commits[0].message.lower() or "refactor" in commits[0].message.lower()

    def test_get_commit_files_returns_list(self):
        from incident_replay.utils.git_utils import get_recent_commits, get_commit_files
        commits = get_recent_commits(self.REPO)
        files = get_commit_files(self.REPO, commits[0].sha)
        assert isinstance(files, list)
        assert len(files) >= 1

    def test_get_commit_metadata_includes_changed_files(self):
        from incident_replay.utils.git_utils import get_recent_commits, get_commit_metadata
        commits = get_recent_commits(self.REPO)
        meta = get_commit_metadata(self.REPO, commits[0].sha)
        assert len(meta.changed_files) >= 1

    def test_get_diff_returns_string_with_diff_markers(self):
        from incident_replay.utils.git_utils import get_recent_commits, get_diff
        commits = get_recent_commits(self.REPO)
        diff = get_diff(self.REPO, commits[0].sha)
        assert "@@" in diff or "diff --git" in diff

    def test_get_file_at_commit_returns_content(self):
        from incident_replay.utils.git_utils import get_recent_commits, get_commit_files, get_file_at_commit
        commits = get_recent_commits(self.REPO)
        files = get_commit_files(self.REPO, commits[0].sha)
        content = get_file_at_commit(self.REPO, commits[0].sha, files[0])
        assert len(content) > 0

    def test_get_log_oneline_format(self):
        from incident_replay.utils.git_utils import get_log_oneline
        lines = get_log_oneline(self.REPO)
        assert len(lines) == 4
        assert all(len(l.split(" ", 1)) == 2 for l in lines)

    def test_commits_near_time_finds_regression_commit(self):
        from incident_replay.utils.git_utils import commits_near_time, get_recent_commits
        # The incident happened at 2026-09-26 22:01:05 UTC.
        # The deployment (and likely the regression commit) was right before that.
        incident_dt = datetime(2026, 9, 26, 22, 1, 5, tzinfo=timezone.utc)
        near = commits_near_time(self.REPO, incident_dt, window_seconds=86400 * 365 * 10)
        assert len(near) >= 1

    def test_commits_near_time_empty_for_future_window(self):
        from incident_replay.utils.git_utils import commits_near_time
        # A window entirely in the future relative to any commit
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        near = commits_near_time(self.REPO, future, window_seconds=60)
        assert near == []


class TestGitUtilsErrors:
    def test_git_error_on_bad_repo_path(self):
        from incident_replay.utils.git_utils import get_recent_commits, GitError
        with pytest.raises(GitError):
            get_recent_commits("/nonexistent/path/repo")

    def test_git_error_on_bad_sha(self):
        from incident_replay.utils.git_utils import get_commit_files, GitError
        if not DEMO_AVAILABLE:
            pytest.skip("demo repo not present")
        with pytest.raises(GitError):
            get_commit_files(str(DEMO_REPO), "deadbeef00000000000000000000000000000000")
