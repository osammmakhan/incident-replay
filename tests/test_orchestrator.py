"""
Integration and unit tests for the orchestration layer.

Test organisation
-----------------
Unit tests
    Validate orchestrator behaviour in isolation — all agents are mocked or
    exercised with trivially small inputs.  No external filesystem beyond
    the tmp_path fixture.

Integration tests (class ``TestOrchestratorIntegration``)
    Run the full pipeline against the self-contained demo project that lives
    at ``tests/demo_project/``.  These tests execute the real agents, the
    timeline builder, evidence normalization, synthesis, and the regression-
    test generator.  The git agent requires the demo project to be a valid
    git repository (initialized by the test setup script).

Failure-mode tests (class ``TestOrchestratorFailureModes``)
    Verify that individual investigator failures do not crash the pipeline
    and that their failures are surfaced in the report rather than silently
    swallowed.

All tests follow the project's anti-hardcoding rule: no test asserts a
fixed string such as "discount" unless it is testing the demo project's
known incident scenario.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from incident_replay.models.schemas import (
    Evidence,
    IncidentReport,
    RegressionTest,
    SuggestedFix,
    TimelineEvent,
    VerificationResult,
)
from incident_replay.orchestrator import OrchestratorError, replay_incident

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
DEMO_PROJECT = _HERE / "demo_project"
DEMO_LOG = DEMO_PROJECT / "logs" / "production.log"
DEMO_STACK = DEMO_PROJECT / "incident" / "stacktrace.txt"
DEMO_REPO = DEMO_PROJECT
DEMO_AVAILABLE = (
    DEMO_LOG.is_file()
    and (DEMO_PROJECT / ".git").is_dir()
    and (DEMO_PROJECT / "tests").is_dir()
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_report(report: IncidentReport) -> None:
    """Assert basic structural validity of any IncidentReport."""
    assert isinstance(report, IncidentReport)
    assert isinstance(report.incident_summary, str)
    assert len(report.incident_summary) > 0
    assert isinstance(report.timeline, list)
    assert isinstance(report.root_cause, str)
    assert isinstance(report.evidence, list)
    assert isinstance(report.affected_files, list)
    assert isinstance(report.affected_functions, list)
    assert 0.0 <= report.confidence <= 1.0
    assert isinstance(report.regression_test, RegressionTest)
    assert isinstance(report.suggested_fix, SuggestedFix)
    assert isinstance(report.verification_result, VerificationResult)


# ---------------------------------------------------------------------------
# OrchestratorError raised on empty inputs
# ---------------------------------------------------------------------------

class TestOrchestratorError:
    def test_raises_when_both_paths_empty(self):
        with pytest.raises(OrchestratorError):
            replay_incident(
                incident_description="boom",
                repo_path="",
                log_path="",
            )

    def test_raises_when_both_paths_whitespace(self):
        with pytest.raises(OrchestratorError):
            replay_incident(
                incident_description="boom",
                repo_path="   ",
                log_path="   ",
            )


# ---------------------------------------------------------------------------
# Unit: report is always returned, never raises, even on all-fake paths
# ---------------------------------------------------------------------------

class TestOrchestratorAlwaysReturnsReport:
    """Orchestrator must return IncidentReport even when inputs don't exist."""

    def test_nonexistent_log_and_repo_returns_report(self, tmp_path):
        fake_repo = str(tmp_path / "no_such_repo")
        # Create a bare directory so the patcher / test agent don't fail on
        # the directory-existence check.  git agent will fail cleanly.
        os.makedirs(fake_repo, exist_ok=True)
        report = replay_incident(
            incident_description="test",
            repo_path=fake_repo,
            log_path=str(tmp_path / "no.log"),
        )
        _valid_report(report)

    def test_only_log_path_supplied(self, tmp_path):
        log = tmp_path / "app.log"
        log.write_text(
            "2026-01-01 00:00:00,000 ERROR svc Crash\n",
            encoding="utf-8",
        )
        report = replay_incident(
            incident_description="crash",
            repo_path="",
            log_path=str(log),
        )
        _valid_report(report)

    def test_only_repo_path_supplied(self, tmp_path):
        """With no log, agents should degrade gracefully."""
        import subprocess
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo), "config", "user.email", "t@t.com"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "config", "user.name", "T"],
            check=True, capture_output=True,
        )
        (repo / "placeholder.py").write_text("x = 1\n")
        subprocess.run(
            ["git", "-C", str(repo), "add", "."],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", "init"],
            check=True, capture_output=True,
        )
        report = replay_incident(
            incident_description="unknown",
            repo_path=str(repo),
            log_path="",
        )
        _valid_report(report)


# ---------------------------------------------------------------------------
# Unit: report fields conform to the schema
# ---------------------------------------------------------------------------

class TestReportConformsToSchema:
    def test_all_timeline_events_are_timeline_event_instances(self, tmp_path):
        log = tmp_path / "app.log"
        log.write_text("2026-01-01 00:00:00,000 ERROR svc Crash\n", encoding="utf-8")
        report = replay_incident("crash", "", str(log))
        for event in report.timeline:
            assert isinstance(event, TimelineEvent)
            assert event.timestamp, "timestamp must not be empty"
            assert event.description, "description must not be empty"

    def test_all_evidence_items_have_valid_source(self, tmp_path):
        log = tmp_path / "app.log"
        log.write_text("2026-01-01 00:00:00,000 ERROR svc Crash\n", encoding="utf-8")
        report = replay_incident("crash", "", str(log))
        valid_sources = {"log", "git", "code", "test"}
        for ev in report.evidence:
            assert ev.source in valid_sources, f"Unexpected source: {ev.source!r}"

    def test_confidence_within_bounds(self, tmp_path):
        log = tmp_path / "app.log"
        log.write_text("2026-01-01 00:00:00,000 ERROR svc Crash\n", encoding="utf-8")
        report = replay_incident("crash", "", str(log))
        assert 0.0 <= report.confidence <= 1.0

    def test_regression_test_has_name_and_code(self, tmp_path):
        log = tmp_path / "app.log"
        log.write_text("2026-01-01 00:00:00,000 ERROR svc Crash\n", encoding="utf-8")
        report = replay_incident("crash", "", str(log))
        assert report.regression_test.name
        assert report.regression_test.code
        assert report.regression_test.language == "python"

    def test_suggested_fix_has_description(self, tmp_path):
        log = tmp_path / "app.log"
        log.write_text("2026-01-01 00:00:00,000 ERROR svc Crash\n", encoding="utf-8")
        report = replay_incident("crash", "", str(log))
        assert report.suggested_fix.description


# ---------------------------------------------------------------------------
# Unit: failure-mode isolation
# ---------------------------------------------------------------------------

class TestOrchestratorFailureModes:
    """Individual agent failures must NOT crash the pipeline."""

    def test_log_agent_failure_surfaces_in_report(self, tmp_path):
        """Make log_agent.run raise; the report must still be returned."""
        with patch(
            "incident_replay.orchestrator.log_agent.run",
            side_effect=RuntimeError("disk read error"),
        ):
            report = replay_incident(
                incident_description="test",
                repo_path="",
                log_path=str(tmp_path / "some.log"),
            )
        _valid_report(report)
        # The error must be visible in the report
        full_text = report.incident_summary + " ".join(
            e.observation for e in report.evidence
        )
        assert "LogAgent" in full_text or "disk read error" in full_text

    def test_git_agent_failure_surfaces_in_report(self, tmp_path):
        log = tmp_path / "app.log"
        log.write_text("2026-01-01 00:00:00,000 ERROR svc Crash\n", encoding="utf-8")
        fake_repo = str(tmp_path / "r")
        os.makedirs(fake_repo)
        with patch(
            "incident_replay.orchestrator.git_agent.run",
            side_effect=RuntimeError("git broken"),
        ):
            report = replay_incident(
                incident_description="test",
                repo_path=fake_repo,
                log_path=str(log),
            )
        _valid_report(report)
        full_text = report.incident_summary + " ".join(
            e.observation for e in report.evidence
        )
        assert "GitAgent" in full_text or "git broken" in full_text

    def test_synthesis_failure_produces_zero_confidence(self, tmp_path):
        log = tmp_path / "app.log"
        log.write_text("2026-01-01 00:00:00,000 ERROR svc Crash\n", encoding="utf-8")
        with patch(
            "incident_replay.orchestrator.synthesis_agent.run",
            side_effect=RuntimeError("synthesis exploded"),
        ):
            report = replay_incident(
                incident_description="test",
                repo_path="",
                log_path=str(log),
            )
        _valid_report(report)
        assert report.confidence == 0.0

    def test_timeline_failure_does_not_hide_evidence(self, tmp_path):
        log = tmp_path / "app.log"
        log.write_text("2026-01-01 00:00:00,000 ERROR svc Crash\n", encoding="utf-8")
        with patch(
            "incident_replay.orchestrator.timeline_module.build_from_findings",
            side_effect=RuntimeError("timeline broken"),
        ):
            report = replay_incident(
                incident_description="test",
                repo_path="",
                log_path=str(log),
            )
        _valid_report(report)
        # Evidence still present despite timeline failure
        assert isinstance(report.evidence, list)

    def test_all_agents_fail_still_returns_report(self, tmp_path):
        log = tmp_path / "app.log"
        log.write_text("2026-01-01 00:00:00,000 ERROR svc Crash\n", encoding="utf-8")
        fake_repo = str(tmp_path / "r")
        os.makedirs(fake_repo)
        boom = RuntimeError("everything is broken")
        with (
            patch("incident_replay.orchestrator.log_agent.run", side_effect=boom),
            patch("incident_replay.orchestrator.git_agent.run", side_effect=boom),
            patch("incident_replay.orchestrator.code_agent.run", side_effect=boom),
            patch("incident_replay.orchestrator.test_agent.run", side_effect=boom),
            patch("incident_replay.orchestrator.synthesis_agent.run", side_effect=boom),
        ):
            report = replay_incident(
                incident_description="total chaos",
                repo_path=fake_repo,
                log_path=str(log),
            )
        _valid_report(report)
        assert report.confidence == 0.0


# ---------------------------------------------------------------------------
# Integration: demo project
# ---------------------------------------------------------------------------

# The canonical buggy line that the patcher expects to find and replace.
_BUGGY_LINE = (
    "        discount_rate = max(0.0, min(order.discount, self.MAX_DISCOUNT_RATE))\n"
)
_CHECKOUT_PY = DEMO_PROJECT / "app" / "checkout.py"


def _restore_buggy_checkout() -> None:
    """
    Ensure checkout.py contains the unguarded buggy line.

    Tries ``git checkout HEAD -- app/checkout.py`` first (idempotent, fast).
    Falls back to a direct write so the test never silently skips.
    """
    if not _CHECKOUT_PY.exists():
        return
    current = _CHECKOUT_PY.read_text(encoding="utf-8")
    if _BUGGY_LINE in current:
        return  # already buggy — nothing to do

    # Try git first (preserves history-friendly state).
    result = subprocess.run(
        ["git", "checkout", "HEAD", "--", "app/checkout.py"],
        cwd=str(DEMO_PROJECT),
        capture_output=True,
    )
    if result.returncode == 0 and _BUGGY_LINE in _CHECKOUT_PY.read_text(encoding="utf-8"):
        return

    # Fall back: rewrite the file directly so tests are never silently broken.
    _CHECKOUT_PY.write_text(
        '"""\nDemo checkout module with a deliberate null-safety bug.\n\n'
        "The bug: ``calculate_discount`` calls ``min(order.discount, ...)`` without\n"
        "checking whether ``order.discount`` is ``None``.  When a customer omits the\n"
        "discount field, the request fails with::\n\n"
        "    TypeError: '<' not supported between instances of 'NoneType' and 'float'\n\n"
        "This file is intentionally kept minimal so that the incident-replay\n"
        "orchestration tests can exercise real code-agent heuristics without any\n"
        "third-party dependency.\n"
        '"""\n\n\n'
        "class CheckoutService:\n"
        "    MAX_DISCOUNT_RATE = 0.5\n\n"
        "    def calculate_discount(self, order):\n"
        '        """Return the capped discount rate for *order*.\n\n'
        "        BUG: order.discount may be None when the field is omitted from the\n"
        "        request payload; the min() call raises TypeError in that case.\n"
        '        """\n'
        + _BUGGY_LINE
        + "        return discount_rate\n\n"
        "    def total(self, order):\n"
        '        """Return the order total after applying the discount."""\n'
        "        rate = self.calculate_discount(order)\n"
        "        return order.subtotal * (1.0 - rate)\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="module", autouse=False)
def _ensure_buggy_checkout():
    """
    Module-scoped fixture: restore checkout.py to its buggy state before
    the integration tests run, and restore it again afterward so the repo
    is left in a clean state for subsequent test runs.
    """
    if not DEMO_AVAILABLE:
        yield
        return
    _restore_buggy_checkout()
    yield
    _restore_buggy_checkout()


@pytest.fixture(scope="module")
def _integration_report(_ensure_buggy_checkout) -> Optional[IncidentReport]:
    """Run replay_incident once; reused by all TestOrchestratorIntegration tests."""
    if not DEMO_AVAILABLE:
        return None
    stack_trace = DEMO_STACK.read_text(encoding="utf-8")
    return replay_incident(
        incident_description=(
            "POST /checkout returns HTTP 500 when discount field is null."
        ),
        repo_path=str(DEMO_REPO),
        log_path=str(DEMO_LOG),
        stack_trace=stack_trace,
    )


@pytest.mark.skipif(
    not DEMO_AVAILABLE,
    reason="demo_project not available (missing git repo, log, or tests dir)",
)
class TestOrchestratorIntegration:
    """Full pipeline against the self-contained demo project."""

    def test_report_is_valid_incident_report(self, _integration_report):
        report = _integration_report
        _valid_report(report)

    def test_timeline_has_at_least_one_event(self, _integration_report):
        assert len(_integration_report.timeline) >= 1

    def test_timeline_events_are_sorted(self, _integration_report):
        """Dated events should be chronologically ordered."""
        report = _integration_report
        dated = [
            e for e in report.timeline
            if e.timestamp != "(time unknown)"
        ]
        timestamps = [e.timestamp for e in dated]
        assert timestamps == sorted(timestamps), (
            "Timeline events are not in chronological order"
        )

    def test_evidence_has_at_least_one_log_entry(self, _integration_report):
        sources = [e.source for e in _integration_report.evidence]
        assert "log" in sources, "Expected at least one log evidence entry"

    def test_affected_files_contains_checkout(self, _integration_report):
        """The checkout.py file should be identified as affected."""
        files_lower = [f.lower() for f in _integration_report.affected_files]
        assert any("checkout" in f for f in files_lower), (
            f"Expected 'checkout' in affected_files, got: {_integration_report.affected_files}"
        )

    def test_affected_functions_contains_calculate_discount(self, _integration_report):
        funcs_lower = [f.lower() for f in _integration_report.affected_functions]
        assert any("calculate_discount" in f for f in funcs_lower), (
            f"Expected 'calculate_discount' in affected_functions, got: "
            f"{_integration_report.affected_functions}"
        )

    def test_root_cause_mentions_discount(self, _integration_report):
        """Root-cause must reference the incident field."""
        report = _integration_report
        assert "discount" in report.root_cause.lower(), (
            f"Expected 'discount' in root_cause: {report.root_cause!r}"
        )

    def test_confidence_above_zero(self, _integration_report):
        report = _integration_report
        assert report.confidence > 0.0, (
            f"Expected confidence > 0 for a well-evidenced incident, "
            f"got {report.confidence}"
        )

    def test_regression_test_name_is_non_empty(self, _integration_report):
        assert _integration_report.regression_test.name

    def test_regression_test_code_is_syntactically_valid(self, _integration_report):
        import ast
        try:
            ast.parse(_integration_report.regression_test.code)
        except SyntaxError as exc:
            pytest.fail(f"Generated regression test has a syntax error: {exc}")

    def test_suspicious_commit_is_string_or_none(self, _integration_report):
        report = _integration_report
        assert report.suspicious_commit is None or isinstance(
            report.suspicious_commit, str
        )

    def test_verification_result_has_output(self, _integration_report):
        assert isinstance(_integration_report.verification_result.passed, bool)

    def test_report_is_serializable(self, _integration_report):
        """IncidentReport.model_dump() must not raise (UI contract)."""
        data = _integration_report.model_dump()
        assert isinstance(data, dict)
        assert "incident_summary" in data
        assert "timeline" in data
        assert "confidence" in data


# ---------------------------------------------------------------------------
# Integration: demo project cleanup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def _cleanup_generated_tests():
    """Remove any regression test files written during integration tests."""
    yield
    reg_dir = DEMO_PROJECT / "tests" / "regression"
    if reg_dir.exists():
        shutil.rmtree(reg_dir, ignore_errors=True)
    # Always leave checkout.py in its buggy (committed) state so the next
    # test run starts from a known-good baseline.
    if DEMO_AVAILABLE:
        _restore_buggy_checkout()
