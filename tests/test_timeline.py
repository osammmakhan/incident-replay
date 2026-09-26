"""
Tests for incident_replay.analysis.timeline.

All unit tests are fully self-contained (no external repo or demo log file).
Integration tests use the demo project at f:/GenAI/incident-replay-demo and
are skipped when it is absent.

The test suite validates:
  - TimelineInput field defaults
  - _fmt / _format_duration helpers
  - build() with each event category in isolation
  - build() with combined inputs (deployment → first_error → recurrence → alert)
  - Ordering: events always oldest-first, unknown-timestamp events at front
  - No invented timestamps: unknown times produce "(time unknown)", never guesses
  - extract_deployment_from_log_lines / extract_alert_from_log_lines helpers
  - build_from_findings convenience wrapper
  - Integration: demo log + git findings produce correct timeline structure
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from incident_replay.analysis.timeline import (
    EVENT_ALERT,
    EVENT_COMMIT,
    EVENT_DEPLOYMENT,
    EVENT_DETECTED,
    EVENT_ERROR,
    EVENT_FIRST_ERROR,
    TimelineInput,
    _fmt,
    _format_duration,
    _sort_and_format,
    build,
    build_from_findings,
    extract_alert_from_log_lines,
    extract_deployment_from_log_lines,
    extract_from_git_findings,
    extract_from_log_findings,
)
from incident_replay.models.schemas import TimelineEvent

# ---------------------------------------------------------------------------
# Constants / fixtures
# ---------------------------------------------------------------------------

DEMO_LOG    = Path("f:/GenAI/incident-replay-demo/logs/production.log")
DEMO_REPO   = Path("f:/GenAI/incident-replay-demo")
DEMO_AVAILABLE = DEMO_LOG.is_file() and (DEMO_REPO / ".git").is_dir()

UTC = timezone.utc

# Reference timestamps taken directly from the demo production.log
T_DEPLOY    = datetime(2026, 9, 26, 22, 0, 1, 42000,  tzinfo=UTC)
T_FIRST_ERR = datetime(2026, 9, 26, 22, 1, 5, 47000,  tzinfo=UTC)
T_ALERT     = datetime(2026, 9, 26, 22, 3, 55, 1000,  tzinfo=UTC)

# A fake commit timestamp (UTC) for unit tests
T_COMMIT    = datetime(2026, 9, 26, 18, 0, 0, tzinfo=UTC)


# Minimal log lines matching the demo format (self-contained)
MINIMAL_LOG_LINES = [
    "2026-09-26 22:00:01,042 INFO  shopco.deploy  Starting deployment version=2.4.1 commit=cd5456a branch=main",
    "2026-09-26 22:00:03,215 INFO  shopco.deploy  Deployment complete version=2.4.1 commit=cd5456a",
    "2026-09-26 22:00:15,334 INFO  shopco.api     checkout request_id=req-aaa order_id=ORD-1 customer_id=CUST-1",
    "2026-09-26 22:00:15,391 INFO  shopco.api     checkout OK request_id=req-aaa total=89.97",
    "2026-09-26 22:01:05,003 INFO  shopco.api     checkout request_id=req-ccc order_id=ORD-2 customer_id=CUST-2 discount=null",
    "2026-09-26 22:01:05,047 ERROR shopco.api     checkout FAILED request_id=req-ccc error=TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'",
    "2026-09-26 22:01:05,049 ERROR shopco.api     POST /checkout 500 44ms request_id=req-ccc",
    "2026-09-26 22:03:55,001 WARN  shopco.ops     5xx spike detected endpoint=POST /checkout error_rate=0.56 window=5m threshold=0.05",
    "2026-09-26 22:04:00,000 INFO  shopco.ops     PagerDuty alert fired severity=high service=shopco-checkout",
]


# ===========================================================================
# Helper unit tests
# ===========================================================================

class TestFmt:
    def test_utc_datetime(self):
        dt = datetime(2026, 9, 26, 22, 1, 5, tzinfo=UTC)
        assert _fmt(dt) == "2026-09-26T22:01:05Z"

    def test_non_utc_normalised(self):
        from datetime import timezone as tz
        import datetime as dt_mod
        # +05:30 offset → 16:31:05 UTC
        tz_ist = dt_mod.timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2026, 9, 26, 22, 1, 5, tzinfo=tz_ist)
        result = _fmt(dt)
        assert result == "2026-09-26T16:31:05Z"

    def test_microseconds_stripped(self):
        dt = datetime(2026, 9, 26, 22, 1, 5, 123456, tzinfo=UTC)
        assert _fmt(dt) == "2026-09-26T22:01:05Z"


class TestFormatDuration:
    def test_under_60_seconds(self):
        assert _format_duration(45) == "45s"

    def test_exactly_60_seconds(self):
        assert _format_duration(60) == "1m"

    def test_minutes_and_seconds(self):
        assert _format_duration(145) == "2m 25s"

    def test_whole_minutes(self):
        assert _format_duration(120) == "2m"

    def test_hours(self):
        assert _format_duration(3600) == "1h"

    def test_hours_and_minutes(self):
        assert _format_duration(3660) == "1h 1m"

    def test_zero(self):
        assert _format_duration(0) == "0s"


class TestSortAndFormat:
    def test_none_timestamp_sorts_first(self):
        raw = [
            (datetime(2026, 9, 26, 22, 0, 0, tzinfo=UTC), "first_error", "Error"),
            (None, "deployment", "Deploy"),
        ]
        events = _sort_and_format(raw)
        assert events[0].event == "deployment"
        assert events[0].timestamp == "(time unknown)"

    def test_chronological_order(self):
        t1 = datetime(2026, 9, 26, 22, 0, 0, tzinfo=UTC)
        t2 = datetime(2026, 9, 26, 22, 1, 0, tzinfo=UTC)
        t3 = datetime(2026, 9, 26, 22, 2, 0, tzinfo=UTC)
        raw = [(t3, "alert", "A"), (t1, "deployment", "D"), (t2, "first_error", "E")]
        events = _sort_and_format(raw)
        assert [e.event for e in events] == ["deployment", "first_error", "alert"]

    def test_returns_timeline_events(self):
        raw = [(datetime(2026, 1, 1, tzinfo=UTC), "commit", "A commit")]
        events = _sort_and_format(raw)
        assert all(isinstance(e, TimelineEvent) for e in events)

    def test_empty_input(self):
        assert _sort_and_format([]) == []


# ===========================================================================
# TimelineInput defaults
# ===========================================================================

class TestTimelineInputDefaults:
    def test_all_none_by_default(self):
        inp = TimelineInput()
        assert inp.first_error_time is None
        assert inp.deployment_time is None
        assert inp.suspicious_commits == []
        assert inp.recurrence_count == 0

    def test_affected_endpoints_defaults_to_empty_list(self):
        inp = TimelineInput()
        assert inp.affected_endpoints == []

    def test_provided_values_stored(self):
        inp = TimelineInput(
            first_error_time=T_FIRST_ERR,
            deployment_version="2.4.1",
        )
        assert inp.first_error_time == T_FIRST_ERR
        assert inp.deployment_version == "2.4.1"


# ===========================================================================
# build() — individual event categories
# ===========================================================================

class TestBuildDeploymentEvent:
    def test_deployment_with_timestamp(self):
        inp = TimelineInput(
            deployment_time=T_DEPLOY,
            deployment_version="2.4.1",
            deployment_commit="cd5456a",
        )
        events = build(inp)
        dep = next(e for e in events if e.event == EVENT_DEPLOYMENT)
        assert dep.timestamp == _fmt(T_DEPLOY)
        assert "2.4.1" in dep.description
        assert "cd5456a" in dep.description

    def test_deployment_without_timestamp_uses_unknown(self):
        inp = TimelineInput(deployment_version="2.4.1", deployment_commit="abc1234")
        events = build(inp)
        dep = next(e for e in events if e.event == EVENT_DEPLOYMENT)
        assert dep.timestamp == "(time unknown)"
        assert "2.4.1" in dep.description
        assert "not recorded" in dep.description

    def test_no_deployment_info_no_event(self):
        inp = TimelineInput(first_error_time=T_FIRST_ERR)
        events = build(inp)
        assert not any(e.event == EVENT_DEPLOYMENT for e in events)

    def test_deployment_host_included(self):
        inp = TimelineInput(
            deployment_time=T_DEPLOY,
            deployment_host="prod-web-01",
        )
        events = build(inp)
        dep = next(e for e in events if e.event == EVENT_DEPLOYMENT)
        assert "prod-web-01" in dep.description


class TestBuildCommitEvent:
    def test_commit_event_present(self):
        inp = TimelineInput(
            suspicious_commits=[(T_COMMIT, "abc1234", "refactor: simplify", ["checkout.py"])]
        )
        events = build(inp)
        commit_ev = next(e for e in events if e.event == EVENT_COMMIT)
        assert "abc1234" in commit_ev.description
        assert "refactor: simplify" in commit_ev.description
        assert "checkout.py" in commit_ev.description

    def test_commit_timestamp_formatted(self):
        inp = TimelineInput(
            suspicious_commits=[(T_COMMIT, "abc1234", "msg", [])]
        )
        events = build(inp)
        commit_ev = next(e for e in events if e.event == EVENT_COMMIT)
        assert commit_ev.timestamp == _fmt(T_COMMIT)

    def test_multiple_changed_files(self):
        inp = TimelineInput(
            suspicious_commits=[(T_COMMIT, "abc1234", "msg", ["a.py", "b.py", "c.py", "d.py"])]
        )
        events = build(inp)
        commit_ev = next(e for e in events if e.event == EVENT_COMMIT)
        assert "+1 more" in commit_ev.description

    def test_no_commits_no_commit_event(self):
        inp = TimelineInput(first_error_time=T_FIRST_ERR)
        events = build(inp)
        assert not any(e.event == EVENT_COMMIT for e in events)


class TestBuildFirstErrorEvent:
    def test_first_error_event_present(self):
        inp = TimelineInput(
            first_error_time=T_FIRST_ERR,
            error_signature="TypeError: unsupported operand",
            affected_endpoints=["POST /checkout"],
        )
        events = build(inp)
        fe = next(e for e in events if e.event == EVENT_FIRST_ERROR)
        assert fe.timestamp == _fmt(T_FIRST_ERR)
        assert "TypeError" in fe.description
        assert "POST /checkout" in fe.description

    def test_no_first_error_time_no_event(self):
        inp = TimelineInput(deployment_time=T_DEPLOY)
        events = build(inp)
        assert not any(e.event == EVENT_FIRST_ERROR for e in events)

    def test_missing_signature_uses_unknown(self):
        inp = TimelineInput(first_error_time=T_FIRST_ERR)
        events = build(inp)
        fe = next(e for e in events if e.event == EVENT_FIRST_ERROR)
        assert "unknown error" in fe.description


class TestBuildErrorRecurrenceEvent:
    def test_recurrence_event_when_count_gt_1(self):
        inp = TimelineInput(
            first_error_time=T_FIRST_ERR,
            recurrence_count=5,
            recurrence_window_seconds=145.0,
        )
        events = build(inp)
        rec = next(e for e in events if e.event == EVENT_ERROR)
        assert "5 times" in rec.description
        assert "2m 25s" in rec.description

    def test_no_recurrence_event_when_count_is_1(self):
        inp = TimelineInput(
            first_error_time=T_FIRST_ERR,
            recurrence_count=1,
            recurrence_window_seconds=0.0,
        )
        events = build(inp)
        assert not any(e.event == EVENT_ERROR for e in events)

    def test_recurrence_without_window_uses_unknown(self):
        inp = TimelineInput(
            first_error_time=T_FIRST_ERR,
            recurrence_count=3,
            recurrence_window_seconds=None,
        )
        events = build(inp)
        rec = next(e for e in events if e.event == EVENT_ERROR)
        assert rec.timestamp == "(time unknown)"

    def test_recurrence_last_occurrence_timestamp(self):
        window = 145.0
        inp = TimelineInput(
            first_error_time=T_FIRST_ERR,
            recurrence_count=5,
            recurrence_window_seconds=window,
        )
        expected_last = T_FIRST_ERR + timedelta(seconds=window)
        events = build(inp)
        rec = next(e for e in events if e.event == EVENT_ERROR)
        assert rec.timestamp == _fmt(expected_last)


class TestBuildAlertEvent:
    def test_alert_with_timestamp_and_description(self):
        inp = TimelineInput(
            alert_time=T_ALERT,
            alert_description="5xx spike detected error_rate=0.56",
        )
        events = build(inp)
        al = next(e for e in events if e.event == EVENT_ALERT)
        assert al.timestamp == _fmt(T_ALERT)
        assert "0.56" in al.description

    def test_alert_description_only(self):
        inp = TimelineInput(alert_description="PagerDuty alert fired")
        events = build(inp)
        al = next(e for e in events if e.event == EVENT_ALERT)
        assert al.timestamp == "(time unknown)"
        assert "PagerDuty" in al.description

    def test_no_alert_no_event(self):
        inp = TimelineInput(first_error_time=T_FIRST_ERR)
        events = build(inp)
        assert not any(e.event == EVENT_ALERT for e in events)


class TestBuildDetectedEvent:
    def test_detected_with_timestamp(self):
        detected_time = datetime(2026, 9, 26, 22, 4, 0, tzinfo=UTC)
        inp = TimelineInput(
            detected_time=detected_time,
            detected_description="Incident INC-2026-0926 filed.",
        )
        events = build(inp)
        det = next(e for e in events if e.event == EVENT_DETECTED)
        assert det.timestamp == _fmt(detected_time)
        assert "INC-2026-0926" in det.description

    def test_detected_no_time_uses_unknown(self):
        inp = TimelineInput(detected_description="Incident filed.")
        events = build(inp)
        det = next(e for e in events if e.event == EVENT_DETECTED)
        assert det.timestamp == "(time unknown)"

    def test_no_detected_no_event(self):
        inp = TimelineInput(first_error_time=T_FIRST_ERR)
        events = build(inp)
        assert not any(e.event == EVENT_DETECTED for e in events)


# ===========================================================================
# build() — ordering and combined scenarios
# ===========================================================================

class TestBuildOrdering:
    def test_deployment_before_first_error(self):
        inp = TimelineInput(
            deployment_time=T_DEPLOY,
            first_error_time=T_FIRST_ERR,
        )
        events = build(inp)
        categories = [e.event for e in events]
        assert categories.index(EVENT_DEPLOYMENT) < categories.index(EVENT_FIRST_ERROR)

    def test_commit_before_first_error_when_earlier(self):
        inp = TimelineInput(
            suspicious_commits=[(T_COMMIT, "abc", "msg", [])],
            first_error_time=T_FIRST_ERR,
        )
        events = build(inp)
        categories = [e.event for e in events]
        assert categories.index(EVENT_COMMIT) < categories.index(EVENT_FIRST_ERROR)

    def test_unknown_timestamp_before_all_dated_events(self):
        inp = TimelineInput(
            deployment_version="1.0",  # no time
            first_error_time=T_FIRST_ERR,
        )
        events = build(inp)
        assert events[0].event == EVENT_DEPLOYMENT
        assert events[0].timestamp == "(time unknown)"

    def test_alert_after_first_error(self):
        inp = TimelineInput(
            first_error_time=T_FIRST_ERR,
            alert_time=T_ALERT,
            alert_description="spike detected",
        )
        events = build(inp)
        cats = [e.event for e in events]
        assert cats.index(EVENT_FIRST_ERROR) < cats.index(EVENT_ALERT)

    def test_full_chain_ordered(self):
        """deployment → commit → first_error → recurrence → alert → detected"""
        detected_time = datetime(2026, 9, 26, 22, 4, 0, tzinfo=UTC)
        inp = TimelineInput(
            deployment_time=T_DEPLOY,
            deployment_version="2.4.1",
            suspicious_commits=[(T_COMMIT, "abc", "refactor", ["checkout.py"])],
            first_error_time=T_FIRST_ERR,
            recurrence_count=5,
            recurrence_window_seconds=145.0,
            alert_time=T_ALERT,
            alert_description="5xx spike",
            detected_time=detected_time,
            detected_description="Incident filed",
        )
        events = build(inp)
        cats = [e.event for e in events]

        # Commit (T_COMMIT at 18:00) before deploy (22:00)
        assert cats.index(EVENT_COMMIT) < cats.index(EVENT_DEPLOYMENT)
        assert cats.index(EVENT_DEPLOYMENT) < cats.index(EVENT_FIRST_ERROR)
        assert cats.index(EVENT_FIRST_ERROR) < cats.index(EVENT_ERROR)
        assert cats.index(EVENT_ERROR) < cats.index(EVENT_ALERT)
        assert cats.index(EVENT_ALERT) < cats.index(EVENT_DETECTED)

    def test_empty_input_returns_empty_list(self):
        inp = TimelineInput()
        assert build(inp) == []

    def test_all_events_are_timeline_event_instances(self):
        inp = TimelineInput(
            deployment_time=T_DEPLOY,
            first_error_time=T_FIRST_ERR,
            alert_time=T_ALERT,
            alert_description="spike",
        )
        events = build(inp)
        for e in events:
            assert isinstance(e, TimelineEvent)

    def test_all_timestamps_non_empty(self):
        inp = TimelineInput(
            deployment_version="1.0",   # unknown time
            first_error_time=T_FIRST_ERR,
        )
        events = build(inp)
        for e in events:
            assert e.timestamp != ""

    def test_all_descriptions_non_empty(self):
        inp = TimelineInput(
            deployment_time=T_DEPLOY,
            first_error_time=T_FIRST_ERR,
        )
        events = build(inp)
        for e in events:
            assert e.description.strip() != ""

    def test_all_event_fields_non_empty(self):
        inp = TimelineInput(
            deployment_time=T_DEPLOY,
            first_error_time=T_FIRST_ERR,
            alert_time=T_ALERT,
            alert_description="spike",
        )
        events = build(inp)
        for e in events:
            assert e.event != ""


# ===========================================================================
# Deployment and alert extraction from raw log lines
# ===========================================================================

class TestExtractDeploymentFromLogLines:
    def test_extracts_version_and_commit(self):
        t, version, commit, host = extract_deployment_from_log_lines(MINIMAL_LOG_LINES)
        assert version == "2.4.1"
        assert commit == "cd5456a"

    def test_extracts_deploy_time(self):
        t, _, _, _ = extract_deployment_from_log_lines(MINIMAL_LOG_LINES)
        assert t is not None
        assert t.hour == 22
        assert t.minute == 0

    def test_returns_none_for_missing_deployment(self):
        lines = [
            "2026-09-26 22:00:15,334 INFO  shopco.api  checkout OK request_id=req-aaa",
        ]
        t, v, c, h = extract_deployment_from_log_lines(lines)
        assert t is None
        assert v is None
        assert c is None

    def test_empty_lines(self):
        t, v, c, h = extract_deployment_from_log_lines([])
        assert t is None and v is None and c is None


class TestExtractAlertFromLogLines:
    def test_extracts_spike_alert(self):
        t, desc = extract_alert_from_log_lines(MINIMAL_LOG_LINES)
        assert t is not None
        assert "spike" in desc.lower() or "alert" in desc.lower()

    def test_alert_timestamp_correct(self):
        t, _ = extract_alert_from_log_lines(MINIMAL_LOG_LINES)
        assert t.hour == 22
        assert t.minute == 3

    def test_returns_none_for_no_alert(self):
        lines = [
            "2026-09-26 22:00:15,334 INFO  shopco.api  checkout OK request_id=req-aaa",
        ]
        t, desc = extract_alert_from_log_lines(lines)
        assert t is None
        assert desc is None


# ===========================================================================
# extract_from_log_findings / extract_from_git_findings
# ===========================================================================

class TestExtractFromFindings:
    def _make_log_findings(self, **overrides):
        """Build a minimal mock LogFindings-like object."""
        class _LF:
            log_start_time = T_DEPLOY
            log_end_time = datetime(2026, 9, 26, 22, 4, 0, tzinfo=UTC)
            first_error_time = T_FIRST_ERR
            dominant_error_signature = "TypeError: unsupported operand"
            affected_endpoints = ["POST /checkout"]
            recurrence_count = 5
            recurrence_window_seconds = 145.0

        lf = _LF()
        for k, v in overrides.items():
            setattr(lf, k, v)
        return lf

    def _make_git_findings(self):
        from datetime import datetime

        class _CF:
            date = T_COMMIT
            short_sha = "abc1234"
            message = "refactor: simplify"
            changed_files = ["app/checkout.py"]

        class _GF:
            suspicious_commits = [_CF()]

        return _GF()

    def test_extract_from_log_findings_keys(self):
        result = extract_from_log_findings(self._make_log_findings())
        assert "first_error_time" in result
        assert "recurrence_count" in result
        assert "affected_endpoints" in result

    def test_extract_from_log_findings_values(self):
        result = extract_from_log_findings(self._make_log_findings())
        assert result["first_error_time"] == T_FIRST_ERR
        assert result["recurrence_count"] == 5

    def test_extract_from_git_findings_commits(self):
        result = extract_from_git_findings(self._make_git_findings())
        assert "suspicious_commits" in result
        assert len(result["suspicious_commits"]) == 1
        dt, sha, msg, files = result["suspicious_commits"][0]
        assert sha == "abc1234"
        assert any("checkout.py" in f for f in files)

    def test_build_from_log_findings(self):
        lf = self._make_log_findings()
        events = build_from_findings(log_findings=lf)
        cats = [e.event for e in events]
        assert EVENT_FIRST_ERROR in cats
        assert EVENT_ERROR in cats  # recurrence_count=5

    def test_build_from_both_findings(self):
        lf = self._make_log_findings()
        gf = self._make_git_findings()
        events = build_from_findings(log_findings=lf, git_findings=gf)
        cats = [e.event for e in events]
        assert EVENT_COMMIT in cats
        assert EVENT_FIRST_ERROR in cats


class TestBuildFromFindingsWithLogLines:
    def test_deployment_extracted_from_log_lines(self):
        class _LF:
            log_start_time = T_DEPLOY
            log_end_time = datetime(2026, 9, 26, 22, 4, 0, tzinfo=UTC)
            first_error_time = T_FIRST_ERR
            dominant_error_signature = "TypeError"
            affected_endpoints = ["POST /checkout"]
            recurrence_count = 1
            recurrence_window_seconds = None

        events = build_from_findings(
            log_findings=_LF(),
            log_lines=MINIMAL_LOG_LINES,
        )
        cats = [e.event for e in events]
        assert EVENT_DEPLOYMENT in cats

    def test_alert_extracted_from_log_lines(self):
        class _LF:
            log_start_time = T_DEPLOY
            log_end_time = datetime(2026, 9, 26, 22, 4, 0, tzinfo=UTC)
            first_error_time = T_FIRST_ERR
            dominant_error_signature = "TypeError"
            affected_endpoints = ["POST /checkout"]
            recurrence_count = 1
            recurrence_window_seconds = None

        events = build_from_findings(
            log_findings=_LF(),
            log_lines=MINIMAL_LOG_LINES,
        )
        cats = [e.event for e in events]
        assert EVENT_ALERT in cats

    def test_deployment_before_first_error(self):
        class _LF:
            log_start_time = T_DEPLOY
            log_end_time = datetime(2026, 9, 26, 22, 4, 0, tzinfo=UTC)
            first_error_time = T_FIRST_ERR
            dominant_error_signature = "TypeError"
            affected_endpoints = []
            recurrence_count = 1
            recurrence_window_seconds = None

        events = build_from_findings(
            log_findings=_LF(),
            log_lines=MINIMAL_LOG_LINES,
        )
        cats = [e.event for e in events]
        assert cats.index(EVENT_DEPLOYMENT) < cats.index(EVENT_FIRST_ERROR)


# ===========================================================================
# Schema validation
# ===========================================================================

class TestTimelineEventSchema:
    def test_event_field_persisted(self):
        e = TimelineEvent(timestamp="2026-09-26T22:00:00Z", event="deployment", description="D")
        assert e.event == "deployment"

    def test_event_field_defaults_to_empty(self):
        e = TimelineEvent(timestamp="2026-09-26T22:00:00Z", description="D")
        assert e.event == ""

    def test_model_dump_includes_event(self):
        e = TimelineEvent(timestamp="2026-09-26T22:00:00Z", event="commit", description="D")
        d = e.model_dump()
        assert "event" in d
        assert d["event"] == "commit"

    def test_roundtrip_via_model_validate(self):
        data = {"timestamp": "2026-09-26T22:00:00Z", "event": "first_error", "description": "E"}
        e = TimelineEvent.model_validate(data)
        assert e.event == "first_error"


# ===========================================================================
# Integration — demo incident
# ===========================================================================

@pytest.mark.skipif(not DEMO_AVAILABLE, reason="incident-replay-demo not present")
class TestTimelineWithDemoIncident:
    """
    End-to-end integration: run log_agent + git_agent, feed their findings
    into the timeline builder, and verify the output matches the known
    demo incident chronology without hardcoding any SHA or timestamp.
    """

    def _build(self):
        from incident_replay.agents.log_agent import run as log_run
        from incident_replay.agents.git_agent import run as git_run
        from datetime import datetime, timezone

        log_path = str(DEMO_LOG)
        stack_path = DEMO_REPO / "incident" / "stacktrace.txt"
        stack = stack_path.read_text(encoding="utf-8") if stack_path.is_file() else None
        raw_lines = DEMO_LOG.read_text(encoding="utf-8").splitlines()

        lf = log_run(log_path, stack_trace=stack)
        gf = git_run(
            str(DEMO_REPO),
            incident_time=lf.first_error_time,
            error_keywords=["discount", "nonetype"],
            affected_files=["app/checkout.py", "app/api.py"],
            time_window_hours=24,
        )

        detected_time = datetime(2026, 9, 26, 22, 4, 0, tzinfo=timezone.utc)
        return build_from_findings(
            log_findings=lf,
            git_findings=gf,
            log_lines=raw_lines,
            detected_time=detected_time,
            detected_description="Incident INC-2026-0926 reported via PagerDuty.",
        )

    def test_returns_list_of_timeline_events(self):
        events = self._build()
        assert isinstance(events, list)
        assert all(isinstance(e, TimelineEvent) for e in events)

    def test_at_least_four_events(self):
        """deployment + commit + first_error + recurrence or alert = ≥4"""
        events = self._build()
        assert len(events) >= 4

    def test_deployment_event_present(self):
        events = self._build()
        assert any(e.event == EVENT_DEPLOYMENT for e in events)

    def test_deployment_mentions_version(self):
        events = self._build()
        dep = next(e for e in events if e.event == EVENT_DEPLOYMENT)
        assert "2.4.1" in dep.description

    def test_deployment_mentions_commit(self):
        events = self._build()
        dep = next(e for e in events if e.event == EVENT_DEPLOYMENT)
        # Short SHA from the log (cd5456a)
        assert "cd5456a" in dep.description

    def test_commit_event_present(self):
        events = self._build()
        assert any(e.event == EVENT_COMMIT for e in events)

    def test_commit_event_mentions_refactor(self):
        events = self._build()
        commit_evs = [e for e in events if e.event == EVENT_COMMIT]
        assert any("simplif" in e.description.lower() or "refactor" in e.description.lower()
                   for e in commit_evs)

    def test_first_error_event_present(self):
        events = self._build()
        assert any(e.event == EVENT_FIRST_ERROR for e in events)

    def test_first_error_mentions_typeerror(self):
        events = self._build()
        fe = next(e for e in events if e.event == EVENT_FIRST_ERROR)
        assert "TypeError" in fe.description

    def test_first_error_timestamp_is_22_01(self):
        events = self._build()
        fe = next(e for e in events if e.event == EVENT_FIRST_ERROR)
        # Timestamp must contain 22:01 (UTC)
        assert "22:01" in fe.timestamp

    def test_alert_event_present(self):
        events = self._build()
        assert any(e.event == EVENT_ALERT for e in events)

    def test_detected_event_present(self):
        events = self._build()
        assert any(e.event == EVENT_DETECTED for e in events)

    def test_detected_event_mentions_incident_id(self):
        events = self._build()
        det = next(e for e in events if e.event == EVENT_DETECTED)
        assert "INC-2026-0926" in det.description

    def test_chronological_order(self):
        """Every dated event must be >= the previous dated event."""
        events = self._build()
        last_dt: datetime | None = None
        for e in events:
            if e.timestamp == "(time unknown)":
                continue
            dt = datetime.fromisoformat(e.timestamp.replace("Z", "+00:00"))
            if last_dt is not None:
                assert dt >= last_dt, (
                    f"Out-of-order events: {last_dt.isoformat()} > {dt.isoformat()}"
                )
            last_dt = dt

    def test_deployment_before_first_error(self):
        events = self._build()
        cats = [e.event for e in events if e.timestamp != "(time unknown)"]
        assert cats.index(EVENT_DEPLOYMENT) < cats.index(EVENT_FIRST_ERROR)

    def test_no_invented_timestamps(self):
        """
        All timestamps must either be '(time unknown)' or a valid ISO-8601 UTC
        string that can be parsed.  No other formats indicate invented data.
        """
        events = self._build()
        for e in events:
            if e.timestamp == "(time unknown)":
                continue
            # Must be parseable ISO-8601
            try:
                datetime.fromisoformat(e.timestamp.replace("Z", "+00:00"))
            except ValueError:
                pytest.fail(f"Unparseable timestamp: {e.timestamp!r}")

    def test_all_descriptions_non_empty(self):
        events = self._build()
        for e in events:
            assert e.description.strip() != "", f"Empty description for event {e.event!r}"

    def test_all_event_fields_non_empty(self):
        events = self._build()
        for e in events:
            assert e.event != "", "event field should never be empty"

    def test_chain_deployment_then_first_error_then_alert(self):
        """The canonical incident chain must be preserved in order."""
        events = self._build()
        dated = [(e.timestamp, e.event) for e in events if e.timestamp != "(time unknown)"]
        cats = [cat for _, cat in dated]
        dep_idx = cats.index(EVENT_DEPLOYMENT)
        fe_idx = cats.index(EVENT_FIRST_ERROR)
        al_idx = cats.index(EVENT_ALERT)
        assert dep_idx < fe_idx < al_idx
