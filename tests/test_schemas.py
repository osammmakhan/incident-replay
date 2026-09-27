"""Unit tests for incident_replay.models.schemas."""

import json

import pytest
from pydantic import ValidationError

from incident_replay.models import (
    Evidence,
    IncidentReport,
    RegressionTest,
    SuggestedFix,
    TimelineEvent,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_report(**overrides) -> IncidentReport:
    """Return a fully-populated IncidentReport with sensible defaults."""
    kwargs = dict(
        incident_summary="Service latency spike detected",
        timeline=[
            TimelineEvent(timestamp="2024-01-15T10:00:00Z", description="Latency exceeds threshold"),
            TimelineEvent(timestamp="2024-01-15T10:05:00Z", description="Alerts fired"),
        ],
        root_cause="N+1 query introduced in commit abc1234",
        evidence=[
            Evidence(source="log", location="app/views.py:42", observation="High DB query count", relevance="1000 queries/req indicates N+1"),
            Evidence(source="git", location="commit abc1234", observation="Offending commit", relevance="Removed eager-load"),
            Evidence(source="code", location="app/views.py:list_users", observation="Missing eager-load", relevance="User.objects.all() fetches each profile separately"),
            Evidence(source="test", location="tests/test_views.py:test_user_list", observation="Regression test failure", relevance="Confirms query count regression"),
        ],
        affected_files=["app/views.py"],
        affected_functions=["list_users"],
        suspicious_commit="abc1234",
        confidence=0.92,
        regression_test=RegressionTest(
            name="test_list_users_query_count",
            code="def test_list_users_query_count(client):\n    with assert_num_queries(1):\n        client.get('/users/')\n",
        ),
        suggested_fix=SuggestedFix(
            description="Add select_related to the queryset",
            patch="- User.objects.all()\n+ User.objects.select_related('profile').all()\n",
        ),
        verification_result=VerificationResult(passed=True, output="1 passed in 0.12s"),
    )
    kwargs.update(overrides)
    return IncidentReport(**kwargs)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_minimal_report_builds(self):
        report = _minimal_report()
        assert report.incident_summary == "Service latency spike detected"
        assert len(report.timeline) == 2
        assert len(report.evidence) == 4
        assert report.suspicious_commit == "abc1234"
        assert report.confidence == 0.92
        assert report.verification_result.passed is True

    def test_suspicious_commit_optional(self):
        report = _minimal_report(suspicious_commit=None)
        assert report.suspicious_commit is None

    def test_evidence_all_sources_accepted(self):
        for src in ("log", "git", "code", "test"):
            ev = Evidence(source=src, location="file.py:1", observation="ok", relevance="relevant")
            assert ev.source == src

    def test_evidence_invalid_source_rejected(self):
        with pytest.raises(ValidationError):
            Evidence(source="network", location="file.py:1", observation="bad source", relevance="none")

    def test_regression_test_defaults_language_python(self):
        rt = RegressionTest(name="t", code="pass")
        assert rt.language == "python"


# ---------------------------------------------------------------------------
# Confidence validation tests
# ---------------------------------------------------------------------------

class TestConfidenceValidation:
    def test_confidence_zero_accepted(self):
        report = _minimal_report(confidence=0.0)
        assert report.confidence == 0.0

    def test_confidence_one_accepted(self):
        report = _minimal_report(confidence=1.0)
        assert report.confidence == 1.0

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            _minimal_report(confidence=1.01)

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            _minimal_report(confidence=-0.01)


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_model_dump_returns_dict(self):
        report = _minimal_report()
        data = report.model_dump()
        assert isinstance(data, dict)
        assert data["incident_summary"] == "Service latency spike detected"
        assert data["confidence"] == 0.92

    def test_model_dump_is_json_serializable(self):
        report = _minimal_report()
        raw = json.dumps(report.model_dump())
        roundtrip = json.loads(raw)
        assert roundtrip["root_cause"] == "N+1 query introduced in commit abc1234"
        assert roundtrip["verification_result"]["passed"] is True

    def test_roundtrip_via_model_validate(self):
        report = _minimal_report()
        data = report.model_dump()
        restored = IncidentReport.model_validate(data)
        assert restored == report

    def test_nested_evidence_serialized(self):
        report = _minimal_report()
        data = report.model_dump()
        sources = {ev["source"] for ev in data["evidence"]}
        assert sources == {"log", "git", "code", "test"}
        assert all("location" in ev and "observation" in ev and "relevance" in ev for ev in data["evidence"])

    def test_timeline_order_preserved(self):
        report = _minimal_report()
        data = report.model_dump()
        timestamps = [e["timestamp"] for e in data["timeline"]]
        assert timestamps == ["2024-01-15T10:00:00Z", "2024-01-15T10:05:00Z"]
