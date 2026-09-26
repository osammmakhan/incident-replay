"""
Tests for incident_replay.agents.log_agent.

All tests are self-contained: no filesystem access needed for the main
test classes.  The final class (TestLogAgentWithDemoLog) exercises the
real production log from the demo project and is skipped when it is absent.
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest

from incident_replay.agents.log_agent import (
    LogAgentError,
    LogFindings,
    RequestSummary,
    run,
    run_from_lines,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

DEMO_LOG = Path("f:/GenAI/incident-replay-demo/logs/production.log")
DEMO_STACKTRACE = Path("f:/GenAI/incident-replay-demo/incident/stacktrace.txt")
DEMO_AVAILABLE = DEMO_LOG.is_file()

# A minimal but representative set of log lines that mirrors the real format
NOMINAL_LINES = [
    # Deployment
    "2026-09-26 22:00:01,042 INFO  shopco.deploy  Starting deployment version=2.4.1 commit=cd5456a",
    # Successful requests (no discount, numeric discount)
    "2026-09-26 22:00:15,334 INFO  shopco.api     checkout request_id=req-aaa order_id=ORD-01 customer_id=CUST-1",
    "2026-09-26 22:00:15,391 INFO  shopco.api     checkout OK request_id=req-aaa total=89.97",
    "2026-09-26 22:00:22,102 INFO  shopco.api     checkout request_id=req-bbb order_id=ORD-02 customer_id=CUST-2 discount=0.1",
    "2026-09-26 22:00:22,149 INFO  shopco.api     checkout OK request_id=req-bbb total=58.48",
    # First failure — discount=null
    "2026-09-26 22:01:05,003 INFO  shopco.api     checkout request_id=req-ccc order_id=ORD-03 customer_id=CUST-3 discount=null",
    "2026-09-26 22:01:05,047 ERROR shopco.api     checkout FAILED request_id=req-ccc error=TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'",
    "2026-09-26 22:01:05,049 ERROR shopco.api     POST /checkout 500 44ms request_id=req-ccc",
    # Second failure
    "2026-09-26 22:01:44,560 INFO  shopco.api     checkout request_id=req-ddd order_id=ORD-04 customer_id=CUST-4 discount=null",
    "2026-09-26 22:01:44,601 ERROR shopco.api     checkout FAILED request_id=req-ddd error=TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'",
    "2026-09-26 22:01:44,603 ERROR shopco.api     POST /checkout 500 41ms request_id=req-ddd",
    # Alert
    "2026-09-26 22:03:55,001 WARN  shopco.ops     5xx spike detected endpoint=POST /checkout error_rate=0.56",
]

CLEAN_LINES = [
    "2026-09-26 22:00:01,042 INFO  shopco.deploy  Starting deployment version=1.0.0",
    "2026-09-26 22:00:05,100 INFO  shopco.api     checkout request_id=req-x1 order_id=ORD-01",
    "2026-09-26 22:00:05,150 INFO  shopco.api     checkout OK request_id=req-x1 total=50.00",
]

STACKTRACE_TEXT = textwrap.dedent("""\
    Traceback (most recent call last):
      File "/app/app/checkout.py", line 45, in calculate_discount
        discount_rate = max(0.0, min(order.discount, self.MAX_DISCOUNT_RATE))
    TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'
""")


# ===========================================================================
# Empty / missing input handling
# ===========================================================================

class TestEdgeCases:
    def test_empty_lines_returns_empty_findings(self):
        f = run_from_lines([])
        assert f.error_count == 0
        assert f.total_lines == 0
        assert "no lines" in f.parse_warnings[0].lower()

    def test_missing_log_file_returns_empty_findings(self, tmp_path):
        f = run(str(tmp_path / "ghost.log"))
        assert f.error_count == 0
        assert f.total_lines == 0
        assert len(f.parse_warnings) == 1
        assert "not found" in f.parse_warnings[0].lower()

    def test_empty_path_string_returns_empty_findings(self):
        f = run("")
        assert f.error_count == 0
        assert f.log_path == "<no path supplied>"

    def test_whitespace_only_path_returns_empty_findings(self):
        f = run("   ")
        assert f.error_count == 0

    def test_directory_path_raises_log_agent_error(self, tmp_path):
        with pytest.raises(LogAgentError):
            run(str(tmp_path))

    def test_clean_log_no_errors(self):
        f = run_from_lines(CLEAN_LINES)
        assert f.error_count == 0
        assert f.dominant_error_signature is None
        assert f.failing_requests == []
        assert f.http_500_count == 0

    def test_malformed_lines_stored_with_warning(self):
        lines = ["this is not a log line at all", "neither is this"]
        f = run_from_lines(lines)
        assert len(f.parse_warnings) == 1
        assert "2" in f.parse_warnings[0]   # "2 line(s)"

    def test_mixed_good_and_bad_lines(self):
        mixed = NOMINAL_LINES[:3] + ["not a log line"] + NOMINAL_LINES[3:]
        f = run_from_lines(mixed)
        assert f.error_count > 0          # real errors still found
        assert len(f.parse_warnings) == 1


# ===========================================================================
# Timeline extraction
# ===========================================================================

class TestTimeline:
    def test_log_start_time_is_first_line(self):
        f = run_from_lines(NOMINAL_LINES)
        assert f.log_start_time == datetime(2026, 9, 26, 22, 0, 1, 42000, tzinfo=timezone.utc)

    def test_log_end_time_is_last_line(self):
        f = run_from_lines(NOMINAL_LINES)
        assert f.log_end_time is not None
        assert f.log_end_time > f.log_start_time

    def test_first_error_time_is_earliest_error(self):
        f = run_from_lines(NOMINAL_LINES)
        assert f.first_error_time == datetime(2026, 9, 26, 22, 1, 5, 47000, tzinfo=timezone.utc)

    def test_no_timestamps_when_all_lines_malformed(self):
        f = run_from_lines(["bad line one", "bad line two"])
        assert f.log_start_time is None
        assert f.log_end_time is None
        assert f.first_error_time is None


# ===========================================================================
# Error pattern
# ===========================================================================

class TestErrorPattern:
    def test_error_count_matches_error_lines(self):
        f = run_from_lines(NOMINAL_LINES)
        # 2x FAILED + 2x HTTP-500 = 4 ERROR lines
        assert f.error_count == 4

    def test_dominant_error_signature_contains_type(self):
        f = run_from_lines(NOMINAL_LINES)
        assert f.dominant_error_signature is not None
        assert "TypeError" in f.dominant_error_signature

    def test_unique_error_signatures_deduplication(self):
        f = run_from_lines(NOMINAL_LINES)
        # All failures carry the same TypeError — only one unique signature
        type_errors = [s for s in f.unique_error_signatures if "TypeError" in s]
        assert len(type_errors) == 1

    def test_affected_endpoints_identified(self):
        f = run_from_lines(NOMINAL_LINES)
        assert "POST /checkout" in f.affected_endpoints

    def test_http_500_count(self):
        f = run_from_lines(NOMINAL_LINES)
        assert f.http_500_count == 2


# ===========================================================================
# Per-request breakdown
# ===========================================================================

class TestFailingRequests:
    def test_one_summary_per_failing_request_id(self):
        f = run_from_lines(NOMINAL_LINES)
        ids = [r.request_id for r in f.failing_requests]
        assert "req-ccc" in ids
        assert "req-ddd" in ids
        assert len(ids) == len(set(ids))   # no duplicates

    def test_request_summary_has_endpoint(self):
        f = run_from_lines(NOMINAL_LINES)
        ccc = next(r for r in f.failing_requests if r.request_id == "req-ccc")
        assert ccc.endpoint == "POST /checkout"
        assert ccc.http_status == 500

    def test_request_summary_has_error_type(self):
        f = run_from_lines(NOMINAL_LINES)
        ccc = next(r for r in f.failing_requests if r.request_id == "req-ccc")
        assert ccc.error_type == "TypeError"
        assert ccc.error_message is not None
        assert "NoneType" in ccc.error_message

    def test_raw_error_lines_are_verbatim(self):
        f = run_from_lines(NOMINAL_LINES)
        ccc = next(r for r in f.failing_requests if r.request_id == "req-ccc")
        assert any("checkout FAILED" in l for l in ccc.raw_error_lines)

    def test_successful_requests_not_in_failing_list(self):
        f = run_from_lines(NOMINAL_LINES)
        ids = {r.request_id for r in f.failing_requests}
        assert "req-aaa" not in ids
        assert "req-bbb" not in ids


# ===========================================================================
# Suspicious input detection
# ===========================================================================

class TestSuspiciousInputs:
    def test_null_discount_identified_as_suspicious(self):
        f = run_from_lines(NOMINAL_LINES)
        assert "discount=null" in f.suspicious_input_patterns

    def test_numeric_discount_not_suspicious(self):
        f = run_from_lines(NOMINAL_LINES)
        # discount=0.1 appears in a successful request — must not be flagged
        assert "discount=0.1" not in f.suspicious_input_patterns

    def test_no_suspicious_inputs_in_clean_log(self):
        f = run_from_lines(CLEAN_LINES)
        assert f.suspicious_input_patterns == []

    def test_per_request_suspicious_kv(self):
        f = run_from_lines(NOMINAL_LINES)
        ccc = next(r for r in f.failing_requests if r.request_id == "req-ccc")
        # discount=null should be captured in the per-request suspicious_kv
        assert "discount" in ccc.suspicious_kv
        assert ccc.suspicious_kv["discount"] == "null"


# ===========================================================================
# Stack trace handling
# ===========================================================================

class TestStackTrace:
    def test_external_stack_trace_stored_verbatim(self):
        f = run_from_lines(NOMINAL_LINES, stack_trace=STACKTRACE_TEXT)
        assert f.stack_trace_present is True
        assert f.stack_trace == STACKTRACE_TEXT

    def test_no_stack_trace_when_absent(self):
        f = run_from_lines(NOMINAL_LINES)
        assert f.stack_trace_present is False
        assert f.stack_trace is None

    def test_inline_traceback_extracted_from_log(self):
        tb_lines = NOMINAL_LINES + [
            "Traceback (most recent call last):",
            '  File "/app/checkout.py", line 45, in calculate_discount',
            "    return order.total * order.discount",
            "TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'",
        ]
        f = run_from_lines(tb_lines)
        assert f.stack_trace_present is True
        assert "Traceback" in f.stack_trace

    def test_external_trace_takes_priority_over_inline(self):
        tb_lines = NOMINAL_LINES + [
            "Traceback (most recent call last):",
            '  File "/app/checkout.py", line 45, in calculate_discount',
            "TypeError: SomeOtherError",
        ]
        f = run_from_lines(tb_lines, stack_trace=STACKTRACE_TEXT)
        # External trace is authoritative
        assert f.stack_trace == STACKTRACE_TEXT


# ===========================================================================
# Recurrence
# ===========================================================================

class TestRecurrence:
    def test_recurrence_count_matches_distinct_failing_ids(self):
        f = run_from_lines(NOMINAL_LINES)
        assert f.recurrence_count == 2

    def test_recurrence_window_seconds_positive(self):
        f = run_from_lines(NOMINAL_LINES)
        assert f.recurrence_window_seconds is not None
        assert f.recurrence_window_seconds > 0

    def test_single_error_has_no_recurrence_window(self):
        # NOMINAL_LINES[:8] contains one failing request (req-ccc) with two
        # ERROR lines (FAILED + HTTP-500).  Both belong to the same request_id,
        # so the recurrence window — measured per distinct failing request —
        # must be None.
        single_error = NOMINAL_LINES[:8]
        f = run_from_lines(single_error)
        assert f.recurrence_count == 1
        assert f.recurrence_window_seconds is None

    def test_no_errors_has_zero_recurrence(self):
        f = run_from_lines(CLEAN_LINES)
        assert f.recurrence_count == 0
        assert f.recurrence_window_seconds is None


# ===========================================================================
# Observation / interpretation separation
# ===========================================================================

class TestObservationInterpretationSeparation:
    """Verify that findings fields contain observations, hints contain guesses."""

    def test_error_signature_is_verbatim_from_log(self):
        f = run_from_lines(NOMINAL_LINES)
        # The signature must be drawn from what's literally in the log
        assert f.dominant_error_signature is not None
        # It must NOT be a narrative claim
        assert "root cause" not in f.dominant_error_signature.lower()
        assert "caused by" not in f.dominant_error_signature.lower()

    def test_hints_are_explicitly_labelled(self):
        f = run_from_lines(NOMINAL_LINES)
        for hint in f.interpretation_hints:
            assert hint.startswith("HINT:"), (
                f"Interpretation hint is not labelled: {hint!r}"
            )

    def test_no_root_cause_claim_in_observations(self):
        f = run_from_lines(NOMINAL_LINES)
        # None of the observational fields should contain causal language
        causal_terms = ["root cause", "caused by", "because", "therefore"]
        for sig in f.unique_error_signatures:
            for term in causal_terms:
                assert term not in sig.lower()
        for pattern in f.suspicious_input_patterns:
            for term in causal_terms:
                assert term not in pattern.lower()

    def test_hints_generated_for_nominal_log(self):
        f = run_from_lines(NOMINAL_LINES)
        assert len(f.interpretation_hints) >= 1


# ===========================================================================
# Integration against the real demo log
# ===========================================================================

@pytest.mark.skipif(not DEMO_AVAILABLE, reason="demo production.log not present")
class TestLogAgentWithDemoLog:
    def _findings(self) -> LogFindings:
        st = DEMO_STACKTRACE.read_text(encoding="utf-8") if DEMO_STACKTRACE.is_file() else None
        return run(str(DEMO_LOG), stack_trace=st)

    def test_runs_without_error(self):
        f = self._findings()
        assert isinstance(f, LogFindings)

    def test_correct_total_line_count(self):
        f = self._findings()
        # The demo log has exactly 31 lines
        assert f.total_lines == 31

    def test_detects_errors(self):
        f = self._findings()
        assert f.error_count > 0

    def test_first_error_time_is_correct(self):
        f = self._findings()
        assert f.first_error_time is not None
        assert f.first_error_time.hour == 22
        assert f.first_error_time.minute == 1

    def test_identifies_affected_endpoint(self):
        f = self._findings()
        assert "POST /checkout" in f.affected_endpoints

    def test_dominant_signature_contains_typeerror(self):
        f = self._findings()
        assert f.dominant_error_signature is not None
        assert "TypeError" in f.dominant_error_signature

    def test_null_discount_flagged_as_suspicious(self):
        f = self._findings()
        assert "discount=null" in f.suspicious_input_patterns

    def test_multiple_failing_requests_detected(self):
        f = self._findings()
        assert f.recurrence_count >= 5   # demo log has 5 distinct failing IDs

    def test_recurrence_window_is_positive(self):
        f = self._findings()
        assert f.recurrence_window_seconds is not None
        assert f.recurrence_window_seconds > 0

    def test_stack_trace_stored_when_provided(self):
        if not DEMO_STACKTRACE.is_file():
            pytest.skip("stacktrace.txt not present")
        f = self._findings()
        assert f.stack_trace_present is True
        assert "TypeError" in f.stack_trace

    def test_per_request_summaries_have_endpoint_and_status(self):
        f = self._findings()
        for req in f.failing_requests:
            assert req.endpoint is not None
            assert req.http_status == 500

    def test_hints_labelled(self):
        f = self._findings()
        for hint in f.interpretation_hints:
            assert hint.startswith("HINT:")
