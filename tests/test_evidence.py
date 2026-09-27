"""
Tests for incident_replay.analysis.evidence.

All tests are fully self-contained: findings objects are constructed inline
as plain dataclass instances (the real agents are never executed).

The test suite validates:
  - each converter (log/git/code/test) emits valid Evidence objects
  - collect() with no inputs returns [], partial inputs keep the fixed
    log → git → code → test order, and duplicates are removed
  - dedupe() removes exact/normalized duplicates while preserving order
  - no invention: values absent from the findings never appear in output
  - edge cases: empty lists, stack_trace_present=False, malformed objects
  - causal-sounding input text is passed through verbatim, deterministically
"""

from __future__ import annotations

from datetime import datetime, timezone

from incident_replay.agents.code_agent import CodeFindings, CodeObservation
from incident_replay.agents.git_agent import CommitFinding, GitFindings
from incident_replay.agents.log_agent import LogFindings, RequestSummary
from incident_replay.agents.test_agent import (
    TestCase,
    TestInvestigatorFindings,
)
from incident_replay.analysis.evidence import (
    collect,
    dedupe,
    from_code_findings,
    from_git_findings,
    from_log_findings,
    from_test_findings,
)
from incident_replay.models.schemas import Evidence

UTC = timezone.utc

T_FIRST_ERR = datetime(2026, 9, 26, 22, 1, 5, 47000, tzinfo=UTC)
T_COMMIT = datetime(2026, 9, 26, 18, 0, 0, tzinfo=UTC)

STACK_TEXT = (
    "Traceback (most recent call last):\n"
    '  File "app/checkout.py", line 87, in calculate_discount\n'
    "    return price * discount\n"
    "TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'\n"
)


# ---------------------------------------------------------------------------
# Inline findings builders (dataclasses only — agents are never run)
# ---------------------------------------------------------------------------

def make_log_findings(**overrides) -> LogFindings:
    base = dict(
        log_path="logs/production.log",
        total_lines=42,
        parse_warnings=[],
        log_start_time=datetime(2026, 9, 26, 22, 0, 0, tzinfo=UTC),
        log_end_time=datetime(2026, 9, 26, 22, 4, 0, tzinfo=UTC),
        first_error_time=T_FIRST_ERR,
        error_count=4,
        dominant_error_signature="TypeError: unsupported operand",
        unique_error_signatures=["TypeError: unsupported operand"],
        affected_endpoints=["POST /checkout"],
        http_500_count=3,
        failing_requests=[RequestSummary(
            request_id="req-ccc",
            endpoint="POST /checkout",
            http_status=500,
            error_type="TypeError",
            error_message="unsupported operand",
            suspicious_kv={"discount": "null"},
            raw_error_lines=["...ERROR..."],
        )],
        suspicious_input_patterns=["discount=null"],
        stack_trace_present=True,
        stack_trace=STACK_TEXT,
        recurrence_count=3,
        recurrence_window_seconds=145.0,
        interpretation_hints=[],
    )
    base.update(overrides)
    return LogFindings(**base)


def make_commit(short_sha: str = "abc1234", **overrides) -> CommitFinding:
    base = dict(
        sha=short_sha + "0" * (40 - len(short_sha)),
        short_sha=short_sha,
        author="Alice",
        date=T_COMMIT,
        message="refactor: simplify discount handling",
        changed_files=["app/checkout.py"],
        relevant_diff="- total = price * discount",
        relevance_reasons=[
            "Commit is within 24h before the incident timestamp.",
            "Diff removes null-safety guard on field 'discount'.",
        ],
        relevance_score=3,
    )
    base.update(overrides)
    return CommitFinding(**base)


def make_git_findings(**overrides) -> GitFindings:
    base = dict(
        repo_path="f:/repo",
        commits_inspected=12,
        parse_warnings=[],
        all_commits=[],
        suspicious_commits=[make_commit()],
        top_suspect=None,
        interpretation_hints=[],
    )
    base.update(overrides)
    return GitFindings(**base)


def make_code_observation(**overrides) -> CodeObservation:
    base = dict(
        file="app/checkout.py",
        function="CheckoutService.calculate_discount",
        start_line=80,
        end_line=95,
        source_snippet="def calculate_discount(price, discount):",
        observation=(
            "calculate_discount multiplies price by discount without a "
            "None guard."
        ),
        relevance="Matches error keyword 'discount'.",
    )
    base.update(overrides)
    return CodeObservation(**base)


def make_code_findings(**overrides) -> CodeFindings:
    base = dict(
        repo_path="f:/repo",
        files_inspected=["app/checkout.py"],
        files_missing=[],
        parse_warnings=[],
        observations=[make_code_observation()],
        affected_files=["app/checkout.py"],
        affected_functions=["CheckoutService.calculate_discount"],
        interpretation_hints=[],
    )
    base.update(overrides)
    return CodeFindings(**base)


def make_test_case(**overrides) -> TestCase:
    base = dict(
        name="test_ten_percent_discount",
        qualified_name="TestCalc.test_ten_percent_discount",
        file="tests/test_checkout.py",
        start_line=10,
        end_line=18,
        source_snippet="def test_ten_percent_discount(): ...",
        field_values_tested={"discount": ["0.10"]},
        covers_affected_function=True,
        scenario_description="10% discount applied to a subtotal",
    )
    base.update(overrides)
    return TestCase(**base)


def make_test_findings(**overrides) -> TestInvestigatorFindings:
    base = dict(
        test_root="tests",
        test_files_found=["tests/test_checkout.py"],
        test_files_relevant=["tests/test_checkout.py"],
        relevant_test_names=["TestCalc.test_ten_percent_discount"],
        covered_scenarios=["10% discount applied to a subtotal"],
        missing_scenarios=[
            "calculate_discount: 'discount' is never passed as None/null"
        ],
        relevance_to_incident="high",
        interpretation_hints=[],
        file_findings=[],
        all_relevant_tests=[make_test_case()],
    )
    base.update(overrides)
    return TestInvestigatorFindings(**base)


def all_fields_non_empty(ev: Evidence) -> bool:
    return bool(
        ev.source
        and ev.location.strip()
        and ev.observation.strip()
        and ev.relevance.strip()
    )


def dumped(evidence) -> str:
    return " || ".join(
        f"{e.source}|{e.location}|{e.observation}|{e.relevance}"
        for e in evidence
    )


# ===========================================================================
# from_log_findings
# ===========================================================================

class TestFromLogFindings:
    def test_none_returns_empty(self):
        assert from_log_findings(None) == []

    def test_emits_valid_evidence(self):
        out = from_log_findings(make_log_findings())
        assert len(out) > 0
        for ev in out:
            assert isinstance(ev, Evidence)
            assert ev.source == "log"
            assert all_fields_non_empty(ev)
            # pydantic accepts the round-tripped object
            assert Evidence.model_validate(ev.model_dump()) == ev

    def test_signature_fact_contains_signature_count_and_time(self):
        out = from_log_findings(make_log_findings())
        sig_ev = next(e for e in out if "TypeError" in e.observation)
        assert "TypeError: unsupported operand" in sig_ev.observation
        assert "4 error line(s)" in sig_ev.observation
        assert "2026-09-26T22:01:05Z" in sig_ev.observation

    def test_signature_location_carries_first_error_time(self):
        out = from_log_findings(make_log_findings())
        sig_ev = next(e for e in out if "TypeError" in e.observation)
        assert sig_ev.location == (
            "logs/production.log (first error 2026-09-26T22:01:05Z)"
        )

    def test_no_signature_emits_no_signature_fact(self):
        out = from_log_findings(make_log_findings(
            dominant_error_signature=None,
            first_error_time=None,
        ))
        assert not any("dominant error signature" in e.observation
                       for e in out)

    def test_endpoint_fact_includes_counts(self):
        out = from_log_findings(make_log_findings())
        ep_ev = next(e for e in out if "POST /checkout" in e.location)
        assert ep_ev.location == "logs/production.log (endpoint=POST /checkout)"
        assert "3 HTTP 500 response(s)" in ep_ev.observation
        assert "req-ccc" in ep_ev.observation

    def test_suspicious_pattern_is_verbatim(self):
        out = from_log_findings(make_log_findings())
        pat_ev = next(e for e in out if "discount=null" in e.observation)
        assert "discount=null" in pat_ev.location
        assert pat_ev.observation.endswith("discount=null")

    def test_recurrence_emitted_when_count_gt_one(self):
        out = from_log_findings(make_log_findings())
        rec = next(e for e in out if "recurred" in e.observation)
        assert "3 distinct" in rec.observation
        assert "145.0 second(s)" in rec.observation
        assert rec.location == "logs/production.log (recurrence)"

    def test_no_recurrence_when_count_is_one(self):
        out = from_log_findings(make_log_findings(recurrence_count=1))
        assert not any("recurred" in e.observation for e in out)

    def test_recurrence_without_window_still_emitted(self):
        out = from_log_findings(make_log_findings(
            recurrence_count=5,
            recurrence_window_seconds=None,
        ))
        rec = next(e for e in out if "recurred" in e.observation)
        assert "recurrence window not recorded" in rec.observation

    def test_stack_trace_fact_includes_exception_and_frame(self):
        out = from_log_findings(make_log_findings())
        st = next(e for e in out if "Stack trace" in e.observation)
        assert "TypeError" in st.observation
        assert "app/checkout.py" in st.observation
        assert "line 87" in st.observation
        assert st.location == "logs/production.log (stack trace)"

    def test_no_stack_fact_when_absent(self):
        out = from_log_findings(make_log_findings(stack_trace_present=False))
        assert not any("Stack trace" in e.observation for e in out)

    def test_empty_findings_emit_nothing(self):
        out = from_log_findings(make_log_findings(
            dominant_error_signature=None,
            first_error_time=None,
            affected_endpoints=[],
            failing_requests=[],
            http_500_count=0,
            suspicious_input_patterns=[],
            recurrence_count=1,
            stack_trace_present=False,
            stack_trace=None,
        ))
        assert out == []

    def test_bare_object_missing_optional_lists_does_not_raise(self):
        class _Bare:
            log_path = "bare.log"
            dominant_error_signature = "ValueError: boom"
            error_count = 1

        out = from_log_findings(_Bare())
        assert len(out) == 1
        assert out[0].source == "log"
        assert out[0].location == "bare.log"
        assert all_fields_non_empty(out[0])

    def test_failing_requests_none_does_not_raise(self):
        out = from_log_findings(make_log_findings(failing_requests=None))
        assert isinstance(out, list)
        assert all_fields_non_empty(out[0])


# ===========================================================================
# from_git_findings
# ===========================================================================

class TestFromGitFindings:
    def test_none_returns_empty(self):
        assert from_git_findings(None) == []

    def test_empty_suspicious_commits_returns_empty(self):
        assert from_git_findings(make_git_findings(suspicious_commits=[])) == []

    def test_emits_valid_evidence_per_commit(self):
        out = from_git_findings(make_git_findings())
        assert len(out) == 1
        ev = out[0]
        assert isinstance(ev, Evidence)
        assert ev.source == "git"
        assert all_fields_non_empty(ev)
        assert Evidence.model_validate(ev.model_dump()) == ev

    def test_location_is_commit_short_sha(self):
        out = from_git_findings(make_git_findings())
        assert out[0].location == "commit abc1234"

    def test_observation_states_commit_factually(self):
        out = from_git_findings(make_git_findings())
        obs = out[0].observation
        assert "abc1234" in obs
        assert "refactor: simplify discount handling" in obs
        assert "app/checkout.py" in obs
        assert "author Alice" in obs

    def test_relevance_joins_reasons(self):
        out = from_git_findings(make_git_findings())
        assert out[0].relevance == (
            "Commit is within 24h before the incident timestamp.; "
            "Diff removes null-safety guard on field 'discount'."
        )

    def test_multiple_commits_emit_one_each(self):
        out = from_git_findings(make_git_findings(suspicious_commits=[
            make_commit("aaaa111"),
            make_commit("bbbb222"),
        ]))
        assert [e.location for e in out] == ["commit aaaa111", "commit bbbb222"]

    def test_top_suspect_is_not_emitted_separately(self):
        out = from_git_findings(make_git_findings(
            suspicious_commits=[make_commit("abc1234")],
            top_suspect=make_commit("topsuspect99"),
        ))
        assert len(out) == 1
        assert "topsuspect99" not in dumped(out)

    def test_empty_reasons_falls_back_without_empty_relevance(self):
        out = from_git_findings(make_git_findings(suspicious_commits=[
            make_commit("cccc333", relevance_reasons=[], relevance_score=2),
        ]))
        assert all_fields_non_empty(out[0])
        assert "score 2" in out[0].relevance

    def test_empty_lists_do_not_raise(self):
        out = from_git_findings(make_git_findings(
            suspicious_commits=[],
            top_suspect=None,
        ))
        assert out == []


# ===========================================================================
# from_code_findings
# ===========================================================================

class TestFromCodeFindings:
    def test_none_returns_empty(self):
        assert from_code_findings(None) == []

    def test_empty_observations_returns_empty(self):
        assert from_code_findings(make_code_findings(observations=[])) == []

    def test_emits_valid_evidence(self):
        out = from_code_findings(make_code_findings())
        assert len(out) == 1
        ev = out[0]
        assert isinstance(ev, Evidence)
        assert ev.source == "code"
        assert all_fields_non_empty(ev)
        assert Evidence.model_validate(ev.model_dump()) == ev

    def test_location_is_file_colon_line(self):
        out = from_code_findings(make_code_findings())
        assert out[0].location == "app/checkout.py:80"

    def test_observation_and_relevance_passed_through(self):
        src = make_code_observation()
        out = from_code_findings(make_code_findings())
        assert out[0].observation == src.observation
        assert out[0].relevance == src.relevance

    def test_entry_without_observation_is_skipped(self):
        out = from_code_findings(make_code_findings(observations=[
            make_code_observation(observation="   "),
            make_code_observation(),
        ]))
        assert len(out) == 1
        assert out[0].location == "app/checkout.py:80"

    def test_empty_findings_emit_nothing(self):
        out = from_code_findings(make_code_findings(
            observations=[],
            affected_files=[],
            affected_functions=[],
        ))
        assert out == []


# ===========================================================================
# from_test_findings
# ===========================================================================

class TestFromTestFindings:
    def test_none_returns_empty(self):
        assert from_test_findings(None) == []

    def test_empty_lists_emit_nothing(self):
        out = from_test_findings(make_test_findings(
            all_relevant_tests=[],
            missing_scenarios=[],
        ))
        assert out == []

    def test_emits_valid_evidence_per_test(self):
        out = from_test_findings(make_test_findings(missing_scenarios=[]))
        assert len(out) == 1
        ev = out[0]
        assert isinstance(ev, Evidence)
        assert ev.source == "test"
        assert all_fields_non_empty(ev)
        assert Evidence.model_validate(ev.model_dump()) == ev

    def test_test_location_is_file_colon_line(self):
        out = from_test_findings(make_test_findings())
        assert out[0].location == "tests/test_checkout.py:10"

    def test_test_observation_names_test_and_covers_flag(self):
        out = from_test_findings(make_test_findings())
        obs = out[0].observation
        assert "TestCalc.test_ten_percent_discount" in obs
        assert "10% discount applied to a subtotal" in obs
        assert "covers_affected_function=True" in obs

    def test_missing_scenario_uses_coverage_gap_location(self):
        out = from_test_findings(make_test_findings())
        gap = next(e for e in out if "Coverage gap" in e.observation)
        assert gap.location == "tests (coverage gap)"
        assert "discount" in gap.observation
        assert all_fields_non_empty(gap)

    def test_covered_scenarios_are_not_emitted(self):
        out = from_test_findings(make_test_findings(
            covered_scenarios=["COVERED-MARKER-XYZ"],
        ))
        assert "COVERED-MARKER-XYZ" not in dumped(out)

    def test_empty_missing_scenarios_emits_no_gap(self):
        out = from_test_findings(make_test_findings(missing_scenarios=[]))
        assert not any("coverage gap" in e.location for e in out)

    def test_relevance_mentions_incident_rating(self):
        out = from_test_findings(make_test_findings())
        assert "high" in out[0].relevance

    def test_covers_false_is_stated_factually(self):
        out = from_test_findings(make_test_findings(all_relevant_tests=[
            make_test_case(covers_affected_function=False),
        ], missing_scenarios=[]))
        assert "covers_affected_function=False" in out[0].observation


# ===========================================================================
# collect
# ===========================================================================

class TestCollect:
    def test_all_none_returns_empty(self):
        assert collect(None, None, None, None) == []
        assert collect() == []

    def test_all_sources_present_in_fixed_order(self):
        out = collect(
            log_findings=make_log_findings(),
            git_findings=make_git_findings(),
            code_findings=make_code_findings(),
            test_findings=make_test_findings(),
        )
        order = ["log", "git", "code", "test"]
        sources = [e.source for e in out]
        assert set(sources) == set(order)
        assert sources == sorted(sources, key=order.index)

    def test_partial_input_only_emits_supplied_sources_in_order(self):
        # keyword order deliberately reversed; emission order is fixed
        out = collect(
            test_findings=make_test_findings(),
            code_findings=make_code_findings(),
        )
        sources = [e.source for e in out]
        assert set(sources) == {"code", "test"}
        assert sources == ["code"] * sources.count("code") + \
            ["test"] * sources.count("test")

    def test_log_only(self):
        out = collect(log_findings=make_log_findings())
        assert out != []
        assert all(e.source == "log" for e in out)

    def test_git_and_log_fixed_order(self):
        out = collect(
            git_findings=make_git_findings(),
            log_findings=make_log_findings(),
        )
        sources = [e.source for e in out]
        assert sources.index("log") < sources.index("git")
        assert set(sources) == {"log", "git"}

    def test_collect_applies_dedupe(self):
        lf = make_log_findings(suspicious_input_patterns=[
            "discount=null",
            "discount=null",
        ])
        out = collect(log_findings=lf)
        pat = [e for e in out if "discount=null" in e.observation]
        assert len(pat) == 1

    def test_collect_returns_valid_evidence(self):
        out = collect(
            log_findings=make_log_findings(),
            git_findings=make_git_findings(),
            code_findings=make_code_findings(),
            test_findings=make_test_findings(),
        )
        for ev in out:
            assert isinstance(ev, Evidence)
            assert all_fields_non_empty(ev)

    def test_no_repeated_source_location_observation(self):
        out = collect(
            log_findings=make_log_findings(),
            git_findings=make_git_findings(),
            code_findings=make_code_findings(),
            test_findings=make_test_findings(),
        )
        keys = [(e.source, e.location, e.observation) for e in out]
        assert len(keys) == len(set(keys))


# ===========================================================================
# dedupe
# ===========================================================================

def _ev(source="log", location="a.log", observation="obs", relevance="rel"):
    return Evidence(
        source=source, location=location,
        observation=observation, relevance=relevance,
    )


class TestDedupe:
    def test_empty_list(self):
        assert dedupe([]) == []

    def test_exact_duplicates_removed_first_kept(self):
        first = _ev(relevance="first")
        second = _ev(relevance="second")
        out = dedupe([first, second])
        assert len(out) == 1
        assert out[0] is first
        assert out[0].relevance == "first"

    def test_normalization_case_and_whitespace(self):
        a = _ev(observation="Dominant   error\tseen")
        b = _ev(observation="dominant error seen")
        assert len(dedupe([a, b])) == 1

    def test_normalization_applies_to_location(self):
        a = _ev(location="commit ABC1234")
        b = _ev(location="commit   abc1234")
        assert len(dedupe([a, b])) == 1

    def test_distinct_entries_preserved_in_order(self):
        a = _ev(location="a.log", observation="one")
        b = _ev(location="b.log", observation="two")
        c = _ev(source="git", location="a.log", observation="one")
        out = dedupe([a, b, c])
        assert out == [a, b, c]

    def test_different_relevance_still_duplicate(self):
        a = _ev(relevance="r1")
        b = _ev(relevance="r2")
        assert len(dedupe([a, b])) == 1

    def test_duplicates_scattered_across_list(self):
        a = _ev(observation="same")
        b = _ev(observation="other")
        c = _ev(observation="same")
        out = dedupe([a, b, c])
        assert [e.observation for e in out] == ["same", "other"]


# ===========================================================================
# No invention
# ===========================================================================

class TestNoInvention:
    def _collect(self):
        return collect(
            log_findings=make_log_findings(),
            git_findings=make_git_findings(),
            code_findings=make_code_findings(),
            test_findings=make_test_findings(),
        )

    def test_fabricated_commit_sha_never_appears(self):
        assert "deadbeef42" not in dumped(self._collect())

    def test_top_suspect_sha_not_in_findings_never_appears(self):
        gf = make_git_findings(top_suspect=make_commit("cafe0bad"))
        out = from_git_findings(gf)
        assert "cafe0bad" not in dumped(out)

    def test_fabricated_endpoint_never_appears(self):
        assert "GET /invented" not in dumped(self._collect())

    def test_fabricated_file_never_appears(self):
        assert "src/never_mentioned.py" not in dumped(self._collect())

    def test_fabricated_timestamp_never_appears(self):
        assert "2030-01-01" not in dumped(self._collect())

    def test_real_values_do_appear(self):
        text = dumped(self._collect())
        assert "abc1234" in text                 # commit short sha
        assert "POST /checkout" in text          # endpoint from log
        assert "app/checkout.py:80" in text      # code file:line
        assert "tests/test_checkout.py:10" in text
        assert "tests (coverage gap)" in text

    def test_generated_text_has_no_causal_language(self):
        out = self._collect()
        banned = ("caused", "root cause", "introduced the bug")
        for ev in out:
            text = f"{ev.observation} {ev.relevance}".lower()
            for phrase in banned:
                assert phrase not in text, (
                    f"causal phrase {phrase!r} in {ev.observation!r}"
                )


# ===========================================================================
# Edge cases / malformed inputs
# ===========================================================================

class TestEdgeCases:
    def test_suspicious_commits_empty(self):
        assert from_git_findings(make_git_findings(suspicious_commits=[])) == []

    def test_missing_scenarios_empty(self):
        out = from_test_findings(make_test_findings(missing_scenarios=[]))
        assert all("coverage gap" not in e.location for e in out)

    def test_stack_trace_present_false_with_stack_text(self):
        out = from_log_findings(make_log_findings(
            stack_trace_present=False,
            stack_trace=STACK_TEXT,
        ))
        assert not any("Stack trace" in e.observation for e in out)

    def test_all_converters_tolerate_none(self):
        assert from_log_findings(None) == []
        assert from_git_findings(None) == []
        assert from_code_findings(None) == []
        assert from_test_findings(None) == []

    def test_log_findings_with_all_lists_empty(self):
        out = from_log_findings(make_log_findings(
            affected_endpoints=[],
            failing_requests=[],
            suspicious_input_patterns=[],
            recurrence_count=0,
            stack_trace_present=False,
            dominant_error_signature=None,
        ))
        assert out == []

    def test_test_findings_with_lists_none(self):
        out = from_test_findings(make_test_findings(
            all_relevant_tests=None,
            missing_scenarios=None,
        ))
        assert out == []

    def test_dedupe_handles_none_input(self):
        assert dedupe(None) == []


# ===========================================================================
# Causal input text is passed through verbatim (deterministic behavior)
# ===========================================================================

CAUSAL_TEXT = "The refactor caused the outage (root cause candidate)."


class TestCausalPassthrough:
    def test_code_observation_passed_through_verbatim(self):
        src = make_code_observation(observation=CAUSAL_TEXT)
        out = from_code_findings(make_code_findings(observations=[src]))
        assert out[0].observation == CAUSAL_TEXT

    def test_log_pattern_passed_through_verbatim(self):
        out = from_log_findings(make_log_findings(
            suspicious_input_patterns=[CAUSAL_TEXT],
        ))
        pat = next(e for e in out if CAUSAL_TEXT in e.observation)
        assert pat.observation.endswith(CAUSAL_TEXT)

    def test_git_message_passed_through_verbatim(self):
        out = from_git_findings(make_git_findings(suspicious_commits=[
            make_commit(message=CAUSAL_TEXT),
        ]))
        assert CAUSAL_TEXT in out[0].observation

    def test_behavior_is_deterministic(self):
        args = dict(
            log_findings=make_log_findings(),
            git_findings=make_git_findings(),
            code_findings=make_code_findings(),
            test_findings=make_test_findings(),
        )
        first = collect(**args)
        second = collect(**args)
        assert first == second
        assert [e.model_dump() for e in first] == \
            [e.model_dump() for e in second]
