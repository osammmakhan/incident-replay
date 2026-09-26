"""
Tests for incident_replay.agents.synthesis (the Evidence Synthesizer).

All unit tests are fully self-contained: findings objects are constructed
inline as plain dataclass instances (the real agents are never executed)
and use invented incident names (field ``surcharge``, endpoint
``POST /price``, file ``app/pricing.py``, commit ``deadbee``).  The one
integration class runs the four real investigators against the demo project
at f:/GenAI/incident-replay-demo and is skipped when it is absent.

The test suite validates:
  - successful correlation of all four sources into a labelled hypothesis
    whose files / functions / commits are traceable to supporting evidence
  - insufficient evidence (< 2 distinct sources) stays below the moderate
    band, states so, never raises, and carries uncertainty entries
  - conflict classes (a) coverage, (b) temporal, (c) scope each move
    evidence to conflicting_evidence, note the conflict, and lower
    confidence below the otherwise-identical aligned baseline
  - anti-hardcoding: renaming the field / commit / endpoint / function
    changes the output accordingly (the synthesizer derives, never recalls)
  - LLM modularity: traceable claims are adopted while confidence stays
    deterministic; raising / None / non-dict / untraceable backends fall
    back to the identical deterministic result
  - robustness: empty, partial or malformed inputs never raise and
    repeated runs are deterministic
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from incident_replay.agents import synthesis_agent as synthesis
from incident_replay.agents.code_agent import (
    CodeFindings,
    CodeObservation,
    run as run_code_agent,
)
from incident_replay.agents.git_agent import (
    CommitFinding,
    GitFindings,
    run as run_git_agent,
)
from incident_replay.agents.log_agent import (
    LogFindings,
    RequestSummary,
    run as run_log_agent,
)
from incident_replay.agents.synthesis_agent import (
    SynthesisAgentError,
    SynthesisResult,
    run,
)
from incident_replay.agents.test_agent import (
    TestCase,
    TestInvestigatorFindings,
    run as run_test_agent,
)
from incident_replay.analysis.evidence import collect

# ---------------------------------------------------------------------------
# Constants / fixtures
# ---------------------------------------------------------------------------

DEMO_REPO = Path("f:/GenAI/incident-replay-demo")
DEMO_LOG = DEMO_REPO / "logs" / "production.log"
DEMO_STACK_TRACE = DEMO_REPO / "incident" / "stacktrace.txt"
DEMO_TESTS = DEMO_REPO / "tests"
DEMO_AVAILABLE = (
    DEMO_LOG.is_file() and (DEMO_REPO / ".git").is_dir() and DEMO_TESTS.is_dir()
)

UTC = timezone.utc

T_FIRST_ERR = datetime(2026, 9, 26, 22, 1, 5, 47000, tzinfo=UTC)
T_FIRST_ERR_BEFORE_COMMIT = datetime(2026, 9, 26, 10, 0, 0, tzinfo=UTC)
T_COMMIT = datetime(2026, 9, 26, 18, 0, 0, tzinfo=UTC)

STACK_TEXT = (
    "Traceback (most recent call last):\n"
    '  File "app/pricing.py", line 42, in apply_surcharge\n'
    "    total = base + surcharge\n"
    "TypeError: unsupported operand type(s) for +: 'float' and 'NoneType'\n"
)


# ---------------------------------------------------------------------------
# Inline findings builders (dataclasses only — agents are never run)
# ---------------------------------------------------------------------------

def make_log_findings(**overrides) -> LogFindings:
    base = dict(
        log_path="logs/service.log",
        total_lines=42,
        parse_warnings=[],
        log_start_time=datetime(2026, 9, 26, 22, 0, 0, tzinfo=UTC),
        log_end_time=datetime(2026, 9, 26, 22, 4, 0, tzinfo=UTC),
        first_error_time=T_FIRST_ERR,
        error_count=4,
        dominant_error_signature="TypeError: unsupported operand",
        unique_error_signatures=["TypeError: unsupported operand"],
        affected_endpoints=["POST /price"],
        http_500_count=3,
        failing_requests=[RequestSummary(
            request_id="req-111",
            endpoint="POST /price",
            http_status=500,
            error_type="TypeError",
            error_message="unsupported operand",
            suspicious_kv={"surcharge": "null"},
            raw_error_lines=["...ERROR..."],
        )],
        suspicious_input_patterns=["surcharge=null"],
        stack_trace_present=True,
        stack_trace=STACK_TEXT,
        recurrence_count=3,
        recurrence_window_seconds=145.0,
        interpretation_hints=[],
    )
    base.update(overrides)
    return LogFindings(**base)


def make_commit(short_sha: str = "deadbee", **overrides) -> CommitFinding:
    base = dict(
        sha=short_sha + "0" * (40 - len(short_sha)),
        short_sha=short_sha,
        author="Alice",
        date=T_COMMIT,
        message="refactor: simplify surcharge handling",
        changed_files=["app/pricing.py"],
        relevant_diff="total = base + surcharge",
        relevance_reasons=[
            "Commit is within 24h before the incident timestamp.",
            "Diff removes null-safety guard on field 'surcharge'.",
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
        file="app/pricing.py",
        function="PricingService.apply_surcharge",
        start_line=40,
        end_line=50,
        source_snippet="def apply_surcharge(base, surcharge):",
        observation=(
            "PricingService.apply_surcharge adds the field 'surcharge' with "
            "no null-safety guard."
        ),
        relevance="Matches error keyword 'surcharge'.",
    )
    base.update(overrides)
    return CodeObservation(**base)


def make_code_findings(**overrides) -> CodeFindings:
    base = dict(
        repo_path="f:/repo",
        files_inspected=["app/pricing.py"],
        files_missing=[],
        parse_warnings=[],
        observations=[make_code_observation()],
        affected_files=["app/pricing.py"],
        affected_functions=["PricingService.apply_surcharge"],
        interpretation_hints=[],
    )
    base.update(overrides)
    return CodeFindings(**base)


def make_test_case(**overrides) -> TestCase:
    base = dict(
        name="test_five_percent_surcharge",
        qualified_name="TestPricing.test_five_percent_surcharge",
        file="tests/test_pricing.py",
        start_line=10,
        end_line=18,
        source_snippet="def test_five_percent_surcharge(): ...",
        field_values_tested={"surcharge": ["0.05"]},
        covers_affected_function=True,
        scenario_description="5% surcharge applied to a subtotal",
    )
    base.update(overrides)
    return TestCase(**base)


def make_test_findings(**overrides) -> TestInvestigatorFindings:
    base = dict(
        test_root="tests",
        test_files_found=["tests/test_pricing.py"],
        test_files_relevant=["tests/test_pricing.py"],
        relevant_test_names=["TestPricing.test_five_percent_surcharge"],
        covered_scenarios=["5% surcharge applied to a subtotal"],
        missing_scenarios=[
            "apply_surcharge: 'surcharge' is never passed as None/null"
        ],
        relevance_to_incident="high",
        interpretation_hints=[],
        file_findings=[],
        all_relevant_tests=[make_test_case()],
    )
    base.update(overrides)
    return TestInvestigatorFindings(**base)


def all_findings(**overrides) -> dict:
    """The aligned four-source fixture as ``run`` keyword arguments."""
    kwargs = dict(
        log_findings=make_log_findings(),
        git_findings=make_git_findings(),
        code_findings=make_code_findings(),
        test_findings=make_test_findings(),
    )
    kwargs.update(overrides)
    return kwargs


def renamed_findings() -> dict:
    """The same incident shape with every incident-specific name changed."""
    return dict(
        log_findings=make_log_findings(
            affected_endpoints=["PATCH /billing"],
            failing_requests=[],
            suspicious_input_patterns=["levy=null"],
            stack_trace=(
                "Traceback (most recent call last):\n"
                '  File "app/billing.py", line 7, in add_levy\n'
                "    total = base + levy\n"
                "TypeError: unsupported operand type(s) for +: "
                "'float' and 'NoneType'\n"
            ),
        ),
        git_findings=make_git_findings(suspicious_commits=[make_commit(
            short_sha="f00d1ed",
            message="refactor: simplify levy handling",
            changed_files=["app/billing.py"],
            relevant_diff="total = base + levy",
            relevance_reasons=[
                "Commit is within 24h before the incident timestamp.",
                "Diff removes null-safety guard on field 'levy'.",
            ],
        )]),
        code_findings=make_code_findings(
            observations=[make_code_observation(
                file="app/billing.py",
                function="TaxEngine.add_levy",
                start_line=7,
                end_line=12,
                source_snippet="def add_levy(base, levy):",
                observation=(
                    "TaxEngine.add_levy adds the field 'levy' with no "
                    "null-safety guard."
                ),
                relevance="Matches error keyword 'levy'.",
            )],
            affected_files=["app/billing.py"],
            affected_functions=["TaxEngine.add_levy"],
        ),
        test_findings=make_test_findings(
            test_files_found=["tests/test_billing.py"],
            test_files_relevant=["tests/test_billing.py"],
            relevant_test_names=["TestBilling.test_flat_levy"],
            covered_scenarios=["flat levy applied to a subtotal"],
            missing_scenarios=["add_levy: 'levy' is never passed as None/null"],
            all_relevant_tests=[make_test_case(
                name="test_flat_levy",
                qualified_name="TestBilling.test_flat_levy",
                file="tests/test_billing.py",
                field_values_tested={"levy": ["0.02"]},
                scenario_description="flat levy applied to a subtotal",
            )],
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def supporting_text(result: SynthesisResult) -> str:
    """Case-folded ``location + observation`` of the supporting evidence."""
    return " || ".join(
        f"{ev.location} {ev.observation}" for ev in result.supporting_evidence
    ).replace("\\", "/").casefold()


def named_tokens(text: str) -> list[str]:
    """
    File / function / sha tokens named in *text*, extracted with the
    synthesizer's own regexes so the traceability check stays in sync.
    """
    found: list[str] = []
    found.extend(synthesis._FILE_TOKEN_RES.findall(text))
    found.extend(synthesis._HEX_SHA_RES.findall(text))
    for match in synthesis._DOTTED_NAME_RES.findall(text):
        if all(len(part) >= 2 for part in match.split(".")):
            found.append(match)
    for match in synthesis._SNAKE_NAME_RES.findall(text):
        if match.casefold() not in synthesis._META_TOKENS:
            found.append(match)
    out: list[str] = []
    for token in found:
        if token and token not in out:
            out.append(token)
    return out


def core(result: SynthesisResult) -> tuple:
    """The deterministic result: every field except the diagnostic notes."""
    return (
        result.root_cause,
        result.affected_files,
        result.affected_functions,
        result.suspicious_commit,
        result.supporting_evidence,
        result.conflicting_evidence,
        result.confidence,
        result.confidence_reasons,
    )


def without_llm_notes(result: SynthesisResult) -> list[str]:
    return [note for note in result.uncertainty if "LLM" not in note]


def conflict_locations(result: SynthesisResult) -> list[str]:
    return [ev.location for ev in result.conflicting_evidence]


def has_conflict_note(result: SynthesisResult, label: str) -> bool:
    return any(label in note for note in result.uncertainty)


# ===========================================================================
# 1. Successful correlation (all four sources aligned)
# ===========================================================================

class TestSuccessfulCorrelation:
    def _result(self) -> SynthesisResult:
        return run(**all_findings())

    def test_root_cause_non_empty_and_labelled_hypothesis(self):
        result = self._result()
        assert result.root_cause.strip()
        assert "hypothesis" in result.root_cause.lower()
        assert result.used_llm is False

    def test_root_cause_names_field_error_and_endpoint(self):
        root_cause = self._result().root_cause
        assert "surcharge" in root_cause
        assert "TypeError" in root_cause
        assert "POST /price" in root_cause

    def test_derived_commit_files_and_functions(self):
        result = self._result()
        assert result.suspicious_commit == "deadbee"
        assert result.affected_files == ["app/pricing.py"]
        assert result.affected_functions == ["PricingService.apply_surcharge"]

    def test_everything_named_in_root_cause_is_traceable(self):
        result = self._result()
        blob = supporting_text(result)
        tokens = named_tokens(result.root_cause)
        assert tokens
        for token in tokens:
            probe = token.replace("\\", "/").casefold()
            assert probe in blob, (
                f"{token!r} appears in root_cause but in no supporting "
                "evidence location or observation"
            )

    def test_supporting_and_conflicting_disjoint_and_subsets_of_collect(self):
        result = self._result()
        produced = collect(**all_findings())
        assert result.supporting_evidence
        for ev in result.supporting_evidence:
            assert ev in produced
        for ev in result.conflicting_evidence:
            assert ev in produced
        for ev in result.supporting_evidence:
            assert ev not in result.conflicting_evidence

    def test_confidence_at_least_point_seven_and_in_range(self):
        result = self._result()
        assert 0.7 <= result.confidence <= 1.0

    def test_confidence_never_claims_certainty(self):
        result = self._result()
        assert result.confidence <= 0.90
        assert result.confidence < 1.0
        assert any("Capped at 0.90" in r for r in result.confidence_reasons)

    def test_hypothesis_prefers_the_stack_trace_frame(self):
        code = make_code_findings(
            observations=[
                # A field mention in an unrelated helper, listed first.
                make_code_observation(
                    file="app/helpers.py",
                    function="Util.normalize",
                    start_line=3,
                    end_line=9,
                    observation=(
                        "Field 'surcharge' is referenced in 'Util.normalize'. "
                        "No null-safety guard was found for this value in this "
                        "function."
                    ),
                    relevance="'surcharge' appears in the error signature.",
                ),
                # The crashing frame, recorded by the code investigator.
                make_code_observation(
                    file="app/pricing.py",
                    function="PricingService.apply_surcharge",
                    observation=(
                        "Function 'PricingService.apply_surcharge' is present "
                        "in the stack trace. It spans lines 40-50."
                    ),
                    relevance=(
                        "Function 'PricingService.apply_surcharge' appears in "
                        "the stack trace at line 45."
                    ),
                ),
                # The same crashing function using the field unguarded.
                make_code_observation(
                    function="PricingService.apply_surcharge",
                    observation=(
                        "Field 'surcharge' is referenced in "
                        "'PricingService.apply_surcharge'. No null-safety "
                        "guard was found for this value in this function."
                    ),
                ),
            ],
            affected_files=["app/helpers.py", "app/pricing.py"],
            affected_functions=[
                "Util.normalize",
                "PricingService.apply_surcharge",
            ],
        )
        result = run(**all_findings(code_findings=code))
        assert "The failure surfaces in PricingService.apply_surcharge" in (
            result.root_cause
        )
        assert result.affected_functions[0] == "PricingService.apply_surcharge"
        assert result.affected_files[0] == "app/pricing.py"

    def test_confidence_reasons_and_uncertainty_non_empty(self):
        result = self._result()
        assert result.confidence_reasons
        assert all(isinstance(r, str) and r.strip()
                   for r in result.confidence_reasons)
        assert result.uncertainty
        assert all(isinstance(u, str) and u.strip()
                   for u in result.uncertainty)

    def test_all_four_sources_support_the_hypothesis(self):
        result = self._result()
        sources = {ev.source for ev in result.supporting_evidence}
        assert sources == {"log", "git", "code", "test"}


# ===========================================================================
# 2. Insufficient evidence
# ===========================================================================

class TestInsufficientEvidence:
    def test_log_only_stays_below_half_and_caps_single_source(self):
        result = run(log_findings=make_log_findings())
        assert 0.0 <= result.confidence < 0.5
        assert result.confidence <= 0.45

    def test_log_only_root_cause_states_insufficient(self):
        result = run(log_findings=make_log_findings())
        assert "insufficient" in result.root_cause.lower()
        assert "hypothesis" in result.root_cause.lower()
        assert result.suspicious_commit is None
        assert result.affected_files == []
        assert result.affected_functions == []

    def test_log_only_uncertainty_non_empty(self):
        result = run(log_findings=make_log_findings())
        assert result.uncertainty
        assert all(isinstance(u, str) for u in result.uncertainty)

    def test_run_with_all_none_arguments_does_not_raise(self):
        result = run()
        assert isinstance(result, SynthesisResult)
        assert 0.0 <= result.confidence < 0.5
        assert "insufficient" in result.root_cause.lower()
        assert result.uncertainty
        assert result.supporting_evidence == []
        assert result.conflicting_evidence == []
        assert result.suspicious_commit is None
        assert result.used_llm is False

    def test_findings_with_empty_lists_never_raise(self):
        result = run(
            log_findings=make_log_findings(
                dominant_error_signature=None,
                first_error_time=None,
                affected_endpoints=[],
                failing_requests=[],
                suspicious_input_patterns=[],
                stack_trace_present=False,
                stack_trace=None,
                recurrence_count=1,
            ),
            git_findings=make_git_findings(suspicious_commits=[]),
            code_findings=make_code_findings(
                observations=[], affected_files=[], affected_functions=[]
            ),
            test_findings=make_test_findings(
                all_relevant_tests=[], missing_scenarios=[]
            ),
        )
        assert isinstance(result, SynthesisResult)
        assert 0.0 <= result.confidence < 0.5
        assert "insufficient" in result.root_cause.lower()
        assert result.uncertainty


# ===========================================================================
# 3. Conflict classes (a), (b), (c) + comparison
# ===========================================================================

class TestConflictClasses:
    def _aligned(self) -> SynthesisResult:
        return run(**all_findings())

    def _coverage_conflict(self) -> SynthesisResult:
        return run(**all_findings(
            test_findings=make_test_findings(all_relevant_tests=[make_test_case(
                field_values_tested={"surcharge": ["None"]},
            )]),
        ))

    def _temporal_conflict(self) -> SynthesisResult:
        return run(**all_findings(
            log_findings=make_log_findings(
                first_error_time=T_FIRST_ERR_BEFORE_COMMIT
            ),
        ))

    def _scope_conflict(self) -> SynthesisResult:
        return run(**all_findings(git_findings=make_git_findings(
            suspicious_commits=[make_commit(
                changed_files=["docs/runbook.md"],
                relevance_reasons=[
                    "Commit is within 24h before the incident timestamp.",
                ],
            )],
        )))

    def test_coverage_conflict_moves_evidence_and_lowers_confidence(self):
        aligned = self._aligned()
        conflicted = self._coverage_conflict()
        assert "tests/test_pricing.py:10" in conflict_locations(conflicted)
        assert has_conflict_note(conflicted, "Conflict (a)")
        assert conflicted.confidence < aligned.confidence

    def test_temporal_conflict_moves_evidence_and_lowers_confidence(self):
        aligned = self._aligned()
        conflicted = self._temporal_conflict()
        assert "commit deadbee" in conflict_locations(conflicted)
        assert has_conflict_note(conflicted, "Conflict (b)")
        assert conflicted.confidence < aligned.confidence

    def test_scope_conflict_moves_evidence_and_lowers_confidence(self):
        aligned = self._aligned()
        conflicted = self._scope_conflict()
        assert "commit deadbee" in conflict_locations(conflicted)
        assert has_conflict_note(conflicted, "Conflict (c)")
        assert conflicted.confidence < aligned.confidence

    def test_temporal_and_scope_conflicts_keep_commit_out_of_root_cause(self):
        for conflicted in (self._temporal_conflict(), self._scope_conflict()):
            assert "deadbee" not in conflicted.root_cause
            assert conflicted.root_cause.strip()

    def test_every_conflict_scores_below_aligned_baseline(self):
        aligned = self._aligned()
        conflicted = [
            self._coverage_conflict(),
            self._temporal_conflict(),
            self._scope_conflict(),
        ]
        for result in conflicted:
            assert 0.0 <= result.confidence <= 1.0
            assert result.confidence < aligned.confidence
            assert result.uncertainty


# ===========================================================================
# 4. Anti-hardcoding: derived, never recalled
# ===========================================================================

class TestAntiHardcoding:
    def test_renamed_incident_changes_root_cause(self):
        root_cause = run(**renamed_findings()).root_cause
        assert "levy" in root_cause
        assert "PATCH /billing" in root_cause
        assert "TypeError" in root_cause
        for stale in ("surcharge", "POST /price", "deadbee"):
            assert stale not in root_cause

    def test_renamed_incident_changes_derived_fields(self):
        result = run(**renamed_findings())
        assert result.suspicious_commit == "f00d1ed"
        assert result.affected_files == ["app/billing.py"]
        assert result.affected_functions == ["TaxEngine.add_levy"]
        assert 0.7 <= result.confidence <= 1.0

    def test_original_fixture_never_leaks_renamed_values(self):
        result = run(**all_findings())
        assert "surcharge" in result.root_cause
        assert "deadbee" in result.root_cause
        for fresh in ("levy", "f00d1ed", "PATCH /billing", "app/billing.py"):
            assert fresh not in result.root_cause
            assert fresh not in " ".join(result.affected_files)
            assert fresh not in " ".join(result.affected_functions)


# ===========================================================================
# 5. LLM modularity (llm_backend is optional and never inflates confidence)
# ===========================================================================

class TestLLMModularity:
    def _baseline(self) -> SynthesisResult:
        return run(**all_findings())

    @staticmethod
    def _traceable_backend(evidence, context):
        return {
            "root_cause": (
                "Hypothesis: app/pricing.py PricingService.apply_surcharge "
                "mishandles the surcharge field and commit deadbee correlates."
            ),
            "affected_files": ["app/pricing.py"],
            "affected_functions": ["PricingService.apply_surcharge"],
            "suspicious_commit": "deadbee",
        }

    @staticmethod
    def _raising_backend(evidence, context):
        raise RuntimeError("backend offline")

    @staticmethod
    def _untraceable_backend(evidence, context):
        return {
            "root_cause": "Hypothesis: src/never/seen.py caused the failure"
        }

    def test_valid_dict_is_adopted(self):
        result = run(**all_findings(), llm_backend=self._traceable_backend)
        assert result.used_llm is True
        assert result.root_cause == (
            "Hypothesis: app/pricing.py PricingService.apply_surcharge "
            "mishandles the surcharge field and commit deadbee correlates."
        )
        assert result.affected_files == ["app/pricing.py"]
        assert result.affected_functions == ["PricingService.apply_surcharge"]
        assert result.suspicious_commit == "deadbee"

    def test_adoption_does_not_change_confidence(self):
        baseline = self._baseline()
        adopted = run(**all_findings(), llm_backend=self._traceable_backend)
        assert adopted.confidence == baseline.confidence
        assert adopted.confidence_reasons == baseline.confidence_reasons
        assert adopted.supporting_evidence == baseline.supporting_evidence

    def test_hypothesis_prefix_added_when_backend_omits_it(self):
        def backend(evidence, context):
            return {
                "root_cause": "the field 'surcharge' in app/pricing.py failed",
            }

        result = run(**all_findings(), llm_backend=backend)
        assert result.used_llm is True
        assert result.root_cause.startswith("Hypothesis: ")
        assert "app/pricing.py" in result.root_cause

    def test_failing_backends_fall_back_to_deterministic_result(self):
        baseline = self._baseline()
        backends = [
            self._raising_backend,
            lambda evidence, context: None,
            lambda evidence, context: "not a dict",
            42,
        ]
        for backend in backends:
            result = run(**all_findings(), llm_backend=backend)
            assert result.used_llm is False, backend
            assert core(result) == core(baseline), backend
            assert without_llm_notes(result) == baseline.uncertainty, backend
            assert any("LLM" in note for note in result.uncertainty), backend

    def test_untraceable_root_cause_falls_back(self):
        baseline = self._baseline()
        result = run(**all_findings(), llm_backend=self._untraceable_backend)
        assert result.used_llm is False
        assert core(result) == core(baseline)
        assert any("absent from supporting evidence" in note
                   for note in result.uncertainty)

    def test_untraceable_claims_are_dropped_while_root_cause_adopts(self):
        def backend(evidence, context):
            return {
                "root_cause": (
                    "Hypothesis: the field 'surcharge' in app/pricing.py failed"
                ),
                "affected_files": ["app/pricing.py", "src/ghost/module.py"],
                "affected_functions": ["PricingService.apply_surcharge"],
                "suspicious_commit": "faced1e",
            }

        result = run(**all_findings(), llm_backend=backend)
        assert result.used_llm is True
        assert result.affected_files == ["app/pricing.py"]
        assert result.suspicious_commit == "deadbee"
        assert any("affected_files" in note for note in result.uncertainty)
        assert any("suspicious_commit" in note for note in result.uncertainty)

    def test_backend_not_consulted_when_evidence_insufficient(self):
        result = run(
            log_findings=make_log_findings(),
            llm_backend=self._traceable_backend,
        )
        assert result.used_llm is False
        assert "insufficient" in result.root_cause.lower()
        assert any("not consulted" in note for note in result.uncertainty)


# ===========================================================================
# 6. Robustness / determinism
# ===========================================================================

class TestRobustness:
    def test_bare_findings_object_missing_optional_attributes(self):
        class _BareLogFindings:
            dominant_error_signature = "ValueError: boom"

        result = run(log_findings=_BareLogFindings())
        assert isinstance(result, SynthesisResult)
        assert 0.0 <= result.confidence < 0.5
        assert result.uncertainty

    def test_none_values_for_optional_fields_never_raise(self):
        result = run(
            log_findings=make_log_findings(
                dominant_error_signature=None,
                first_error_time=None,
                stack_trace=None,
            ),
            git_findings=make_git_findings(top_suspect=None),
            code_findings=make_code_findings(),
            test_findings=make_test_findings(
                file_findings=None, interpretation_hints=None
            ),
        )
        assert isinstance(result, SynthesisResult)
        assert 0.0 <= result.confidence <= 1.0
        assert "hypothesis" in result.root_cause.lower()

    def test_repeated_runs_are_identical(self):
        first = run(**all_findings())
        second = run(**all_findings())
        assert first == second
        assert first.root_cause == second.root_cause
        assert first.confidence == second.confidence
        assert first.confidence_reasons == second.confidence_reasons
        assert [ev.model_dump() for ev in first.supporting_evidence] == \
            [ev.model_dump() for ev in second.supporting_evidence]

    def test_non_evidence_entries_are_dropped_with_a_note(self):
        result = run(log_findings=make_log_findings(),
                     evidence=["junk", 42, None])
        assert result.supporting_evidence == []
        assert result.conflicting_evidence == []
        assert result.confidence == 0.0
        assert any("skipped" in note for note in result.uncertainty)
        assert result.uncertainty

    def test_empty_evidence_argument_returns_low_confidence_result(self):
        result = run(evidence=[])
        assert isinstance(result, SynthesisResult)
        assert result.confidence < 0.5
        assert "insufficient" in result.root_cause.lower()
        assert result.uncertainty

    def test_structural_evidence_mistake_raises(self):
        with pytest.raises(SynthesisAgentError):
            run(evidence="not an iterable of evidence")
        with pytest.raises(SynthesisAgentError):
            run(evidence=42)


# ===========================================================================
# Integration — demo incident
# ===========================================================================

@pytest.mark.skipif(not DEMO_AVAILABLE, reason="incident-replay-demo not present")
class TestSynthesisWithDemoIncident:
    """
    End-to-end integration: run the four real investigators against the
    demo project and verify the synthesizer correlates them into one
    hypothesis without hardcoding any commit sha or function name.
    """

    def _result(self) -> SynthesisResult:
        stack = (
            DEMO_STACK_TRACE.read_text(encoding="utf-8")
            if DEMO_STACK_TRACE.is_file()
            else None
        )
        log_findings = run_log_agent(str(DEMO_LOG), stack_trace=stack)
        git_findings = run_git_agent(
            str(DEMO_REPO),
            incident_time=log_findings.first_error_time,
            error_keywords=["discount", "nonetype"],
            affected_files=["app/checkout.py", "app/api.py"],
            time_window_hours=24,
        )
        code_findings = run_code_agent(
            str(DEMO_REPO),
            stack_trace=stack,
            error_keywords=["discount", "nonetype"],
            changed_files=["app/checkout.py", "app/api.py"],
            error_type="TypeError",
        )
        test_findings = run_test_agent(
            str(DEMO_TESTS),
            affected_functions=code_findings.affected_functions,
            error_keywords=["discount", "nonetype"],
            incident_field_values={"discount": "None"},
        )
        return run(
            log_findings=log_findings,
            git_findings=git_findings,
            code_findings=code_findings,
            test_findings=test_findings,
        )

    def test_suspicious_commit_is_correlated(self):
        result = self._result()
        assert isinstance(result, SynthesisResult)
        assert result.suspicious_commit is not None
        assert result.suspicious_commit != ""

    def test_checkout_file_is_affected(self):
        assert "app/checkout.py" in self._result().affected_files

    def test_calculate_discount_function_is_affected(self):
        functions = self._result().affected_functions
        assert any(fn.endswith("calculate_discount") for fn in functions)

    def test_root_cause_mentions_null_discount(self):
        root_cause = self._result().root_cause.lower()
        assert "discount" in root_cause
        assert "null" in root_cause or "none" in root_cause
