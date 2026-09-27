"""
Tests for incident_replay.analysis.confidence.

Every test is self-contained and works against the public API only:
``ConfidenceInput``, ``ConfidenceResult``, and ``calculate``.

Coverage:
- Score is a float in [0.0, MAX_CONFIDENCE]
- Score is rounded to two decimal places (no false precision)
- Base score table (0 – 4+ sources)
- Each bonus individually (+commit, +signature, +gap, +unguarded)
- Bonuses do not stack past MAX_CONFIDENCE
- Each conflict-class penalty individually
- Single-source cap (< 2 sources → score ≤ CAP_SINGLE_SOURCE)
- Zero-source floor (score == 0.00)
- Duplicate sources in input are deduplicated
- Band labels: none / low / moderate / high
- Reasons list is non-empty and final entry mentions the score
- Determinism: same inputs → same result on two calls
- ``IncidentReport.confidence`` field accepts the returned score (contract)
"""

from __future__ import annotations

import pytest

from incident_replay.analysis.confidence import (
    BAND_HIGH,
    BAND_LOW,
    BAND_MODERATE,
    BONUS_COMMIT_CORROBORATED,
    BONUS_COVERAGE_GAP,
    BONUS_ERROR_SIGNATURE,
    BONUS_UNGUARDED_FIELD,
    CAP_SINGLE_SOURCE,
    MAX_CONFIDENCE,
    PENALTY_PER_CONFLICT,
    ConfidenceInput,
    ConfidenceResult,
    calculate,
)
from incident_replay.models.schemas import IncidentReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(**kwargs) -> ConfidenceInput:
    """Build a ConfidenceInput with sensible defaults for unspecified fields."""
    defaults: dict = dict(
        supporting_sources=[],
        has_log_signature=False,
        has_coverage_gap=False,
        has_unguarded_field=False,
        commit_corroborated=False,
        conflict_classes=[],
    )
    defaults.update(kwargs)
    return ConfidenceInput(**defaults)


def _calc(**kwargs) -> ConfidenceResult:
    return calculate(_make(**kwargs))


# ---------------------------------------------------------------------------
# Return-type contract
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_confidence_result(self):
        result = _calc()
        assert isinstance(result, ConfidenceResult)

    def test_score_is_float(self):
        result = _calc(supporting_sources=["log", "git"])
        assert isinstance(result.score, float)

    def test_score_in_range(self):
        result = _calc(
            supporting_sources=["log", "git", "code", "test"],
            has_log_signature=True,
            has_coverage_gap=True,
            has_unguarded_field=True,
            commit_corroborated=True,
        )
        assert 0.0 <= result.score <= MAX_CONFIDENCE

    def test_score_two_decimal_places(self):
        # Round-trip through str — if the value already has ≤ 2 decimal
        # digits the string representation is stable.
        result = _calc(supporting_sources=["log", "git"])
        assert result.score == round(result.score, 2)

    def test_reasons_is_non_empty_list(self):
        result = _calc()
        assert isinstance(result.reasons, list)
        assert len(result.reasons) >= 1

    def test_final_reason_mentions_score(self):
        result = _calc(supporting_sources=["log", "git"])
        final = result.reasons[-1]
        assert f"{result.score:.2f}" in final

    def test_band_is_string(self):
        result = _calc()
        assert isinstance(result.band, str)

    def test_band_values_are_valid(self):
        for sources in [[], ["log"], ["log", "git"], ["log", "git", "code", "test"]]:
            r = _calc(supporting_sources=sources)
            assert r.band in {"none", "low", "moderate", "high"}


# ---------------------------------------------------------------------------
# Base score table
# ---------------------------------------------------------------------------

class TestBaseScore:
    def test_zero_sources_base_is_zero(self):
        r = _calc(supporting_sources=[])
        assert r.score == 0.00

    def test_one_source_base_is_0_35(self):
        # With one source the base is 0.35, which is below CAP_SINGLE_SOURCE,
        # so no cap message; final score == 0.35.
        r = _calc(supporting_sources=["log"])
        assert r.score == 0.35

    def test_two_sources_base_is_0_55(self):
        r = _calc(supporting_sources=["log", "git"])
        assert r.score == 0.55

    def test_three_sources_base_is_0_70(self):
        r = _calc(supporting_sources=["log", "git", "code"])
        assert r.score == 0.70

    def test_four_sources_base_is_0_80(self):
        r = _calc(supporting_sources=["log", "git", "code", "test"])
        assert r.score == 0.80

    def test_base_reason_mentions_source_count(self):
        r = _calc(supporting_sources=["log", "git"])
        assert "2 distinct supporting source(s)" in r.reasons[0]

    def test_base_reason_lists_sources(self):
        r = _calc(supporting_sources=["log", "git"])
        assert "log" in r.reasons[0]
        assert "git" in r.reasons[0]


# ---------------------------------------------------------------------------
# Bonus: commit corroborated
# ---------------------------------------------------------------------------

class TestBonusCommitCorroborated:
    def test_bonus_is_added(self):
        without = _calc(supporting_sources=["log", "git"]).score
        with_bonus = _calc(
            supporting_sources=["log", "git"],
            commit_corroborated=True,
        ).score
        assert with_bonus == round(without + BONUS_COMMIT_CORROBORATED, 2)

    def test_reason_mentions_commit_bonus(self):
        r = _calc(supporting_sources=["log", "git"], commit_corroborated=True)
        assert any("commit touches" in reason for reason in r.reasons)


# ---------------------------------------------------------------------------
# Bonus: log error signature
# ---------------------------------------------------------------------------

class TestBonusErrorSignature:
    def test_bonus_is_added(self):
        without = _calc(supporting_sources=["log", "git"]).score
        with_bonus = _calc(
            supporting_sources=["log", "git"],
            has_log_signature=True,
        ).score
        assert with_bonus == round(without + BONUS_ERROR_SIGNATURE, 2)

    def test_reason_mentions_signature(self):
        r = _calc(supporting_sources=["log", "git"], has_log_signature=True)
        assert any("dominant error" in reason for reason in r.reasons)


# ---------------------------------------------------------------------------
# Bonus: coverage gap
# ---------------------------------------------------------------------------

class TestBonusCoverageGap:
    def test_bonus_is_added(self):
        without = _calc(supporting_sources=["log", "git"]).score
        with_bonus = _calc(
            supporting_sources=["log", "git"],
            has_coverage_gap=True,
        ).score
        assert with_bonus == round(without + BONUS_COVERAGE_GAP, 2)

    def test_reason_mentions_gap(self):
        r = _calc(supporting_sources=["log", "git"], has_coverage_gap=True)
        assert any("coverage gap" in reason for reason in r.reasons)


# ---------------------------------------------------------------------------
# Bonus: unguarded field
# ---------------------------------------------------------------------------

class TestBonusUnguardedField:
    def test_bonus_is_added(self):
        without = _calc(supporting_sources=["log", "git"]).score
        with_bonus = _calc(
            supporting_sources=["log", "git"],
            has_unguarded_field=True,
        ).score
        assert with_bonus == round(without + BONUS_UNGUARDED_FIELD, 2)

    def test_reason_mentions_null_safety(self):
        r = _calc(supporting_sources=["log", "git"], has_unguarded_field=True)
        assert any("null-safety" in reason for reason in r.reasons)


# ---------------------------------------------------------------------------
# MAX_CONFIDENCE ceiling
# ---------------------------------------------------------------------------

class TestMaxConfidenceCeiling:
    def test_score_never_exceeds_max(self):
        r = _calc(
            supporting_sources=["log", "git", "code", "test"],
            has_log_signature=True,
            has_coverage_gap=True,
            has_unguarded_field=True,
            commit_corroborated=True,
        )
        assert r.score <= MAX_CONFIDENCE

    def test_score_equals_max_when_all_bonuses_fire(self):
        # 4 sources → base 0.80; all bonuses add 0.25 → 1.05, capped at 0.90
        r = _calc(
            supporting_sources=["log", "git", "code", "test"],
            has_log_signature=True,
            has_coverage_gap=True,
            has_unguarded_field=True,
            commit_corroborated=True,
        )
        assert r.score == MAX_CONFIDENCE

    def test_cap_reason_mentions_correlation(self):
        r = _calc(
            supporting_sources=["log", "git", "code", "test"],
            has_log_signature=True,
            commit_corroborated=True,
        )
        # Score: 0.80 + 0.10 + 0.05 = 0.95 → capped
        assert any("correlation alone" in reason for reason in r.reasons)


# ---------------------------------------------------------------------------
# Conflict penalties
# ---------------------------------------------------------------------------

class TestConflictPenalties:
    def test_single_conflict_reduces_score(self):
        without = _calc(supporting_sources=["log", "git", "code"]).score
        with_conflict = _calc(
            supporting_sources=["log", "git", "code"],
            conflict_classes=["a-tests-cover-incident-value"],
        ).score
        assert with_conflict == round(without - PENALTY_PER_CONFLICT, 2)

    def test_two_conflicts_reduce_score_twice(self):
        without = _calc(supporting_sources=["log", "git", "code"]).score
        with_two = _calc(
            supporting_sources=["log", "git", "code"],
            conflict_classes=["a-tests-cover-incident-value", "b-error-precedes-change"],
        ).score
        assert with_two == round(without - 2 * PENALTY_PER_CONFLICT, 2)

    def test_conflict_reason_mentions_class_id(self):
        r = _calc(
            supporting_sources=["log", "git"],
            conflict_classes=["c-commit-uncorroborated"],
        )
        assert any("c-commit-uncorroborated" in reason for reason in r.reasons)

    def test_penalties_cannot_push_below_zero(self):
        r = _calc(
            supporting_sources=["log"],
            conflict_classes=["a", "b", "c", "d", "e"],
        )
        assert r.score >= 0.0

    def test_empty_conflict_list_no_penalty(self):
        base_score = _calc(supporting_sources=["log", "git"]).score
        same = _calc(supporting_sources=["log", "git"], conflict_classes=[]).score
        assert same == base_score


# ---------------------------------------------------------------------------
# Single-source cap
# ---------------------------------------------------------------------------

class TestSingleSourceCap:
    def test_one_source_with_all_bonuses_capped(self):
        # base 0.35 + all bonuses (0.25) = 0.60 → capped at 0.45
        r = _calc(
            supporting_sources=["log"],
            has_log_signature=True,
            has_coverage_gap=True,
            has_unguarded_field=True,
            commit_corroborated=True,
        )
        assert r.score == CAP_SINGLE_SOURCE

    def test_one_source_below_cap_not_further_reduced(self):
        # base 0.35, no bonuses → 0.35 which is already < CAP_SINGLE_SOURCE
        r = _calc(supporting_sources=["log"])
        assert r.score == 0.35

    def test_single_source_cap_reason_appears(self):
        r = _calc(
            supporting_sources=["git"],
            commit_corroborated=True,
            has_log_signature=True,
            has_coverage_gap=True,
            has_unguarded_field=True,
        )
        assert any(f"{CAP_SINGLE_SOURCE:.2f}" in reason for reason in r.reasons)

    def test_two_sources_not_subject_to_single_source_cap(self):
        r = _calc(supporting_sources=["log", "git"])
        assert r.score > CAP_SINGLE_SOURCE


# ---------------------------------------------------------------------------
# Zero-source floor
# ---------------------------------------------------------------------------

class TestZeroSourceFloor:
    def test_no_sources_score_is_zero(self):
        r = _calc(supporting_sources=[])
        assert r.score == 0.00

    def test_no_sources_band_is_none(self):
        r = _calc(supporting_sources=[])
        assert r.band == "none"

    def test_no_sources_reason_mentions_zero(self):
        r = _calc(supporting_sources=[])
        assert any("0.00" in reason for reason in r.reasons)

    def test_no_sources_bonuses_ignored(self):
        r = _calc(
            supporting_sources=[],
            has_log_signature=True,
            commit_corroborated=True,
        )
        assert r.score == 0.00


# ---------------------------------------------------------------------------
# Duplicate-source deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_duplicate_sources_treated_as_one(self):
        r_dup = _calc(supporting_sources=["log", "log", "log"])
        r_one = _calc(supporting_sources=["log"])
        assert r_dup.score == r_one.score

    def test_duplicate_sources_reason_shows_correct_count(self):
        r = _calc(supporting_sources=["git", "git"])
        assert "1 distinct supporting source(s)" in r.reasons[0]

    def test_mixed_duplicates_deduplicated(self):
        r = _calc(supporting_sources=["log", "git", "log", "git"])
        assert "2 distinct supporting source(s)" in r.reasons[0]


# ---------------------------------------------------------------------------
# Band labels
# ---------------------------------------------------------------------------

class TestBandLabels:
    def test_band_none_when_score_below_low(self):
        # 0 sources → 0.00
        r = _calc(supporting_sources=[])
        assert r.band == "none"

    def test_band_low_when_score_in_low_range(self):
        # 1 source → 0.35 (BAND_LOW = 0.25 ≤ 0.35 < BAND_MODERATE = 0.50)
        r = _calc(supporting_sources=["log"])
        assert r.band == "low"

    def test_band_moderate_when_score_in_moderate_range(self):
        # 2 sources → 0.55 (BAND_MODERATE = 0.50 ≤ 0.55 < BAND_HIGH = 0.75)
        r = _calc(supporting_sources=["log", "git"])
        assert r.band == "moderate"

    def test_band_high_when_score_ge_band_high(self):
        # 3 sources + commit bonus → 0.70 + 0.10 = 0.80 ≥ BAND_HIGH 0.75
        r = _calc(
            supporting_sources=["log", "git", "code"],
            commit_corroborated=True,
        )
        assert r.band == "high"

    def test_band_in_final_reason(self):
        r = _calc(supporting_sources=["log", "git"])
        assert r.band in r.reasons[-1]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_same_result(self):
        inputs = _make(
            supporting_sources=["log", "git", "code"],
            has_log_signature=True,
            commit_corroborated=True,
            conflict_classes=["a-tests-cover-incident-value"],
        )
        r1 = calculate(inputs)
        r2 = calculate(inputs)
        assert r1.score == r2.score
        assert r1.band == r2.band
        assert r1.reasons == r2.reasons

    def test_order_of_sources_does_not_change_score(self):
        r1 = _calc(supporting_sources=["log", "git", "code"])
        r2 = _calc(supporting_sources=["code", "git", "log"])
        assert r1.score == r2.score


# ---------------------------------------------------------------------------
# IncidentReport contract: score must be accepted by the schema
# ---------------------------------------------------------------------------

class TestIncidentReportContract:
    """
    Verify that a score produced by ``calculate`` can be used as the
    ``confidence`` field of an ``IncidentReport`` without Pydantic raising.
    The contract is: float, ge=0.0, le=1.0.
    We do NOT change IncidentReport — only assert that the output is valid.
    """

    _REGRESSION_TEST = {
        "name": "test_regression",
        "code": "def test_regression(): pass",
        "language": "python",
    }
    _SUGGESTED_FIX = {"description": "Fix it.", "patch": ""}
    _VERIFICATION = {"passed": True, "output": ""}

    def _build_report(self, score: float) -> IncidentReport:
        return IncidentReport(
            incident_summary="Demo incident",
            timeline=[],
            root_cause="Hypothesis: field missing guard.",
            evidence=[],
            affected_files=["app/pricing.py"],
            affected_functions=["calculate_price"],
            suspicious_commit="abc1234",
            confidence=score,
            regression_test=self._REGRESSION_TEST,
            suggested_fix=self._SUGGESTED_FIX,
            verification_result=self._VERIFICATION,
        )

    def test_zero_source_score_is_valid(self):
        r = _calc(supporting_sources=[])
        report = self._build_report(r.score)
        assert report.confidence == 0.00

    def test_single_source_score_is_valid(self):
        r = _calc(supporting_sources=["log"])
        report = self._build_report(r.score)
        assert 0.0 <= report.confidence <= 1.0

    def test_four_source_max_score_is_valid(self):
        r = _calc(
            supporting_sources=["log", "git", "code", "test"],
            has_log_signature=True,
            commit_corroborated=True,
        )
        report = self._build_report(r.score)
        assert 0.0 <= report.confidence <= 1.0

    def test_score_never_exceeds_one(self):
        for n in range(5):
            sources = ["log", "git", "code", "test"][:n]
            r = _calc(
                supporting_sources=sources,
                has_log_signature=True,
                has_coverage_gap=True,
                has_unguarded_field=True,
                commit_corroborated=True,
            )
            assert r.score <= 1.0, f"score {r.score} exceeds 1.0 for {n} sources"
