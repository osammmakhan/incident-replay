"""
Transparent confidence calculation for root-cause hypotheses.

Purpose
-------
Given a set of evidence-quality signals collected from four investigator
agents (log, git, code, test), return a float in ``[0.0, 1.0]`` that
reflects how well the evidence *corroborates* the hypothesis — not how
certain the analyst *feels*.

Design principles
-----------------
Explainability first
    Every point added or subtracted is logged in a human-readable
    ``reasons`` list so the caller can show exactly why the score is what
    it is.

Corroboration, not completeness
    A high score requires *independent* sources to agree.  A single very
    detailed source still caps out below the moderate band (< 0.50).

No false precision
    The score is rounded to two decimal places.  The rubric uses integer
    percentages (e.g. 0.10, 0.05) to make the arithmetic obvious in the
    reasons text.

Hard ceiling
    No hypothesis reaches 1.0.  Correlation is not confirmation — the
    verification stage (running the regression test) would be needed for
    that.  The ceiling is set at ``MAX_CONFIDENCE = 0.90``.

Scoring rubric (in order of application)
-----------------------------------------
1.  **Base score** — from the number of *distinct* sources with supporting
    evidence (log / git / code / test):

    ======  =====
    count   base
    ======  =====
    0       0.00
    1       0.35
    2       0.55
    3       0.70
    ≥ 4     0.80
    ======  =====

2.  **Bonuses** (additive, applied after the base):

    * ``+0.10`` — the suspicious commit touches an affected file *and* is
      matched by a code observation (commit corroborated by code).
    * ``+0.05`` — the log records a dominant error signature (a recurring
      error pattern was detected, not just isolated noise).
    * ``+0.05`` — the test findings record a coverage gap for the incident
      scenario (the missing test is indirect evidence the code path was not
      guarded).
    * ``+0.05`` — a code observation reports the field used without a
      null-safety guard (unguarded-field pattern detected).

3.  **Cap at MAX_CONFIDENCE** before penalties (correlation ≠ certainty).

4.  **Penalties** (subtractive, applied after the cap):

    * ``-0.15`` per detected conflict class.  Conflict classes represent
      evidence that *contradicts* the hypothesis (e.g. existing tests
      already cover the incident value, or the error precedes the change).

5.  **Single-source cap**: when fewer than two distinct sources contribute,
    the score is further capped at ``CAP_SINGLE_SOURCE = 0.45`` regardless
    of bonuses, so a single rich source never crosses the moderate band.

6.  **Zero-source floor**: when no source contributes, the score is fixed
    at ``0.00``.

7.  **Final clamp**: ``max(0.0, min(MAX_CONFIDENCE, score))``, rounded to
    two decimal places.

Usage
-----
::

    from incident_replay.analysis.confidence import ConfidenceInput, calculate

    result = calculate(ConfidenceInput(
        supporting_sources=["log", "git", "code"],
        has_log_signature=True,
        has_coverage_gap=True,
        has_unguarded_field=False,
        commit_corroborated=True,
        conflict_classes=[],
    ))
    print(result.score)    # e.g. 0.90
    print(result.reasons)  # human-readable list
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Rubric constants (mirrors synthesis_agent for consistency)
# ---------------------------------------------------------------------------

BASE_BY_SOURCE_COUNT: dict[int, float] = {0: 0.00, 1: 0.35, 2: 0.55, 3: 0.70, 4: 0.80}

BONUS_COMMIT_CORROBORATED: float = 0.10
BONUS_ERROR_SIGNATURE: float = 0.05
BONUS_COVERAGE_GAP: float = 0.05
BONUS_UNGUARDED_FIELD: float = 0.05

PENALTY_PER_CONFLICT: float = 0.15

# Single source: high detail in one log or one commit does not constitute
# multi-source corroboration.
CAP_SINGLE_SOURCE: float = 0.45

# Correlation alone is never certainty; the regression-test run (verification
# stage) would be required to push above this ceiling.
MAX_CONFIDENCE: float = 0.90

BAND_HIGH: float = 0.75
BAND_MODERATE: float = 0.50
BAND_LOW: float = 0.25


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceInput:
    """
    Evidence-quality signals used to calculate hypothesis confidence.

    All four investigator sources (log, git, code, test) contribute
    independently; the caller populates whichever are available.

    Parameters
    ----------
    supporting_sources:
        Distinct source labels — a subset of ``{"log", "git", "code",
        "test"}`` — for which at least one piece of supporting (non-
        conflicting) evidence exists.  Duplicates are ignored.
    has_log_signature:
        ``True`` when the log agent recorded a dominant error signature
        (a recurring pattern, not isolated noise).
    has_coverage_gap:
        ``True`` when the test agent recorded a coverage gap for the
        incident scenario (the missing test is indirect evidence the code
        path was not guarded against the incident value).
    has_unguarded_field:
        ``True`` when the code agent observed the incident field being
        used without a null-safety guard.
    commit_corroborated:
        ``True`` when the suspicious commit both touches an affected file
        *and* is matched by a code observation — i.e. two independent
        checks agree on the same commit.
    conflict_classes:
        List of detected conflict-class identifiers.  Each class
        represents evidence that contradicts the hypothesis (e.g.
        ``"a-tests-cover-incident-value"``).  Each entry incurs a
        ``PENALTY_PER_CONFLICT`` deduction.
    """

    supporting_sources: list[str] = field(default_factory=list)
    has_log_signature: bool = False
    has_coverage_gap: bool = False
    has_unguarded_field: bool = False
    commit_corroborated: bool = False
    conflict_classes: list[str] = field(default_factory=list)


@dataclass
class ConfidenceResult:
    """
    Output of :func:`calculate`.

    Parameters
    ----------
    score:
        Float in ``[0.0, MAX_CONFIDENCE]``, rounded to two decimal
        places.  Do not treat the two-digit precision as measurement
        accuracy — differences smaller than a bonus/penalty step
        (0.05) are not meaningful.
    reasons:
        Ordered list of human-readable strings documenting each
        adjustment made to the score.  The final entry always states the
        band (low / moderate / high).
    band:
        One of ``"high"``, ``"moderate"``, ``"low"``, or ``"none"``
        describing the confidence tier.
    """

    score: float
    reasons: list[str]
    band: str


# ---------------------------------------------------------------------------
# Band helper
# ---------------------------------------------------------------------------

def _band(score: float) -> str:
    """Return the confidence tier label for *score*."""
    if score >= BAND_HIGH:
        return "high"
    if score >= BAND_MODERATE:
        return "moderate"
    if score >= BAND_LOW:
        return "low"
    return "none"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate(inputs: ConfidenceInput) -> ConfidenceResult:
    """
    Apply the documented rubric to *inputs* and return a
    :class:`ConfidenceResult`.

    The calculation is a pure function of the inputs — no randomness,
    no LLM calls.  The same inputs always produce the same score.
    """
    reasons: list[str] = []

    # Deduplicate sources while preserving order so the reasons text is
    # deterministic regardless of how the caller assembled the list.
    seen: set[str] = set()
    unique_sources: list[str] = []
    for src in (inputs.supporting_sources or []):
        if src not in seen:
            seen.add(src)
            unique_sources.append(src)

    count = len(unique_sources)
    base = BASE_BY_SOURCE_COUNT.get(min(count, 4), 0.80)
    labels = ", ".join(unique_sources) if unique_sources else "none"
    reasons.append(
        f"Base {base:.2f} from {count} distinct supporting source(s): {labels}."
    )

    score = base

    if inputs.commit_corroborated:
        score += BONUS_COMMIT_CORROBORATED
        reasons.append(
            f"+{BONUS_COMMIT_CORROBORATED:.2f}: the suspicious commit touches "
            "an affected file and is matched by a code observation."
        )
    if inputs.has_log_signature:
        score += BONUS_ERROR_SIGNATURE
        reasons.append(
            f"+{BONUS_ERROR_SIGNATURE:.2f}: the log records a dominant error "
            "signature."
        )
    if inputs.has_coverage_gap:
        score += BONUS_COVERAGE_GAP
        reasons.append(
            f"+{BONUS_COVERAGE_GAP:.2f}: the test findings record a coverage "
            "gap for the incident scenario."
        )
    if inputs.has_unguarded_field:
        score += BONUS_UNGUARDED_FIELD
        reasons.append(
            f"+{BONUS_UNGUARDED_FIELD:.2f}: a code observation reports the "
            "incident field used without a null-safety guard."
        )

    # Cap before penalties: correlation alone cannot reach certainty.
    if score > MAX_CONFIDENCE:
        score = MAX_CONFIDENCE
        reasons.append(
            f"Capped at {MAX_CONFIDENCE:.2f}: correlation alone is not "
            "certainty — the hypothesis has not been confirmed by the "
            "verification stage."
        )

    # Penalties for conflicting evidence.
    for class_id in (inputs.conflict_classes or []):
        score -= PENALTY_PER_CONFLICT
        reasons.append(
            f"-{PENALTY_PER_CONFLICT:.2f}: conflicting evidence detected "
            f"(conflict class {class_id!r})."
        )

    # Single-source cap: one rich source does not constitute corroboration.
    if count == 0:
        score = 0.0
        reasons.append("No supporting evidence; confidence fixed at 0.00.")
    elif count < 2:
        if score > CAP_SINGLE_SOURCE:
            score = CAP_SINGLE_SOURCE
            reasons.append(
                f"Capped at {CAP_SINGLE_SOURCE:.2f}: fewer than two distinct "
                "sources support the hypothesis, so confidence must stay "
                "below the moderate band."
            )
        else:
            reasons.append(
                f"Only {count} distinct source(s) support the hypothesis; "
                f"confidence is already below the moderate-band cap "
                f"({CAP_SINGLE_SOURCE:.2f})."
            )

    score = round(max(0.0, min(MAX_CONFIDENCE, score)), 2)
    tier = _band(score)
    reasons.append(f"Final confidence {score:.2f} (band: {tier}).")
    return ConfidenceResult(score=score, reasons=reasons, band=tier)
